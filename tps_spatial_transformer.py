import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TPSSpatialTransformer(nn.Module):
    def __init__(self, output_image_size=None, num_control_points=None, margins=None):
        # margins: (x, y)  x, y in [0, 1)
        super(TPSSpatialTransformer, self).__init__()
        self.output_image_size = output_image_size
        self.num_control_points = num_control_points
        self.margins = margins

        # self.source_ctrl_points = torch.Tensor([
        #     [-1, -1],
        #     [-0.5, -1],
        #     [0, -1],
        #     [0.5, -1],
        #     [1, -1],
        #     [-1, 1],
        #     [-0.5, 1],
        #     [0, 1],
        #     [0.5, 1],
        #     [1, 1]])
        self.source_ctrl_points = self.build_output_control_points(num_control_points, margins)

    def build_output_control_points(self, num_control_points, margins):
        margin_x, margin_y = margins
        num_ctrl_pts_per_side = num_control_points // 2
        ctrl_pts_x = np.linspace(-1.0 + margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
        ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * (-1.0 + margin_y)
        ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        output_ctrl_pts_arr = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        output_ctrl_pts = torch.Tensor(output_ctrl_pts_arr)
        return output_ctrl_pts

    def b_inv(self, b_mat):
        eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
        b_inv, _ = torch.solve(eye, b_mat)
        return b_inv

    def _repeat(self, x, n_repeats):
        rep = torch.unsqueeze(torch.ones(n_repeats), 1).transpose(0, 1)
        x = torch.matmul(x.reshape(-1, 1).int(), rep.int())
        return x.reshape(-1)

    def _interpolate(self, im, x, y):
        # constants
        num_batch, height, width, channels = im.shape

        x = x.float()
        y = y.float()
        out_height, out_width = self.output_image_size
        height_f = torch.tensor(height, dtype=torch.float32)
        width_f = torch.tensor(width, dtype=torch.float32)
        zero = torch.tensor(0, dtype=torch.int32)
        max_y = torch.tensor(height - 1, dtype=torch.int32)
        max_x = torch.tensor(width - 1, dtype=torch.int32)

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, min=zero, max=max_x)
        x1 = torch.clamp(x1, min=zero, max=max_x)
        y0 = torch.clamp(y0, min=zero, max=max_y)
        y1 = torch.clamp(y1, min=zero, max=max_y)

        dim2 = width
        dim1 = width * height
        # base = _repeat(torch.range(0, num_batch-1)*dim1, out_height*out_width)
        base = self._repeat(torch.arange(0, num_batch) * dim1, out_height * out_width).cuda()

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore channels dim
        im_flat = im.reshape(-1, channels)
        im_flat = im_flat.float()

        # tmp = idx_a.unsqueeze(1).long()
        idx_a = idx_a.unsqueeze(1).long()
        idx_b = idx_b.unsqueeze(1).long()
        idx_c = idx_c.unsqueeze(1).long()
        idx_d = idx_d.unsqueeze(1).long()
        if channels != 1:
            tmp_idx_a = idx_a.long()
            tmp_idx_b = idx_b.long()
            tmp_idx_c = idx_c.long()
            tmp_idx_d = idx_d.long()
            for i in range(channels - 1):
                idx_a = torch.cat((idx_a, tmp_idx_a), 1)
                idx_b = torch.cat((idx_b, tmp_idx_b), 1)
                idx_c = torch.cat((idx_c, tmp_idx_c), 1)
                idx_d = torch.cat((idx_d, tmp_idx_d), 1)

        Ia = torch.gather(im_flat, 0, idx_a)
        Ib = torch.gather(im_flat, 0, idx_b)
        Ic = torch.gather(im_flat, 0, idx_c)
        Id = torch.gather(im_flat, 0, idx_d)

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)

        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output

    def solve_system(self, target_ctrl_points):
        """Thin Plate Spline Spatial Transformer layer
        TPS control points are arranged in arbitrary positions given by `coord`.
        coord : float Tensor [num_batch, num_point, 2]
            Relative coordinate of the control points.
        vec : float Tensor [num_batch, num_point, 2]
            The vector on the control points.
        """
        coord = target_ctrl_points
        num_batch = coord.shape[0]
        num_point = self.num_control_points

        vec = torch.ones(num_batch)[:, None, None] * self.source_ctrl_points
        vec = vec.reshape(num_batch, num_point, 2).cuda()
        p = torch.cat([torch.ones([num_batch, num_point, 1]).cuda(), coord], 2)  # [bn, pn, 3]

        p_1 = torch.reshape(p, [num_batch, -1, 1, 3])  # [bn, pn, 1, 3]
        p_2 = torch.reshape(p, [num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
        d = p_1 - p_2  # [bn, pn, pn, 3]
        d2 = torch.sum(torch.pow(d, 2), 3)  # [bn, pn, pn]
        r = d2 * torch.log(d2 + 1e-6)  # [bn, pn, pn]

        W_0 = torch.cat([p, r], 2)  # [bn, pn, 3+pn]
        W_1 = torch.cat([torch.zeros([num_batch, 3, 3]).cuda(), torch.transpose(p, 2, 1)], 2)  # [bn, 3, pn+3]
        W = torch.cat([W_0, W_1], 1)  # [bn, pn+3, pn+3]
        W_inv = self.b_inv(W)

        tp = F.pad(vec, (0, 0, 0, 3))

        tp = tp.squeeze(1)  # [bn, pn+3, 2]
        T = torch.matmul(W_inv, tp)  # [bn, pn+3, 2]
        T = torch.transpose(T, 2, 1)  # [bn, 2, pn+3]

        return T

    def _meshgrid(self, height, width, coord):
        x_t = torch.linspace(-1.0, 1.0, steps=width).reshape(1, width).expand(height, width)
        y_t = torch.linspace(-1.0, 1.0, steps=height).reshape(height, 1).expand(height, width)
        x_t_flat = x_t.reshape(1, 1, -1).cuda()
        y_t_flat = y_t.reshape(1, 1, -1).cuda()

        num_batch = coord.shape[0]
        px = torch.unsqueeze(coord[:, :, 0], 2)  # [bn, pn, 1]
        py = torch.unsqueeze(coord[:, :, 1], 2)  # [bn, pn, 1]

        d2 = torch.pow(x_t_flat - px, 2) + torch.pow(y_t_flat - py, 2)

        r = d2 * torch.log(d2 + 1e-6)  # [bn, pn, h*w]
        x_t_flat_g = x_t_flat.expand(num_batch, x_t_flat.shape[1], x_t_flat.shape[2])
        y_t_flat_g = y_t_flat.expand(num_batch, y_t_flat.shape[1], y_t_flat.shape[2])

        grid = torch.cat((torch.ones(x_t_flat_g.shape).cuda(), x_t_flat_g, y_t_flat_g, r), 1)
        return grid

    def forward(self, input_dim, coord):
        T = self.solve_system(coord)
        input_dim = input_dim.permute(0, 2, 3, 1)
        num_batch, height, width, num_channels = input_dim.shape

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        out_height, out_width = self.output_image_size
        grid = self._meshgrid(out_height, out_width, coord)  # [2, h*w]
        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = torch.matmul(T, grid)
        x_s = torch.unsqueeze(T_g[:, 0, :], 1)
        y_s = torch.unsqueeze(T_g[:, 1, :], 1)
        x_s_flat = x_s.reshape(-1)
        y_s_flat = y_s.reshape(-1)

        input_transformed = self._interpolate(input_dim, x_s_flat, y_s_flat)

        output = input_transformed.reshape(num_batch, out_height, out_width, num_channels)
        output = output.permute(0, 3, 1, 2)
        return output, None

    def point_transform(point, T, coord):
        point = torch.Tensor(point.reshape([1, 1, 2]))
        d2 = torch.sum(torch.pow(point - coord, 2), 2)
        r = d2 * torch.log(d2 + 1e-6)
        q = torch.Tensor(np.array([[1, point[0, 0, 0], point[0, 0, 1]]]))
        x = torch.cat([q, r], 1)
        point_T = torch.matmul(T, torch.transpose(x.unsqueeze(1), 2, 1))
        return point_T
