import torch
import torch.nn.functional as F

gpu_num = torch.cuda.device_count()
epsilon = 1e-8

def build_discriminator_loss(x_true, x_fake):
    d_loss = -torch.mean(torch.log(torch.clamp(x_true, epsilon, 1.0)) +
                         torch.log(torch.clamp(1.0 - x_fake, epsilon, 1.0)))
    return d_loss


def build_dice_loss(x_t, x_o):
    iflat = x_o.view(-1)
    tflat = x_t.view(-1)
    intersection = (iflat * tflat).sum()
    return 1. - torch.mean((2. * intersection + epsilon) / (iflat.sum() + tflat.sum() + epsilon))


def build_l1_loss(x_t, x_o):
    return torch.mean(torch.abs(x_t - x_o))


def build_l2_loss(x_t, x_o):
    return torch.mean((x_t - x_o) ** 2)


def build_perceptual_loss(x):
    l = []
    for i, f in enumerate(x):
        l.append(build_l1_loss(f[0], f[1]))
    l = torch.stack(l, dim=0)
    l = l.sum()
    return l


def build_gram_matrix(x):
    x_shape = x.shape
    c, h, w = x_shape[1], x_shape[2], x_shape[3]
    matrix = x.view((-1, c, h * w))
    matrix1 = torch.transpose(matrix, 1, 2)
    gram = torch.matmul(matrix, matrix1) / (h * w * c)
    return gram


def build_style_loss(x):
    l = []
    for i, f in enumerate(x):
        f_shape = f[0].shape[0] * f[0].shape[1] * f[0].shape[2]
        f_norm = 1. / f_shape
        gram_true = build_gram_matrix(f[0])
        gram_pred = build_gram_matrix(f[1])
        l.append(f_norm * (build_l1_loss(gram_true, gram_pred)))
    l = torch.stack(l, dim=0)
    l = l.sum()
    return l


def build_vgg_loss(x):
    splited = []
    for i, f in enumerate(x):
        splited.append(torch.chunk(f, 2))
    l_per = build_perceptual_loss(splited)
    l_style = build_style_loss(splited)
    return l_per, l_style


def build_gan_loss(x_pred):
    gen_loss = -torch.mean(torch.log(torch.clamp(x_pred, epsilon, 1.0)))
    return gen_loss


def build_recognizer_loss(preds, target):
    loss = F.cross_entropy(preds, target, ignore_index=0)
    return loss


def build_generator_loss(cfg, out_g, out_d, out_vgg, labels):
    if cfg.with_recognizer:
        o_b, o_f, o_mask_s, o_mask_t, rec_preds = out_g
        t_b, t_f, mask_t, mask_s, rec_target = labels
    else:
        o_b, o_f, o_mask_s, o_mask_t = out_g
        t_b, t_f, mask_t, mask_s = labels
    o_db_pred, o_df_pred = out_d
    o_vgg = out_vgg

    # Background Inpainting module loss
    l_b_gan = build_gan_loss(o_db_pred)
    l_b_l2 = cfg.lb_beta * build_l2_loss(t_b, o_b)
    l_b_mask = cfg.lb_mask * build_dice_loss(mask_s, o_mask_s)
    l_b = l_b_gan + l_b_l2 + l_b_mask

    l_f_gan = build_gan_loss(o_df_pred)
    l_f_l2 = cfg.lf_theta_1 * build_l2_loss(t_f, o_f)
    l_f_vgg_per, l_f_vgg_style = build_vgg_loss(o_vgg)
    l_f_vgg_per = cfg.lf_theta_2 * l_f_vgg_per
    l_f_vgg_style = cfg.lf_theta_3 * l_f_vgg_style
    l_f_mask = cfg.lf_mask * build_dice_loss(mask_t, o_mask_t)
    if cfg.with_recognizer:
        l_f_rec = cfg.lf_rec * build_recognizer_loss(rec_preds.view(-1, rec_preds.shape[-1]), rec_target.contiguous().view(-1))
        l_f = l_f_gan + l_f_vgg_per + l_f_vgg_style + l_f_l2 + l_f_mask + l_f_rec
    else:
        l_f = l_f_gan + l_f_vgg_per + l_f_vgg_style + l_f_l2 + l_f_mask
    l = cfg.lb * l_b + cfg.lf * l_f

    metrics = {}
    metrics['l_b'] = {}
    metrics['l_b']['l_b'] = l_b
    metrics['l_b']['l_b_gan'] = l_b_gan
    metrics['l_b']['l_b_l2'] = l_b_l2
    metrics['l_b']['l_b_mask'] = l_b_mask
    metrics['l_f'] = {}
    metrics['l_f']['l_f'] = l_f
    metrics['l_f']['l_f_gan'] = l_f_gan
    metrics['l_f']['l_f_l2'] = l_f_l2
    metrics['l_f']['l_f_vgg_per'] = l_f_vgg_per
    metrics['l_f']['l_f_vgg_style'] = l_f_vgg_style
    metrics['l_f']['l_f_mask'] = l_f_mask
    if cfg.with_recognizer:
        metrics['l_f']['l_f_rec'] = l_f_rec

    return l, metrics


def build_generator_loss_with_real(cfg, out_g, out_d, out_vgg, labels):
    if cfg.with_recognizer:
        o_b, o_f, o_mask_s, o_mask_t, rec_preds = out_g
        t_b, t_f, mask_t, mask_s, rec_target = labels
    else:
        o_b, o_f, o_mask_s, o_mask_t = out_g
        t_b, t_f, mask_t, mask_s = labels
    o_db_pred, o_df_pred = out_d
    o_vgg = out_vgg

    synth_bs = (cfg.batch_size - cfg.real_bs) // gpu_num
    # Background Inpainting module loss
    l_b_gan = build_gan_loss(o_db_pred)
    l_b_l2 = cfg.lb_beta * build_l2_loss(t_b[:synth_bs], o_b[:synth_bs])
    l_b_mask = cfg.lb_mask * build_dice_loss(mask_s[:synth_bs], o_mask_s[:synth_bs])
    l_b = l_b_gan + l_b_l2 + l_b_mask

    l_f_gan = build_gan_loss(o_df_pred)
    l_f_l2 = cfg.lf_theta_1 * build_l2_loss(t_f, o_f)
    l_f_vgg_per, l_f_vgg_style = build_vgg_loss(o_vgg)
    l_f_vgg_per = cfg.lf_theta_2 * l_f_vgg_per
    l_f_vgg_style = cfg.lf_theta_3 * l_f_vgg_style
    l_f_mask = cfg.lf_mask * build_dice_loss(mask_t[:synth_bs], o_mask_t[:synth_bs])
    if cfg.with_recognizer:
        l_f_rec = cfg.lf_rec * build_recognizer_loss(rec_preds.view(-1, rec_preds.shape[-1]), rec_target.contiguous().view(-1))
        l_f = l_f_gan + l_f_vgg_per + l_f_vgg_style + l_f_l2 + l_f_mask + l_f_rec
    else:
        l_f = l_f_gan + l_f_vgg_per + l_f_vgg_style + l_f_l2 + l_f_mask
    l = cfg.lb * l_b + cfg.lf * l_f

    metrics = {}
    metrics['l_b'] = {}
    metrics['l_b']['l_b'] = l_b
    metrics['l_b']['l_b_gan'] = l_b_gan
    metrics['l_b']['l_b_l2'] = l_b_l2
    metrics['l_b']['l_b_mask'] = l_b_mask
    metrics['l_f'] = {}
    metrics['l_f']['l_f'] = l_f
    metrics['l_f']['l_f_gan'] = l_f_gan
    metrics['l_f']['l_f_l2'] = l_f_l2
    metrics['l_f']['l_f_vgg_per'] = l_f_vgg_per
    metrics['l_f']['l_f_vgg_style'] = l_f_vgg_style
    metrics['l_f']['l_f_mask'] = l_f_mask
    if cfg.with_recognizer:
        metrics['l_f']['l_f_rec'] = l_f_rec
        
    return l, metrics

def build_generator_erase_loss(cfg, out_g, out_d, labels):
    o_b, o_mask_s = out_g
    t_b, mask_s = labels
    o_db_pred = out_d

    l_b_gan = build_gan_loss(o_db_pred)
    l_b_l2 = cfg.lb_beta * build_l2_loss(t_b, o_b)
    l_b_mask = cfg.lb_mask * build_dice_loss(mask_s, o_mask_s)
    l_b = l_b_gan + l_b_l2 + l_b_mask
    l = cfg.lb * l_b

    metrics = {}
    metrics['l_b'] = {}
    metrics['l_b']['l_b'] = l_b
    metrics['l_b']['l_b_gan'] = l_b_gan
    metrics['l_b']['l_b_l2'] = l_b_l2
    metrics['l_b']['l_b_mask'] = l_b_mask

    return l, metrics