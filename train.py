import os
import argparse
import logging
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F
from mmcv import Config
from tqdm import tqdm
from torch.utils.data import DataLoader
from loss import build_generator_loss, build_discriminator_loss, build_generator_loss_with_real
from datagen import custom_dataset, TwoStreamBatchSampler
from model import Generator, Discriminator, Vgg19
from rec_model import Rec_Model
from rec_utils import AttnLabelConverter
from torch.utils.tensorboard import SummaryWriter


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def rgb2grey(img):
    img = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
    img = img.unsqueeze(1)

    return img


def get_logger(cfg, log_filename='log.txt', log_level=logging.INFO):
    logger = logging.getLogger(log_filename)
    logger.setLevel(log_level)
    formatter = logging.Formatter('[%(asctime)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if not os.path.exists(cfg.checkpoint_savedir):
        os.makedirs(cfg.checkpoint_savedir)
    fh = logging.FileHandler(os.path.join(cfg.checkpoint_savedir, log_filename))
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    gpu_num = torch.cuda.device_count()

    logger = get_logger(cfg)
    logger.info('Config path: {}'.format(args.config))
    writer = SummaryWriter(cfg.checkpoint_savedir + 'tensorboard/')

    train_data = custom_dataset(cfg, mode='train', with_real_data=cfg.with_real_data)
    if cfg.with_real_data:
        len_synth, len_real = train_data.custom_len()
        synth_idxs = list(range(len_synth))
        real_idxs = list(range(len_synth, len_synth + len_real))
        batch_sampler = TwoStreamBatchSampler(synth_idxs, real_idxs, cfg.batch_size, cfg.real_bs)  # default: shuffle = True, drop_last = True
        train_loader = DataLoader(
            dataset=train_data,
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True)
    else:
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True)
    eval_data = custom_dataset(cfg, data_dir=cfg.example_data_dir, mode='eval')
    eval_loader = DataLoader(
        dataset=eval_data,
        batch_size=1,
        shuffle=False)

    G = Generator(cfg, in_channels=3).cuda()
    D1 = Discriminator(cfg, in_channels=6).cuda()
    D2 = Discriminator(cfg, in_channels=6).cuda()
    vgg_features = Vgg19(cfg.vgg19_weights).cuda()
    if cfg.with_recognizer:
        converter = AttnLabelConverter('0123456789abcdefghijklmnopqrstuvwxyz')
        Recognizer = Rec_Model(cfg)
        rec_state_dict = torch.load(cfg.rec_ckpt_path, map_location='cpu')
        if len(rec_state_dict) == 1:
            rec_state_dict = rec_state_dict['recognizer']
        rec_state_dict = {k.replace('module.', ''): v for k, v in rec_state_dict.items()}
        Recognizer.cuda()
        Recognizer.load_state_dict(rec_state_dict)
        logger.info('Recognizer module loaded: {}'.format(cfg.rec_ckpt_path))
    G_solver = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    D1_solver = torch.optim.Adam(D1.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    D2_solver = torch.optim.Adam(D2.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    if cfg.with_recognizer and cfg.train_recognizer:
        Rec_solver = torch.optim.Adam(Recognizer.parameters(), lr=cfg.rec_lr_weight * cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))

    if os.path.exists(cfg.ckpt_path):
        checkpoint = torch.load(cfg.ckpt_path, map_location='cpu')
        G.load_state_dict(checkpoint['generator'])
        D1.load_state_dict(checkpoint['discriminator1'])
        D2.load_state_dict(checkpoint['discriminator2'])
        G_solver.load_state_dict(checkpoint['g_optimizer'])
        D1_solver.load_state_dict(checkpoint['d1_optimizer'])
        D2_solver.load_state_dict(checkpoint['d2_optimizer'])
        logger.info('Model loaded: {}'.format(cfg.ckpt_path))
    else:
        logger.info('Model not found')
    if os.path.exists(cfg.inpaint_ckpt_path):
        checkpoint = torch.load(cfg.inpaint_ckpt_path, map_location='cpu')
        G.load_state_dict(checkpoint['generator'], strict=False)
        logger.info('Inpainting module loaded: {}'.format(cfg.inpaint_ckpt_path))
    else:
        logger.info('Inpainting module not found')

    if gpu_num > 1:
        logger.info('Parallel Computing. Using {} GPUs.'.format(gpu_num))
    G = torch.nn.DataParallel(G, device_ids=range(gpu_num))
    D1 = torch.nn.DataParallel(D1, device_ids=range(gpu_num))
    D2 = torch.nn.DataParallel(D2, device_ids=range(gpu_num))
    vgg_features = torch.nn.DataParallel(vgg_features, device_ids=range(gpu_num))
    if cfg.with_recognizer:
        Recognizer = torch.nn.DataParallel(Recognizer, device_ids=range(gpu_num))

    # Train discriminator
    requires_grad(G, False)
    requires_grad(D1, True)
    requires_grad(D2, True)

    trainiter = iter(train_loader)
    for step in tqdm(range(cfg.max_iter)):
        D1_solver.zero_grad()
        D2_solver.zero_grad()
        
        if ((step + 1) % cfg.save_ckpt_interval == 0):
            torch.save(
                {
                    'generator': G.module.state_dict(),
                    'discriminator1': D1.module.state_dict(),
                    'discriminator2': D2.module.state_dict(),
                    'g_optimizer': G_solver.state_dict(),
                    'd1_optimizer': D1_solver.state_dict(),
                    'd2_optimizer': D2_solver.state_dict(),
                },
                cfg.checkpoint_savedir + f'train_step-{step + 1}.model',
            )
            if cfg.with_recognizer:
                torch.save({'recognizer': Recognizer.module.state_dict()}, cfg.checkpoint_savedir + 'best_recognizer.model')

        try:
            i_t, i_s, t_b, t_f, mask_t, mask_s, texts = trainiter.next()
        except StopIteration:
            trainiter = iter(train_loader)
            i_t, i_s, t_b, t_f, mask_t, mask_s, texts = trainiter.next()
        i_t = i_t.cuda()
        i_s = i_s.cuda()
        t_b = t_b.cuda()
        t_f = t_f.cuda()
        mask_t = mask_t.cuda()
        mask_s = mask_s.cuda()

        if cfg.with_recognizer:
            texts, texts_length = converter.encode(texts, batch_max_length=34)
            texts = texts.cuda()
            rec_target = texts[:, 1:]
            labels = [t_b, t_f, mask_t, mask_s, rec_target]
        else:
            labels = [t_b, t_f, mask_t, mask_s]

        o_b_ori, o_b, o_f, x_t_tps, o_mask_s, o_mask_t = G(i_t, i_s)

        if cfg.with_real_data:
            i_db_true = torch.cat((t_b[:(cfg.batch_size - cfg.real_bs) // gpu_num], i_s[:(cfg.batch_size - cfg.real_bs) // gpu_num]), dim=1)
            i_db_pred = torch.cat((o_b[:(cfg.batch_size - cfg.real_bs) // gpu_num], i_s[:(cfg.batch_size - cfg.real_bs) // gpu_num]), dim=1)
        else:
            i_db_true = torch.cat((t_b, i_s), dim=1)
            i_db_pred = torch.cat((o_b, i_s), dim=1)
        o_db_true = D1(i_db_true)
        o_db_pred = D1(i_db_pred)
        i_df_true = torch.cat((t_f, i_t), dim=1)
        i_df_pred = torch.cat((o_f, i_t), dim=1)
        o_df_true = D2(i_df_true)
        o_df_pred = D2(i_df_pred)

        db_loss = build_discriminator_loss(o_db_true, o_db_pred)
        df_loss = build_discriminator_loss(o_df_true, o_df_pred)
        db_loss.backward()
        df_loss.backward()
        D1_solver.step()
        D2_solver.step()

        # Train generator
        requires_grad(G, True)
        requires_grad(D1, False)
        requires_grad(D2, False)

        G_solver.zero_grad()
        if cfg.with_recognizer and cfg.train_recognizer:
            Rec_solver.zero_grad()
        o_b_ori, o_b, o_f, x_t_tps, o_mask_s, o_mask_t = G(i_t, i_s)

        if cfg.with_real_data:
            i_db_pred = torch.cat((o_b[:(cfg.batch_size - cfg.real_bs) // gpu_num], i_s[:(cfg.batch_size - cfg.real_bs) // gpu_num]), dim=1)
        else:
            i_db_pred = torch.cat((o_b, i_s), dim=1)
        i_df_pred = torch.cat((o_f, i_t), dim=1)
        o_db_pred = D1(i_db_pred)
        o_df_pred = D2(i_df_pred)
        i_vgg = torch.cat((t_f, o_f), dim=0)
        out_vgg = vgg_features(i_vgg)
        if cfg.with_recognizer:
            if cfg.use_rgb:
                tmp_o_f = o_f
                tmp_t_f = t_f
            else:
                tmp_o_f = rgb2grey(o_f)
                tmp_t_f = rgb2grey(t_f)
            rec_preds = Recognizer(tmp_o_f, texts[:, :-1], is_train=False)
            out_g = [o_b, o_f, o_mask_s, o_mask_t, rec_preds]
        else:
            out_g = [o_b, o_f, o_mask_s, o_mask_t]
        out_d = [o_db_pred, o_df_pred]

        if cfg.with_real_data:
            g_loss, metrics = build_generator_loss_with_real(cfg, out_g, out_d, out_vgg, labels)
        else:
            g_loss, metrics = build_generator_loss(cfg, out_g, out_d, out_vgg, labels)
        g_loss.backward()
        G_solver.step()
        if cfg.with_recognizer and cfg.train_recognizer:
            Rec_solver.step()

        requires_grad(G, False)
        requires_grad(D1, True)
        requires_grad(D2, True)

        if ((step + 1) % cfg.write_log_interval == 0):
            loss_str = 'Iter: {}/{} | Gen:{:<10.6f} | D_bg:{:<10.6f} | D_fus:{:<10.6f} | G_lr:{} | D_lr:{}'.format(
                step + 1, cfg.max_iter,
                g_loss.item(),
                db_loss.item(),
                df_loss.item(),
                G_solver.param_groups[0]['lr'],
                D1_solver.param_groups[0]['lr'])
            writer.add_scalar('main/G_loss', g_loss.item(), step)
            writer.add_scalar('main/db_loss', db_loss.item(), step)
            writer.add_scalar('main/df_loss', df_loss.item(), step)

            logger.info(loss_str)
            for name, metric in metrics.items():
                loss_str = ' | '.join(['{:<7}: {:<10.6f}'.format(sub_name, sub_metric) for sub_name, sub_metric in metric.items()])
                for sub_name, sub_metric in metric.items():
                    writer.add_scalar(name + '/' + sub_name, sub_metric, step)
                logger.info(loss_str)

        if ((step + 1) % cfg.gen_example_interval == 0):
            savedir = os.path.join(cfg.example_result_dir, 'iter-' + str(step + 1).zfill(len(str(cfg.max_iter))))
            with torch.no_grad():
                for inp in eval_loader:
                    i_t = inp[0].cuda()
                    i_s = inp[1].cuda()
                    name = str(inp[2][0])
                    name, suffix = name.split('.')

                    G.eval()
                    o_b_ori, o_b, o_f, x_t_tps, o_mask_s, o_mask_t = G(i_t, i_s)
                    G.train()
                    
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                    o_mask_s = o_mask_s.detach().squeeze(0).to('cpu').numpy().transpose(1, 2, 0)
                    o_mask_t = o_mask_t.detach().squeeze(0).to('cpu').numpy().transpose(1, 2, 0)
                    x_t_tps = x_t_tps.detach().squeeze(0).to('cpu').numpy().transpose(1, 2, 0)
                    o_b_ori = o_b_ori.detach().squeeze(0).to('cpu').numpy().transpose(1, 2, 0)
                    o_b = o_b.detach().squeeze(0).to('cpu').numpy().transpose(1, 2, 0)
                    o_f = o_f.detach().squeeze(0).to('cpu').numpy().transpose(1, 2, 0)
                    cv2.imwrite(os.path.join(savedir, name + '_o_f.' + suffix), o_f[:, :, ::-1] * 255)
                    cv2.imwrite(os.path.join(savedir, name + '_o_b_ori.' + suffix), o_b_ori[:, :, ::-1] * 255)
                    cv2.imwrite(os.path.join(savedir, name + '_o_b.' + suffix), o_b[:, :, ::-1] * 255)
                    cv2.imwrite(os.path.join(savedir, name + '_o_mask_s.' + suffix), o_mask_s * 255)
                    cv2.imwrite(os.path.join(savedir, name + '_o_mask_t.' + suffix), o_mask_t * 255)
                    cv2.imwrite(os.path.join(savedir, name + '_x_t_tps.' + suffix), x_t_tps[:, :, ::-1] * 255)    
                    

if __name__ == '__main__':
    main()
