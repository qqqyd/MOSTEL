import os
import argparse
import logging
import numpy as np
import cv2
import torch
from mmcv import Config
from tqdm import tqdm
from torch.utils.data import DataLoader
from loss import build_generator_erase_loss, build_discriminator_loss
from datagen import erase_dataset
from model_erase import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


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

    logger = get_logger(cfg)
    logger.info('Config path: {}'.format(args.config))
    writer = SummaryWriter(cfg.checkpoint_savedir + 'tensorboard/')

    train_data = erase_dataset(cfg, mode='train')
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True)
    eval_data = erase_dataset(cfg, data_dir=cfg.example_data_dir, mode='eval')
    eval_loader = DataLoader(
        dataset=eval_data,
        batch_size=1,
        shuffle=False)

    G = Generator(cfg, in_channels=3).cuda()
    D1 = Discriminator(cfg, in_channels=6).cuda()
    G_solver = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    D1_solver = torch.optim.Adam(D1.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))

    try:
        checkpoint = torch.load(cfg.ckpt_path)
        G.load_state_dict(checkpoint['generator'])
        D1.load_state_dict(checkpoint['discriminator1'])
        G_solver.load_state_dict(checkpoint['g_optimizer'])
        D1_solver.load_state_dict(checkpoint['d1_optimizer'])
        logger.info('Model loaded: {}'.format(cfg.ckpt_path))
    except FileNotFoundError:
        logger.info('Model not found')

    requires_grad(G, False)
    requires_grad(D1, True)

    trainiter = iter(train_loader)
    for step in tqdm(range(cfg.max_iter)):
        D1_solver.zero_grad()
        if ((step + 1) % cfg.save_ckpt_interval == 0):
            torch.save(
                {
                    'generator': G.state_dict(),
                    'discriminator1': D1.state_dict(),
                    'g_optimizer': G_solver.state_dict(),
                    'd1_optimizer': D1_solver.state_dict(),
                },
                cfg.checkpoint_savedir + f'train_step-{step+1}.model',
            )

        try:
            i_s, t_b, mask_s = trainiter.next()
        except StopIteration:
            trainiter = iter(train_loader)
            i_s, t_b, mask_s = trainiter.next()

        i_s = i_s.cuda()
        t_b = t_b.cuda()
        mask_s = mask_s.cuda()
        labels = [t_b, mask_s]

        o_b_ori, o_b, o_mask_s = G(i_s)
        i_db_true = torch.cat((t_b, i_s), dim=1)
        i_db_pred = torch.cat((o_b, i_s), dim=1)
        o_db_true = D1(i_db_true)
        o_db_pred = D1(i_db_pred)

        db_loss = build_discriminator_loss(o_db_true, o_db_pred)
        db_loss.backward()
        D1_solver.step()

        # Train generator
        requires_grad(G, True)
        requires_grad(D1, False)
        G_solver.zero_grad()

        o_b_ori, o_b, o_mask_s = G(i_s)
        i_db_pred = torch.cat((o_b, i_s), dim=1)
        o_db_pred = D1(i_db_pred)

        out_g = [o_b, o_mask_s]
        out_d = o_db_pred

        g_loss, metrics = build_generator_erase_loss(cfg, out_g, out_d, labels)
        g_loss.backward()
        G_solver.step()

        requires_grad(G, False)
        requires_grad(D1, True)

        if ((step + 1) % cfg.write_log_interval == 0):
            loss_str = 'Iter: {}/{} | Gen:{:<10.6f} | D_bg:{:<10.6f} | G_lr:{} | D_lr:{}'.format(
                step + 1, cfg.max_iter,
                g_loss.item(),
                db_loss.item(),
                G_solver.param_groups[0]['lr'],
                D1_solver.param_groups[0]['lr'])
            writer.add_scalar('main/G_loss', g_loss.item(), step)
            writer.add_scalar('main/db_loss', db_loss.item(), step)

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
                    i_s = inp[0].cuda()
                    name = str(inp[1][0])
                    name, suffix = name.split('.')

                    G.eval()
                    o_b_ori, o_b, o_mask_s = G(i_s)
                    G.train()

                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                    o_mask_s = o_mask_s.detach().squeeze(0).to('cpu').numpy().transpose(1, 2, 0)
                    o_b = o_b.detach().squeeze(0).to('cpu').numpy().transpose(1, 2, 0)
                    cv2.imwrite(os.path.join(savedir, name + '_o_mask_s.' + suffix), o_mask_s * 255)
                    cv2.imwrite(os.path.join(savedir, name + '_o_b.' + suffix), o_b[:, :, ::-1] * 255)


if __name__ == '__main__':
    main()
