import os
import re
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from PIL import Image
import standard_text


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size), grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    # Collect data into fixed-length chunks or blocks
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class custom_dataset(Dataset):
    def __init__(self, cfg, data_dir=None, i_t_name='i_t.txt', mode='train', with_real_data=False):
        self.cfg = cfg
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize(cfg.data_shape),
            transforms.ToTensor(),
        ])
        self.std_text = standard_text.Std_Text(cfg.font_path)

        if(self.mode == 'train'):
            self.data_dir = cfg.data_dir
            if isinstance(self.data_dir, str):
                self.data_dir = [self.data_dir]
            assert isinstance(self.data_dir, list)

            self.name_list = []
            self.i_t_list = {}
            for tmp_data_dir in self.data_dir:
                tmp_dataset_name = tmp_data_dir.rsplit('/', 1)[-1]
                with open(os.path.join(tmp_data_dir, i_t_name), 'r') as f:
                    lines = f.readlines()
                self.name_list += [os.path.join(tmp_data_dir, '{}', line.strip().split()[0]) for line in lines]
                for line in lines:
                    tmp_key, tmp_val = line.strip().split()
                    self.i_t_list[tmp_dataset_name + '_' + tmp_key] = tmp_val

            self.len_synth = len(self.name_list)
            assert self.len_synth == len(self.i_t_list)

            if with_real_data:
                self.real_data_dir = cfg.real_data_dir
                if isinstance(self.real_data_dir, str):
                    self.real_data_dir = [self.real_data_dir]
                assert isinstance(self.real_data_dir, list)

                self.real_name_list = []
                self.real_i_t_list = {}
                for tmp_data_dir in self.real_data_dir:
                    tmp_dataset_name = tmp_data_dir.rsplit('/', 1)[-1]
                    with open(os.path.join(tmp_data_dir, i_t_name), 'r') as f:
                        lines = f.readlines()
                    self.real_name_list += [os.path.join(tmp_data_dir, '{}', line.strip().split()[0]) for line in lines]
                    for line in lines:
                        tmp_key, tmp_val = line.strip().split()
                        self.real_i_t_list[tmp_dataset_name + '_' + tmp_key] = tmp_val

                self.len_real = len(self.real_name_list)
                assert self.len_real == len(self.real_i_t_list)
                self.name_list += self.real_name_list
        else:
            assert data_dir is not None
            self.data_dir = data_dir
            with open(os.path.join(data_dir, '../' + i_t_name), 'r') as f:
                lines = f.readlines()
            self.name_list = [line.strip().split()[0] for line in lines]
            self.i_t_list = {line.strip().split()[0]: line.strip().split()[1] for line in lines}

    def custom_len(self):
        return self.len_synth, self.len_real

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx]
        if self.mode == 'train':
            if idx < self.len_synth:
                _, tmp_dataset_name, _, tmp_key = img_name.rsplit('/', 3)
                tmp_text = self.i_t_list[tmp_dataset_name + '_' + tmp_key]
                i_t = self.std_text.draw_text(tmp_text)
                i_t = Image.fromarray(np.uint8(i_t))
                i_s = Image.open(img_name.format(self.cfg.i_s_dir))
                if i_s.mode != 'RGB':
                    i_s = i_s.convert('RGB')
                t_b = Image.open(img_name.format(self.cfg.t_b_dir))
                t_f = Image.open(img_name.format(self.cfg.t_f_dir))
                mask_t = Image.open(img_name.format(self.cfg.mask_t_dir))
                mask_s = Image.open(img_name.format(self.cfg.mask_s_dir))
                with open(img_name.format(self.cfg.txt_dir)[:-4] + '.txt', 'r') as f:
                    lines = f.readlines()
                text = lines[0].strip().split()[-1]
                text = re.sub("[^0-9a-zA-Z]+", "", text).lower()
                i_t = self.transform(i_t)
                i_s = self.transform(i_s)
                t_b = self.transform(t_b)
                t_f = self.transform(t_f)
                mask_t = self.transform(mask_t)
                mask_s = self.transform(mask_s)
            else:
                _, tmp_dataset_name, _, tmp_key = img_name.rsplit('/', 3)
                tmp_text = self.real_i_t_list[tmp_dataset_name + '_' + tmp_key]
                i_t = self.std_text.draw_text(tmp_text)
                i_t = Image.fromarray(np.uint8(i_t))
                i_s = Image.open(img_name.format(self.cfg.i_s_dir))
                if i_s.mode != 'RGB':
                    i_s = i_s.convert('RGB')
                with open(img_name.format(self.cfg.txt_dir)[:-4] + '.txt', 'r') as f:
                    lines = f.readlines()
                text = lines[0].strip().split()[-1]
                text = re.sub("[^0-9a-zA-Z]+", "", text).lower()
                i_t = self.transform(i_t)
                i_s = self.transform(i_s)
                t_f = i_s
                t_b = -1 * torch.ones([3] + self.cfg.data_shape)
                mask_t = -1 * torch.ones([1] + self.cfg.data_shape)
                mask_s = -1 * torch.ones([1] + self.cfg.data_shape)

            return [i_t, i_s, t_b, t_f, mask_t, mask_s, text]
        else:
            main_name = img_name
            i_s = Image.open(os.path.join(self.data_dir, img_name))
            if i_s.mode != 'RGB':
                i_s = i_s.convert('RGB')
            tmp_text = self.i_t_list[img_name]
            i_t = self.std_text.draw_text(tmp_text)
            i_t = Image.fromarray(np.uint8(i_t))
            i_s = self.transform(i_s)
            i_t = self.transform(i_t)

            return [i_t, i_s, main_name]


class erase_dataset(Dataset):
    def __init__(self, cfg, data_dir=None, mode='train'):
        self.cfg = cfg
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize(cfg.data_shape),
            transforms.ToTensor()
        ])
        if(self.mode == 'train'):
            self.data_dir = cfg.data_dir
            if isinstance(self.data_dir, str):
                self.data_dir = [self.data_dir]
            assert isinstance(self.data_dir, list)
            self.name_list = []
            for tmp_data_dir in self.data_dir:
                self.name_list += [os.path.join(tmp_data_dir, '{}', filename) for filename in os.listdir(os.path.join(tmp_data_dir, cfg.i_s_dir))]
        else:
            assert data_dir is not None
            self.data_dir = data_dir
            self.name_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx]
        if self.mode == 'train':
            i_s = Image.open(img_name.format(self.cfg.i_s_dir))
            t_b = Image.open(img_name.format(self.cfg.t_b_dir))
            mask_s = Image.open(img_name.format(self.cfg.mask_s_dir))
            i_s = self.transform(i_s)
            t_b = self.transform(t_b)
            mask_s = self.transform(mask_s)

            return [i_s, t_b, mask_s]
        else:
            main_name = img_name
            i_s = Image.open(os.path.join(self.data_dir, img_name))
            if i_s.mode != 'RGB':
                i_s = i_s.convert('RGB')
            i_s = self.transform(i_s)

            return [i_s, main_name]
        