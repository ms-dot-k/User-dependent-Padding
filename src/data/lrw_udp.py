import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from src.data.transforms import Crop, StatefulRandomHorizontalFlip

class MultiDataset(Dataset):
    def __init__(self, lrw, mode, max_v_timesteps=155, augmentations=False, num_mel_bins=80, subject=0, adapt_min=1, fold=1):
        assert mode in ['train', 'test', 'val']
        assert adapt_min in [1, 3, 5]
        self.mode = mode
        self.fold = fold
        self.max_v_timesteps = max_v_timesteps
        self.augmentations = augmentations if mode == 'train' else False
        self.speed = [0.95, 1.0, 1.05]
        self.num_mel_bins = num_mel_bins
        self.skip_long_samples = True
        self.file_paths, self.word_list = self.build_file_list(lrw, mode, subject, adapt_min)
        self.word2int = {word: index for index, word in self.word_list.items()}

    def build_file_list(self, lrw, mode, target_subject, adapt_min):
        word = {}
        sub_dict = {}
        check = []

        sub_num = -1
        with open(f'./data/LRW_ID_{mode}.txt', 'r') as f:
            lines = f.readlines()
        for l in lines:
            subject, f_name = l.strip().split()
            if not subject in check:
                check.append(subject)
                sub_num += 1
            if not sub_num in sub_dict:
                sub_dict[sub_num] = [os.path.join(lrw, f_name + '.mp4')]
            else:
                sub_dict[sub_num] += [os.path.join(lrw, f_name + '.mp4')]

        classes = sorted(os.listdir(lrw))
        for i, cla in enumerate(classes):
            word[i] = cla

        file_list = sub_dict[target_subject]
        if self.fold != 1:
            np.random.seed(self.fold)
            file_list = list(np.random.permutation(np.array(file_list)))

        if mode != 'test':
            file_list = file_list[:round(adapt_min * 60 / 1.16)]
            print(f'subject ind:{target_subject} #{check[target_subject]}, {mode} with fold {self.fold} (val) refers to enrollment, Use data of {adapt_min} min, so {len(file_list)} data')
        else:
            print(f'subject ind:{target_subject} #{check[target_subject]}, {mode}, Use full data, so {len(file_list)} data')
        return file_list, word

    def __len__(self):
        return len(self.file_paths)

    def build_tensor(self, frames):
        if self.augmentations:
            x, y = [random.randint(-5, 5) for _ in range(2)]
        else:
            x, y = 0, 0
        crop = [59 + x, 95 + y, 195 + x, 231 + y]  # 136, 136
        if self.augmentations:
            augmentations1 = transforms.Compose([StatefulRandomHorizontalFlip(0.5)])
        else:
            augmentations1 = transforms.Compose([])

        transform = transforms.Compose([
            transforms.ToPILImage(),
            Crop(crop),
            transforms.Resize([112, 112]),
            augmentations1,
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(0.4136, 0.1700)
        ])

        temporalVolume = torch.zeros(self.max_v_timesteps, 1, 112, 112)
        for i, frame in enumerate(frames):
            temporalVolume[i] = transform(frame)

        ### Random Spatial Erasing ###
        if self.augmentations:
            x_s, y_s = [random.randint(-10, 66) for _ in range(2)]  # starting point
            temporalVolume[:, :, np.maximum(0, y_s):np.minimum(112, y_s + 56), np.maximum(0, x_s):np.minimum(112, x_s + 56)] = 0.

        ### Random Temporal Erasing ###
        if self.augmentations:
            t_s = random.randint(0, 29 - 3)  # starting point
            temporalVolume[t_s:t_s + 3, :, :, :] = 0.

        temporalVolume = temporalVolume.transpose(1, 0)  # (C, T, H, W)
        return temporalVolume

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        f_name = os.path.split(file_path)[-1]
        content = os.path.split(file_path)[-1].split('_')[0]
        target = self.word2int[content]

        video, _, info = torchvision.io.read_video(file_path, pts_unit='sec')

        ## Video ##
        video = video.permute(0, 3, 1, 2)  # T C H W

        frames = self.build_tensor(video)

        return frames, target, f_name

