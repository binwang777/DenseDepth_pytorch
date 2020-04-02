import torch
import os
import sys
from torch.utils.data import Dataset
from util import data_transform
from PIL import Image
import pandas as pd
import numpy as np
import h5py
import random

random.seed(0)

def DepthNorm(x, maxDepth):
    return maxDepth / x

def nyu_resize(img, resolution=480, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, resolution), preserve_range=True, mode='reflect', anti_aliasing=True )


class NYUDataset(Dataset):
    # nyu depth dataset
    def __init__(self, root_dir, split,seed=None):


        self.split = split
        self.data_list = os.listdir(root_dir)
        self.rootdir = root_dir
        self.maxDepth = 1000
        self.minDepth = 10


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx,seed):
        # read input image
        file_name = self.data_list[idx]
        file_name = self.rootdir+file_name
        # 获取图片为numpy
        f = h5py.File(file_name, 'r')

        haze =f['gt'][:]
        depth =f['depth'][:]

        # z是深度图,y是传播率图,x是雾图.
        if self.split == 'train':
            z = np.clip(depth.reshape(512, 512, 1) / 255 * self.maxDepth, 0,self.maxDepth)
            z = DepthNorm(z, maxDepth=self.maxDepth)
            x = nyu_resize(haze, 256)
            z = nyu_resize(z, 128)

            # 数据增强的策略需要自定义设置
            # 数据增强
            if random.uniform(0, 1) <= 0.5:
                x = x[..., ::-1, :]
                z = z[..., ::-1, :]


            #  Flip image vertically
            if random.uniform(0, 1) < 0.5:
                x = x[..., ::-1, :, :]
                z = z[..., ::-1, :, :]



        if self.split == 'val':
            z = np.asarray(depth, dtype=np.float32).reshape(512, 512, 1).copy().astype(float) / 10.0
            z = np.clip(z / 255 * self.maxDepth, 0, self.maxDepth)
            z = DepthNorm(z, maxDepth=self.maxDepth)

            x = nyu_resize(haze, 256)
            z = nyu_resize(z, 128)


        x = np.transpose(x,[2,0,1])

        z = np.transpose(z, [2, 0, 1])

        x = torch.from_numpy(x).to(torch.float32)

        z = torch.from_numpy(z).to(torch.float32)

        sample = {'haze': x, 'depth': z}

        return sample



