import h5py
import util.color_compl
import os
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
dataroot = 'G:/gen/'
savepath = 'G:/outhaze/'

def makehaze(nohaze,depth):
    # 归一化所有素材，depth在输入之前已经进行过插值resize
    maxhaze = np.max(depth)
    minhaze = np.min(depth)

    depth = (depth-minhaze)/(maxhaze-minhaze)
    # 固定随机种子
    # manualSeed = random.randint(1, 10000)
    # random.seed(0)

    # 随机初始化大气散射系数
    beta = uniform(1.5, 2.7)


    # 初步生成大气传播图
    tx1 = np.exp(-beta * depth)

    # 随机大气光值
    a = 1 - 0.5 * uniform(0, 1)
    m = nohaze.shape[0]
    n = nohaze.shape[1]
    A = np.zeros((1, 1, 3))
    A = A + a
    rep_atmosphere = np.tile(A, [m, n, 1])
    tx1 = np.reshape(tx1, [m, n, 1])
    # 大气传播率图
    max_transmission = np.tile(tx1, [1, 1, 3])
    # 雾图
    haze_image = nohaze * max_transmission + rep_atmosphere * (1 - max_transmission)

    return haze_image,tx1,rep_atmosphere

def make_val_outhaze():
    files = os.listdir(dataroot)
    for idx in files:
        filename = dataroot+idx
        print(filename)
        f = h5py.File(filename,mode='r')
        spotdepth = f['odepth'][:]
        nohaze = f['input'][:]
        c_depth = util.color_compl.fill_depth_colorization(nohaze,spotdepth)
        outhaze,out_tran,out_ato = makehaze(nohaze,c_depth)
        save = h5py.File(savepath+'haze'+idx,mode='w')
        save.create_dataset('haze',data=outhaze)
        save.create_dataset('tran', data=out_tran)
        save.create_dataset('ato', data=out_ato)
        save.create_dataset('depth', data=c_depth)
        save.create_dataset('gt', data=nohaze)


if __name__ == '__main__':
    make_val_outhaze()