import numpy as np
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
from keras.utils import Sequence
import h5py
import matplotlib.pyplot as plt
from random import uniform
import random
img_size = 512
def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

def nyu_resize(img, resolution=512, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, resolution), preserve_range=True, mode='reflect', anti_aliasing=True )

def get_eign_nyu(nyu_data_zipfile='G:/nyu_test.zip'):
    data = extract_zip(nyu_data_zipfile)
    rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
    depth = np.load(BytesIO(data['eigen_test_depth.npy']))


def get_nyu_data(nyu_data_zipfile='H:/dataset/nyu_data1.zip'):
    data = extract_zip(nyu_data_zipfile)

    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))


    # Helpful for testing...
    return data, nyu2_train, nyu2_test


def makehaze(nohaze,depth):
    # 归一化所有素材，depth在输入之前已经进行过插值resize
    nohaze = nohaze/255
    maxhaze = np.max(depth)
    minhaze = np.min(depth)

    depth = (depth-minhaze)/(maxhaze-minhaze)
    # 固定随机种子
    # manualSeed = random.randint(1, 10000)
    # random.seed(0)

    plt.imshow(depth)
    plt.show()
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



def make_nyu_dataset():
    data,train,test = get_nyu_data()
    dataset = train
    testset = test
    # 保存文件的根目录
    save_path = ''
    for tidx in range(len(dataset)):
        sample = dataset[tidx]
        img = np.asarray(Image.open(BytesIO(data[sample[0]])))
        img = nyu_resize(img,img_size)
        depth = np.asarray(Image.open(BytesIO(data[sample[1]])))
        depth = nyu_resize(depth,img_size)
        hazeimg,tran,ato = makehaze(img,depth)
        img = img/255
        f = h5py.File(save_path+'/train/'+str(tidx)+'.h5','w')
        f.create_dataset('gt',img)
        f.create_dataset('haze', hazeimg)
        f.create_dataset('depth',depth)
        f.create_dataset('tran',tran)
        f.create_dataset('ato',ato)

    for vidx in range(len(testset)):
        sample = testset[vidx]
        img = np.asarray(Image.open(BytesIO(data[sample[0]])))
        img = nyu_resize(img,img_size)
        depth = np.asarray(Image.open(BytesIO(data[sample[1]])))
        depth = nyu_resize(depth,img_size)
        hazeimg,tran,ato = makehaze(img,depth)
        img = img/255
        f = h5py.File(save_path+'/val/'+str(vidx)+'.h5','w')
        f.create_dataset('gt',img)
        f.create_dataset('haze', hazeimg)
        f.create_dataset('depth',depth)
        f.create_dataset('tran',tran)
        f.create_dataset('ato',ato)




if __name__ == '__main__':
     make_nyu_dataset()
