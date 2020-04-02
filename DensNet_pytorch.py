import torch
import torch.nn as nn
import torchvision
import scipy.misc as sc
import numpy as np
from torchvision import models
from torchvision import datasets, transforms
import torch.nn.functional as F
from model.ECTDnewnet import BaseResNet_DE
import data_transform
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import argparse
import matplotlib
import numpy as np
from keras.models import load_model
from model.loss import depth_loss_function
from model.layers import BilinearUpSampling2D
from keras import applications
# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

Windows_filepath = 'H:\\dataset\\code\\Pytorch_ECTNet\\'
Window_datapath = 'H:\\dataset\\contras_ehnce\\'
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='H:/dataset/code/DenseDepth-master/nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

class Upproject(nn.Module):
   def __init__(self,in_channels,nf):
       super(Upproject,self).__init__()
       # self.upsample = F.upsample_bilinear
       self.upsample = F.interpolate
       self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=nf,stride=1,kernel_size=3,padding=1,bias=True)
       # self.relu = nn.LeakyReLU(0.2,inplace=True)
       self.conv2 = nn.Conv2d(in_channels=nf,out_channels=nf,kernel_size=3,stride=1,padding=1,bias=True)
       self.relu2 = nn.LeakyReLU(0.2)


   def forward(self, input, to_cat):
        shape_out = input.data.size()
        shape_out = shape_out[2:4]
        # print(shape_out)
        x1 = self.upsample(input,size=(shape_out[0]*2,shape_out[1]*2),mode='bilinear',align_corners=True)
        x1 = torch.cat([x1, to_cat], dim=1)
        # x1 = self.upsample(x1,size=(shape_out[0]*2,shape_out[1]*2))
        x2 = self.conv1(x1)
        # x2 = self.relu(x2)
        x3 = self.conv2(x2)
        x3 = self.relu2(x3)
        return x3

class DenseNet_pytorch(nn.Module):
    def __init__(self,in_channels,out_channels,):
        super(DenseNet_pytorch, self).__init__()
        # self.model = models.resnet34(pretrained=False)
        self.model = models.densenet169(pretrained=False)
        # self.model.load_state_dict(torch.load(Windows_filepath+'densenet169-b2777c0a.pth'))
        self.conv0 = self.model.features.conv0
        self.norm0 = self.model.features.norm0
        self.relu0 = self.model.features.relu0
        self.pool0 = self.model.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1 = self.model.features.denseblock1
        self.trans_block1 = self.model.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2 = self.model.features.denseblock2
        self.trans_block2 = self.model.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3 = self.model.features.denseblock3
        self.trans_block3 = self.model.features.transition3

        ############# Block4-down  16-16 ##############
        self.dense_block4 = self.model.features.denseblock4


        self.model_out = self.model.features.norm5
        self.model_relu = F.relu

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_out_channels = 1664
        self.midconv = nn.Conv2d(in_channels=self.model_out_channels,out_channels=self.model_out_channels,kernel_size=1,stride=1,padding=0,bias=True)
        # self.midrelu = nn.LeakyReLU(0.2,inplace=True)
        # 输出：1664
        self.up1 = Upproject(1920,832)
        self.up2 = Upproject(960,416)
        self.up3 = Upproject(480,208)
        self.up4 = Upproject(272, 104)
        self.finalconv = nn.Conv2d(in_channels=104,out_channels=1,kernel_size=3,stride=1,padding=1,bias=True)


    def forward(self, x):
        tempx = x
        shape_out = x.data.size()
        shape_out = shape_out[2:4]

        x0 = self.relu0(self.norm0(self.conv0(x)))
        tx1 =x0
        x0=self.pool0(x0)
        tx2 = x0
        x1 = self.trans_block1(self.dense_block1(x0))
        tx3 = x1
        x2 = self.trans_block2(self.dense_block2(x1))
        tx4 =x2

        x3 = self.trans_block3(self.dense_block3(x2))

        x4 = self.dense_block4(x3)
        finnalout = self.model_out(x4)
        finnalout = self.model_relu(finnalout)

        mid = self.midconv(finnalout)
        # output:640*8*8
        up1 = self.up1(mid, tx4)
        # output:256*16*16
        up2 = self.up2(up1, tx3)
        # output:128*40*40
        up3 = self.up3(up2, tx2)
        # output:64*80*80
        up4 = self.up4(up3, tx1)

        result = self.finalconv(up4)

        return result

def test():

    root = 'H:\\dataset\\pair'
    filename1 = root + '\\test_gt.png'
    img = sc.imread(filename1)
    print(img.shape)
    # input = torch.from_numpy(img)
    input = transforms.ToTensor()(img)
    input = input.float()
    input = torch.unsqueeze(input, 0)
    net = DenseNet_pytorch(3,3)

    net.load_state_dict(torch.load('G:\\my.pth',map_location='cpu'))

    net.eval()
    out= net(input)

    out = out.detach().numpy()
    out = out[0, 0, :, :]
    result = out
    # result = out.detach().numpy()
    # result = np.swapaxes(result, 0, 2)
    # result = np.swapaxes(result, 0, 1)
    # result = result[:, :, 0]
    print(np.max(result))
    result = 10 / result
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    sc.imsave('G:\\show.png',result)
    plt.imshow(result)
    plt.show()



def keras_2pytorch():
    # Argument Parser
    # Custom object needed for inference and training
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}

    print('Loading model...')

    # Load model into GPU / CPU
    model = load_model(args.model, custom_objects=custom_objects, compile=False)
    # 抓取keras的模型参数

    net = DenseNet_pytorch(3,3)
    # dest = torch.load('H:/dataset/code/Pytorch_ECTDnet/testnet_0.pth',map_location='cpu')
    dest = net.state_dict()
    keys = net.state_dict().keys()
    keys = list(keys)
    # str = 'model.features'
    # 初始层赋值
    weights1 = np.asarray(model.get_layer('conv1/conv').get_weights())
    weights1 = handle_w(weights1)
    # dest[keys[1015]] = weights1
    dest[keys[1015]].copy_(weights1)
    load_right(dest,keys[1015],weights1)


    weight_norm = np.asarray(model.get_layer('conv1/bn').get_weights())
    bn_weight = handle_bn(weight_norm)
    dest[keys[1016]].copy_(bn_weight[0])
    load_right(dest,keys[1016],bn_weight[0])

    dest[keys[1017]].copy_(bn_weight[1])
    load_right(dest,keys[1017],bn_weight[1])

    dest[keys[1018]].copy_(bn_weight[2])
    load_right(dest,keys[1018],bn_weight[2])
    # dest[keys[1018]] = bn_weight[2]

    dest[keys[1019]].copy_(bn_weight[3])
    load_right(dest,keys[1019],bn_weight[3])
    # dest[keys[1019]] = bn_weight[3]

    num = 1019
    num = num+2

    # dense_block1
    for i in range(6):
        i = i+1
        tag = str(i)
        weight_norm = np.asarray(model.get_layer('conv2_block'+tag+'_0_bn').get_weights())
        bn_weight = handle_bn(weight_norm)
        dest[keys[num]].copy_(bn_weight[0])
        dest[keys[num+1]].copy_(bn_weight[1])
        dest[keys[num+2]].copy_(bn_weight[2])
        dest[keys[num+3]].copy_(bn_weight[3])
        load_right(dest, keys[num], bn_weight[0])
        load_right(dest, keys[num+1], bn_weight[1])
        load_right(dest, keys[num+2], bn_weight[2])
        load_right(dest, keys[num+3], bn_weight[3])
        # print(keys[num])
        # print(keys[num+1])
        # print(keys[num+2])
        # print(keys[num+3])
        num = num+5
        weights1 = np.asarray(model.get_layer('conv2_block'+tag+'_1_conv').get_weights())
        weights1 = handle_w(weights1)
        dest[keys[num]].copy_(weights1)
        load_right(dest, keys[num], weights1)
        # print(keys[num])
        num = num+1
        weight_norm = np.asarray(model.get_layer('conv2_block'+tag+'_1_bn').get_weights())
        bn_weight = handle_bn(weight_norm)
        dest[keys[num]].copy_(bn_weight[0])
        dest[keys[num+1]].copy_(bn_weight[1])
        dest[keys[num+2]].copy_(bn_weight[2])
        dest[keys[num+3]].copy_(bn_weight[3])
        load_right(dest, keys[num], bn_weight[0])
        load_right(dest, keys[num+1], bn_weight[1])
        load_right(dest, keys[num+2], bn_weight[2])
        load_right(dest, keys[num+3], bn_weight[3])
        # print(keys[num])
        # print(keys[num + 1])
        # print(keys[num + 2])
        # print(keys[num + 3])
        num = num+5
        weights1 = np.asarray(model.get_layer('conv2_block'+tag+'_2_conv').get_weights())
        weights1 = handle_w(weights1)
        dest[keys[num]].copy_(weights1)
        load_right(dest, keys[num], weights1)
        # print(keys[num])
        num = num+1
    weight_norm = np.asarray(model.get_layer('pool2_bn').get_weights())
    bn_weight = handle_bn(weight_norm)
    dest[keys[num]].copy_(bn_weight[0])
    dest[keys[num + 1]].copy_(bn_weight[1])
    dest[keys[num + 2]].copy_(bn_weight[2])
    dest[keys[num + 3]].copy_(bn_weight[3])
    load_right(dest, keys[num], bn_weight[0])
    load_right(dest, keys[num + 1], bn_weight[1])
    load_right(dest, keys[num + 2], bn_weight[2])
    load_right(dest, keys[num + 3], bn_weight[3])
    # print(keys[num])
    # print(keys[num + 1])
    # print(keys[num + 2])
    # print(keys[num + 3])
    num = num + 5
    weights1 = np.asarray(model.get_layer('pool2_conv').get_weights())
    weights1 = handle_w(weights1)
    dest[keys[num]].copy_(weights1)
    load_right(dest, keys[num], weights1)
    # print(keys[num])
    num = num + 1
    # # dense_block2
    for i in range(12):
        i = i+1
        tag = str(i)
        weight_norm = np.asarray(model.get_layer('conv3_block'+tag+'_0_bn').get_weights())
        bn_weight = handle_bn(weight_norm)
        dest[keys[num]].copy_(bn_weight[0])
        dest[keys[num+1]].copy_(bn_weight[1])
        dest[keys[num+2]].copy_(bn_weight[2])
        dest[keys[num+3]].copy_(bn_weight[3])
        load_right(dest, keys[num], bn_weight[0])
        load_right(dest, keys[num + 1], bn_weight[1])
        load_right(dest, keys[num + 2], bn_weight[2])
        load_right(dest, keys[num + 3], bn_weight[3])
        # print(keys[num])
        # print(keys[num + 1])
        # print(keys[num + 2])
        # print(keys[num + 3])
        num = num+5
        weights1 = np.asarray(model.get_layer('conv3_block'+tag+'_1_conv').get_weights())
        weights1 = handle_w(weights1)
        dest[keys[num]].copy_(weights1)
        load_right(dest, keys[num], weights1)
        # print(keys[num])
        num = num+1
        weight_norm = np.asarray(model.get_layer('conv3_block'+tag+'_1_bn').get_weights())
        bn_weight = handle_bn(weight_norm)
        dest[keys[num]].copy_(bn_weight[0])
        dest[keys[num+1]].copy_(bn_weight[1])
        dest[keys[num+2]].copy_(bn_weight[2])
        dest[keys[num+3]].copy_(bn_weight[3])
        load_right(dest, keys[num], bn_weight[0])
        load_right(dest, keys[num + 1], bn_weight[1])
        load_right(dest, keys[num + 2], bn_weight[2])
        load_right(dest, keys[num + 3], bn_weight[3])
        # print(keys[num])
        # print(keys[num + 1])
        # print(keys[num + 2])
        # print(keys[num + 3])
        num = num+5
        weights1 = np.asarray(model.get_layer('conv3_block'+tag+'_2_conv').get_weights())
        weights1 = handle_w(weights1)
        dest[keys[num]].copy_(weights1)
        load_right(dest, keys[num], weights1)
        # print(keys[num])
        num = num+1
    weight_norm = np.asarray(model.get_layer('pool3_bn').get_weights())
    bn_weight = handle_bn(weight_norm)
    dest[keys[num]].copy_(bn_weight[0])
    dest[keys[num + 1]].copy_(bn_weight[1])
    dest[keys[num + 2]].copy_(bn_weight[2])
    dest[keys[num + 3]].copy_(bn_weight[3])
    load_right(dest, keys[num], bn_weight[0])
    load_right(dest, keys[num + 1], bn_weight[1])
    load_right(dest, keys[num + 2], bn_weight[2])
    load_right(dest, keys[num + 3], bn_weight[3])
    # print(keys[num])
    # print(keys[num + 1])
    # print(keys[num + 2])
    # print(keys[num + 3])
    num = num + 5
    weights1 = np.asarray(model.get_layer('pool3_conv').get_weights())
    weights1 = handle_w(weights1)
    dest[keys[num]].copy_(weights1)
    load_right(dest, keys[num], weights1)
    # print(keys[num])
    #

    num = num + 1
    # # dense_block3
    for i in range(32):
        i = i+1
        tag = str(i)
        weight_norm = np.asarray(model.get_layer('conv4_block'+tag+'_0_bn').get_weights())
        bn_weight = handle_bn(weight_norm)
        dest[keys[num]].copy_(bn_weight[0])
        dest[keys[num+1]].copy_(bn_weight[1])
        dest[keys[num+2]].copy_(bn_weight[2])
        dest[keys[num+3]].copy_(bn_weight[3])
        load_right(dest, keys[num], bn_weight[0])
        load_right(dest, keys[num + 1], bn_weight[1])
        load_right(dest, keys[num + 2], bn_weight[2])
        load_right(dest, keys[num + 3], bn_weight[3])

        num = num+5
        weights1 = np.asarray(model.get_layer('conv4_block'+tag+'_1_conv').get_weights())
        weights1 = handle_w(weights1)
        dest[keys[num]].copy_(weights1)
        load_right(dest, keys[num], weights1)


        num = num+1
        weight_norm = np.asarray(model.get_layer('conv4_block'+tag+'_1_bn').get_weights())
        bn_weight = handle_bn(weight_norm)
        dest[keys[num]].copy_(bn_weight[0])
        dest[keys[num+1]].copy_(bn_weight[1])
        dest[keys[num+2]].copy_(bn_weight[2])
        dest[keys[num+3]].copy_(bn_weight[3])
        load_right(dest, keys[num], bn_weight[0])
        load_right(dest, keys[num + 1], bn_weight[1])
        load_right(dest, keys[num + 2], bn_weight[2])
        load_right(dest, keys[num + 3], bn_weight[3])

        num = num+5
        weights1 = np.asarray(model.get_layer('conv4_block'+tag+'_2_conv').get_weights())
        weights1 = handle_w(weights1)
        dest[keys[num]].copy_(weights1)
        load_right(dest, keys[num], weights1)

        num = num+1

    weight_norm = np.asarray(model.get_layer('pool4_bn').get_weights())
    bn_weight = handle_bn(weight_norm)
    dest[keys[num]].copy_(bn_weight[0])
    dest[keys[num + 1]].copy_(bn_weight[1])
    dest[keys[num + 2]].copy_(bn_weight[2])
    dest[keys[num + 3]].copy_(bn_weight[3])
    load_right(dest, keys[num], bn_weight[0])
    load_right(dest, keys[num + 1], bn_weight[1])
    load_right(dest, keys[num + 2], bn_weight[2])
    load_right(dest, keys[num + 3], bn_weight[3])

    # print(keys[num + 2])
    # print(keys[num + 3])
    num = num + 5
    weights1 = np.asarray(model.get_layer('pool4_conv').get_weights())
    weights1 = handle_w(weights1)
    dest[keys[num]].copy_(weights1)
    load_right(dest, keys[num], weights1)

    num = num + 1

    # dense_block4
    for i in range(32):
        i = i+1
        tag = str(i)
        weight_norm = np.asarray(model.get_layer('conv5_block'+tag+'_0_bn').get_weights())
        bn_weight = handle_bn(weight_norm)
        dest[keys[num]].copy_(bn_weight[0])
        dest[keys[num+1]].copy_(bn_weight[1])
        dest[keys[num+2]].copy_(bn_weight[2])
        dest[keys[num+3]].copy_(bn_weight[3])
        num = num+5
        weights1 = np.asarray(model.get_layer('conv5_block'+tag+'_1_conv').get_weights())
        weights1 = handle_w(weights1)
        dest[keys[num]].copy_(weights1)
        num = num+1
        weight_norm = np.asarray(model.get_layer('conv5_block'+tag+'_1_bn').get_weights())
        bn_weight = handle_bn(weight_norm)
        dest[keys[num]].copy_(bn_weight[0])
        dest[keys[num+1]].copy_(bn_weight[1])
        dest[keys[num+2]].copy_(bn_weight[2])
        dest[keys[num+3]].copy_(bn_weight[3])

        num = num+5
        weights1 = np.asarray(model.get_layer('conv5_block'+tag+'_2_conv').get_weights())
        weights1 = handle_w(weights1)
        dest[keys[num]].copy_(weights1)
        num = num+1
    weight_norm = np.asarray(model.get_layer('bn').get_weights())
    bn_weight = handle_bn(weight_norm)
    dest[keys[num]].copy_(bn_weight[0])
    dest[keys[num + 1]].copy_(bn_weight[1])
    dest[keys[num + 2]].copy_(bn_weight[2])
    dest[keys[num + 3]].copy_(bn_weight[3])

    # print(keys[num + 2])
    # print(keys[num + 3])
    num = num + 5
    # 前面都是没问题的

    # 解码器赋值
    weights1 = np.asarray(model.get_layer('conv2').get_weights())
    weights1,wb = handle_wab(weights1)
    print(keys[num])
    print(keys[num + 1])
    dest[keys[num]].copy_(weights1)
    dest[keys[num+1]].copy_(wb)

    num = num+2
    weights1 = np.asarray(model.get_layer('up1_convA').get_weights())
    weights1,wb = handle_wab(weights1)
    dest[keys[num]].copy_(weights1)
    dest[keys[num+1]].copy_(wb)
    print(keys[num])
    print(keys[num + 1])
    num = num+2
    weights1 = np.asarray(model.get_layer('up1_convB').get_weights())
    weights1,wb = handle_wab(weights1)
    dest[keys[num]].copy_(weights1)
    dest[keys[num+1]].copy_(wb)
    print(keys[num])
    print(keys[num + 1])
    num = num+2
    weights1 = np.asarray(model.get_layer('up2_convA').get_weights())
    weights1, wb = handle_wab(weights1)
    dest[keys[num]].copy_(weights1)
    dest[keys[num + 1]].copy_(wb)
    print(keys[num])
    print(keys[num + 1])
    num = num + 2
    weights1 = np.asarray(model.get_layer('up2_convB').get_weights())
    weights1, wb = handle_wab(weights1)
    dest[keys[num]].copy_(weights1)
    dest[keys[num + 1]].copy_(wb)
    print(keys[num])
    print(keys[num + 1])
    num = num + 2
    weights1 = np.asarray(model.get_layer('up3_convA').get_weights())
    weights1, wb = handle_wab(weights1)
    dest[keys[num]].copy_(weights1)
    dest[keys[num + 1]].copy_(wb)
    print(keys[num])
    print(keys[num + 1])
    num = num + 2
    weights1 = np.asarray(model.get_layer('up3_convB').get_weights())
    weights1, wb = handle_wab(weights1)
    dest[keys[num]].copy_(weights1)
    dest[keys[num + 1]].copy_(wb)
    print(keys[num])
    print(keys[num + 1])
    num = num + 2
    weights1 = np.asarray(model.get_layer('up4_convA').get_weights())
    weights1, wb = handle_wab(weights1)
    dest[keys[num]].copy_(weights1)
    dest[keys[num + 1]].copy_(wb)
    print(keys[num])
    print(keys[num + 1])
    num = num + 2
    weights1 = np.asarray(model.get_layer('up4_convB').get_weights())
    weights1, wb = handle_wab(weights1)
    dest[keys[num]].copy_(weights1)
    dest[keys[num + 1]].copy_(wb)
    print(keys[num])
    print(keys[num + 1])
    num = num + 2
    weights1 = np.asarray(model.get_layer('conv3').get_weights())
    weights1, wb = handle_wab(weights1)
    dest[keys[num]].copy_(weights1)
    dest[keys[num + 1]].copy_(wb)
    print(keys[num])
    print(keys[num + 1])




    # img = sc.imread('H:\\dataset\\contras_ehnce\\testinput\\106_4.png')
    root = 'H:\\dataset\\pair'
    filename1 = root + '\\444nohaze.png'
    img = sc.imread(filename1)
    print(img.shape)
    # input = torch.from_numpy(img)
    input = transforms.ToTensor()(img)
    input = input.float()
    input = torch.unsqueeze(input, 0)
    torch.save(dest,'G:\\my.pth')
    net.load_state_dict(dest)
    net.eval()

    out,finalout = net(input)

    finalout = finalout.detach().numpy()
    finalout = finalout[0, :, :, :]
    finalout = np.swapaxes(finalout, 0, 2)
    finalout = np.swapaxes(finalout, 0, 1)
    np.save('G:/py_thr.npy', finalout)
    out = out.detach().numpy()
    out = out[0, :,:,:]
    result = out
    # result = out.detach().numpy()
    result = np.swapaxes(result, 0, 2)
    result = np.swapaxes(result, 0, 1)
    result = result[:, :, 0]
    print(np.max(result))
    # result = 10 / result
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    sc.imsave('G:\\show.png',result)
    plt.imshow(result)
    plt.show()



    # dense_block1.denselayer1 赋值
def load_right(dest,name,tensor):

    bt = (dest[name] == tensor)
    bt = bt.detach().numpy()
    if not(bt.any()==True):
       print(name)
       print('error')
       exit(-1)
    print('same')


    # for idx in range(1015, 1015 + 6):
    #     dest.get(keys[idx])
def handle_wab(weight):
    weights1 = weight[0]
    weights1 = np.transpose(weights1, [3, 2, 0, 1])
    weights1 = torch.from_numpy(weights1).to(torch.float32)
    weight_b = weight[1]
    weight_b = torch.from_numpy(weight_b).to(torch.float32)
    return weights1,weight_b

def handle_w(weights1):
    weights1 = weights1[0,:,:,:,:]
    weights1 = np.transpose(weights1, [3, 2, 0, 1])
    weights1 = torch.from_numpy(weights1).to(torch.float32)
    return weights1

def handle_bn(weight_norm):
    bn_w = weight_norm[0, :]
    bn_w = torch.from_numpy(bn_w).to(torch.float32)
    bn_b = weight_norm[1, :]
    bn_b = torch.from_numpy(bn_b).to(torch.float32)
    bn_m = weight_norm[2, :]
    bn_m = torch.from_numpy(bn_m).to(torch.float32)
    bn_v = weight_norm[3, :]
    bn_v = torch.from_numpy(bn_v).to(torch.float32)
    return bn_w,bn_b,bn_m,bn_v

def same(arr1, arr2):
    # type: (np.ndarray, np.ndarray) -> bool
    # 判断shape是否相同
    assert arr1.shape == arr2.shape
    # 对应元素相减求绝对值
    diff = np.abs(arr1 - arr2)
    gap = np.max(diff)
    print(gap)
    if gap > 1e-3:
        print(gap)
        print('no same')
        return False
    return True
    # 判断是否有任意一个两元素之差大于阈值1e-5

def testresult():
    keras_result = np.load('G:/keras_mid.npy')
    # print(keras_result.shape)
    # keras_result = np.swapaxes(keras_result,0,1)
    torch_result = np.load('G:/py_mid.npy')
    # # same函数之前有提到
    print(same(keras_result, torch_result))

if __name__ == '__main__':
     # stact = torch.load('G:\\my.pth', map_location='cpu')
     # # midw = stact['midconv.weight']
     # custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
     #
     # print('Loading model...')
     #
     # # Load model into GPU / CPU
     # model = load_model(args.model, custom_objects=custom_objects, compile=False)
     # weights1 = np.asarray(model.get_layer('conv2').get_weights())
     # weights1, wb = handle_wab(weights1)
     # print(wb)
     # load_right(stact,'midconv.bias',wb)

     # testresult()
     torch.manual_seed(0)
     test()
     # keras_2pytorch()