from __future__ import print_function

import argparse
import os

import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from model import DTTDnet
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from model.DTTDnet import DenseNet_pytorch
from torch.autograd import Variable
from util import *
from util import lr_scheduler as lrs
from tqdm import tqdm
from model import myloss
from  dataloader import NYUdata_loader as dataloader


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
                    default='SCIE', help='')
parser.add_argument('--dataroot', required=False,
                    default='', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
                    default='', help='path to val dataset')
parser.add_argument('--nesterov', '-n', action='store_true', help='enables Nesterov momentum')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=16, help='input batch size')
parser.add_argument('--originalSize', type=int,
                    default=256, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
                    default=256, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int,
                    default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
                    default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
parser.add_argument('--poolSize', type=int, default=50,
                    help='Buffer size for storing previously generated samples from G')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netDetail', default='/home/ubuntu/PycharmProjects/Pytorch_ECTDnet/sample/DetailNetnet_epoch_26.pth', help="path to netG (to continue training)")
parser.add_argument('--netL', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--evalIter', type=int, default=50,
                    help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']= '0'
print(opt)
os.makedirs(opt.exp)
# 设置随机种子
opt.manualSeed = random.randint(1, 10000)
# opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

torch.cuda.manual_seed_all(opt.manualSeed)

print("Random Seed: ", opt.manualSeed)

opt.workers = 1



# get logger
# window:
trainLogger = open('%s/train.log' % opt.exp, 'w')



whatnet = 'DetailNet'
ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize = opt.outputChannelSize

opt.netDetail = ''
net = DenseNet_pytorch(inputChannelSize,outputChannelSize)
print(opt.netDetail)
print(net)


use_cuda = torch.cuda.is_available()

if use_cuda:
    net.cuda()
    assert torch.cuda.device_count() == 1, 'only support single gpu'
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


trainset = dataloader.NYUDataset(root_dir='',split='train')
valset = dataloader.NYUDataset(root_dir='',split='val')


trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=opt.batchSize,
                                          shuffle=True,
                                          num_workers=2,
                                          pin_memory=True,
                                          drop_last=True)
valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=opt.valBatchSize,
                                            shuffle=False,
                                            num_workers=2,
                                            pin_memory=True,
                                            drop_last=True)


MAE = nn.L1Loss()
depthloss = myloss.depthloss()
depthloss = depthloss.cuda()
MAE.cuda()

# get optimizer

optimizer = optim.Adam(net.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999), weight_decay=0.00005)


scheduler = lrs.ReduceLROnPlateau(optimizer, 'min') # set up scheduler
matlabpath = 'C:\Program Files\Polyspace\R2019a'
def train(epoch):
    net.train()
    depthloss_a = 0.0
    tranloss_a = 0.0
    tbar = tqdm(trainloader)
    for i, sample in enumerate(tbar):
        [haze, depth] = [sample['haze'], sample['depth']]
        if use_cuda:
            haze, depth,tran = haze.cuda(), depth.cuda()
        optimizer.zero_grad()
        haze, depth = Variable(haze), Variable(depth)
        depth_p,tran_p = net(haze)
        depth_loss = depthloss(depth, depth_p)
        depth_loss.backward(retain_graph=True)


        trainLogger.write('%d\t%d\t depthloss:%f ,\t\n ' % (epoch, i, depthloss.item()))
        trainLogger.flush()
        optimizer.step()
        depthloss_a += depthloss.item()

        error_str = 'Epoch: %d, depth_loss=%.4f' % (epoch, depthloss_a / (i + 1))
        tbar.set_description(error_str)
    if epoch % 2 == 0:
        torch.save(net.module.state_dict(), '%s/Densedepth_NET%d.pth' % (opt.exp, epoch))

def val(epoch):
    net.eval()
    eval_loss = 0.0
    tran_loss = 0.0
    # NOTE valing loop
    count = 0
    tbar = tqdm(valloader)
    for i, sample in enumerate(tbar):
        [haze, depth] = [sample['haze'], sample['depth']]
        with torch.no_grad():
            if use_cuda:
                haze, depth = haze.cuda(), depth.cuda()
            haze, depth = Variable(haze,volatile=True), Variable(depth)
            depth_p,tran_p = net(haze)
        depth_loss = depthloss(depth, depth_p)

        loss = depth_loss.data.cpu()

        eval_loss += loss.item()

        count = count+1
        error_str = 'Epoch: %d, depth_val_loss=%.4f' % (epoch, eval_loss / (i + 1))
        tbar.set_description(error_str)
    # print(count)
    avg_loss = eval_loss/(count*opt.valBatchSize)
    scheduler.step(avg_loss, epoch)


def train_val():
    for epoch in range(opt.niter):
        train(epoch)
        val(epoch)
    trainLogger.close()

if __name__ == '__main__':
    train_val()