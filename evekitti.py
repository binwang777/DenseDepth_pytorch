import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import os
import glob
import h5py
import matplotlib.pyplot as plt
import argparse
import scipy.misc
from keras.models import load_model
from util.utils import predict, load_images, display_images
from matplotlib import pyplot as plt
import PIL.Image as pil
import util.color_compl
depth_path = 'G:/kitti_depth/train/'
kitti_path = 'G:/kitti/'
def DepthNorm(x, maxDepth):
    return maxDepth / x




def predict(model, images, minDepth=10, maxDepth=1000, batch_size=1):
    # Support multiple RGBs, one RGB image, even grayscale
    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    showp = predictions[0,:,:,0]
    print(predictions.shape)
    plt.imshow(showp)
    plt.show()
    # Put in expected range
    return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth


def load_modelm():
    # Argument Parser
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--model', default='H:/dataset/code/DenseDepth-master/kitti.h5', type=str, help='Trained Keras model file.')
    parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
    args = parser.parse_args()

    # Custom object needed for inference and training
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

    print('Loading model...')

    # Load model into GPU / CPU
    model = load_model(args.model, custom_objects=custom_objects, compile=False)
    return model



def scale_up(scale, images):
    from skimage.transform import resize
    scaled = []

    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True))

    return np.stack(scaled)



def compute_errors1(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    print(np.max(gt))
    print(np.max(pred))
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

import skimage.measure
from skimage.transform import resize
def evalkitti():
    model = load_modelm()
    eval_split = 'eigen'
    # eval_split = ''

    # txtpath = 'H:/dataset/code/monodepth2-master/splits/eigen/test_files.txt'
    txtpath = 'H:/dataset/code/monodepth2-master/splits/eigen_benchmark/test_files.txt'


    f = open(txtpath, "r")  # 设置文件对象

    all_abs_rel = 0,
    all_sq_rel=0,
    all_rms = 0,
    all_log_rms = 0,
    all_a1 = 0,
    all_a2 = 0,
    all_a3 = 0
    data = []
    for line in open(txtpath, "r"):
        data.append(line)
    num = 0
    for spilts in data:
        spilts = spilts.strip('\n')
        sps = spilts.split(' ')
        input_path =''
        gtdepth_path = ''

        if sps[2] == 'l':
            tag = sps[1]
            tag = tag.zfill(10)
            input_path = kitti_path+sps[0]+'/image_02/data/'+tag+'.png'
            dets = sps[0].split('/')
            gtdepth_path = depth_path+dets[1]+'/proj_depth/groundtruth/image_02/'+tag+'.png'
            # print(input_path)
        if sps[2] == 'r':
            tag = sps[1]
            tag = tag.zfill(10)
            input_path = kitti_path + sps[0] + '/image_03/data/' + tag + '.png'
            dets = sps[0].split('/')
            gtdepth_path = depth_path + dets[1] + '/proj_depth/groundtruth/image_03/' + tag + '.png'
        x1 = os.path.exists(input_path)
        x2 = os.path.exists(gtdepth_path)

        if not x1:
            print('no input flie:'+spilts)
            continue
        if not x2:
            print('no depth flie:' + spilts)
            continue

        num = num + 1
        print('-------------------------------------' + str(num))

        # 获取输入
        img = scipy.misc.imread(input_path)
        img = scipy.misc.imresize(img,[384,1280])/255
        predepth = predict(model, img,minDepth=10,maxDepth=8000)
        predepth = scale_up(2, predepth[:, :, :, 0])
        predepth = predepth[0, :, :]
        # 存储直接输出，方便下次评估


        # 加载gt
        spotdepth = pil.open(gtdepth_path)

        gt_depth = spotdepth.resize([1280,384], pil.NEAREST)
        spsave = np.asarray(gt_depth)
        gt_depth = np.asarray(gt_depth).astype(np.float32)/256

        # h5f = h5py.File('G:/gen/' + str(num) + '.h5', 'w')
        # # 转存的预测深度没有进行加工
        # h5f.create_dataset('predepth', data=predepth)
        # # 转存的真实深度经过加工
        # h5f.create_dataset('gtdepth', data=gt_depth)
        # # 转存的为resize的原始点云图
        # h5f.create_dataset('odepth', data=spsave)
        # # 转存的为经过归一化的输入图像
        # h5f.create_dataset('input', data=img)
        garg_crop = True
        eigen_crop = True
        if eval_split == "eigen":
            mask = np.logical_and(gt_depth > 1e-3, gt_depth < 80)
            # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
            # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
            gt_height, gt_width = gt_depth.shape
            if garg_crop:
                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            # crop we found by trial and error to reproduce Eigen NIPS14 results
            elif eigen_crop:
                crop = np.array([0.3324324 * gt_height, 0.91351351 * gt_height,
                                 0.0359477 * gt_width, 0.96405229 * gt_width]).astype(np.int32)

            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
            scalor = np.median(gt_depth[mask])/np.median(predepth[mask])
        else:
            mask = gt_depth > 1e-3
            scalor = 80

        predepth[mask] *= scalor
        # predepth *= 80
        predepth[predepth < 1e-3] = 1e-3
        predepth[predepth > 80] = 80



        print('max predepth:%.3f---------max gtdepth:%.3f'%(np.max(predepth[mask]),np.max(gt_depth[mask])))

        a1, a2, a3, abs_rel, rmse, log_10 = compute_errors1(gt_depth[mask], predepth[mask])
        # abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = \
        #     compute_errors1(gt_depth[mask], predepth[mask])

        all_abs_rel+=abs_rel
        all_log_rms+=log_10
        # all_sq_rel +=sq_rel
        all_rms+=rmse
        all_a1+=a1
        all_a2+=a2
        all_a3+=a3
        # all_sq_rel / num,sq_rel:%.5f,
        print('mean error:')
        print('abs_rel:%.5f,   rmse :%.5f,    log_10:%.5f,    a1:%.5f,   a2:%.5f,    a3:%.5f'%(all_abs_rel/num, all_rms/num, all_log_rms/num,  all_a1/num,all_a2/num,all_a3/num))
        print('-------------------------------------' + str(num))
        print('I')
        # print(abs_rel, sq_rel, rms, log_rms, a1, a2, a3)
    f.close()

if __name__ == '__main__':
    evalkitti()