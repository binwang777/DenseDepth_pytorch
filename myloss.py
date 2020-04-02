import torch
import torch.nn as nn
from util import mssim_loss
criterionCAE = nn.L1Loss()

def gradient(y):

    gradient_h = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_y = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

    return gradient_h, gradient_y

class depthloss(torch.nn.Module):
    def __init__(self):
        super(depthloss, self).__init__()
        self.maxDepthVal = 1000.0 / 10.0

    def forward(self, pred, depth):
        w1 = 1.0
        w2 = 1.0
        w3 = 0.1

        l_ssim = torch.clamp((1 - mssim_loss.ssim(pred, depth, val_range=self.maxDepthVal)) * 0.5, 0, 1)

        loss1 = criterionCAE(pred,depth)

        gradie_h_est, gradie_v_est = gradient(pred)
        gradie_h_gt, gradie_v_gt = gradient(depth)
        L_h = criterionCAE(gradie_h_est, gradie_h_gt)
        L_v = criterionCAE(gradie_v_est, gradie_v_gt)
        grad = L_h+L_v
        L_all = w1*l_ssim+w2*grad+w3*loss1

        return L_all








