import torch
import torch.nn as nn
from torch.autograd import Variable
#import tensorflow as tf
#import keras.backend as K
import numpy as np
import torch.nn.functional as F
import kornia


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

def gradient_loss(gen_frames, gt_frames, alpha=1):

    def gradient(x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top 
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    # gradient
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    #
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)
    # condense into one tensor and avg
    return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss,self).__init__()

    def forward(self,pred,target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
   
        maxRatio = torch.max(pred / target, target / pred)
        l_delta = 1.0 - float((maxRatio < 1.25).float().mean())

        smooth_l1 = torch.nn.SmoothL1Loss().cuda()
        l_depth = smooth_l1(pred, target)
       
        inversedepthsmooth = kornia.losses.InverseDepthSmoothnessLoss().cuda()
        l_inverse = inversedepthsmooth(pred,target)

        ssim_loss = kornia.losses.SSIM(11).cuda()
        l_ssim = ssim_loss(pred,target)
        l_ssim = (1 - l_ssim.mean())
     
        # Weights
        w1 = 10.0
        w2 = 10.0
        w3 = 0.7
        
        #return (w1 * l_ssim) + (w2 * l_delta) + (w3 * l_depth)
        return (w3 * l_depth) + (w1*l_ssim)
