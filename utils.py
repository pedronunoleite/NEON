import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import dataloaders.read_flow_lib as read_flow_lib_

cmap = plt.cm.jet
#cmap = plt.cm.viridis

def parse_command():
    
    model_names = ['squeezenet','shufflenetv2','vgg11','densenet121','densenet121_skipadd' ,
                   'resnet18', 'resnet50', 'resnet18skipadd', 'resnet18skipadd_dw', 'resnet18skipconcat',
                   'mobilenet', 'mobilenetskipadd', 'mobilenetskipconcat'] 

    loss_names = ['l1', 'l2', 'custom', 'smoothl1', 'inversedepthsmoothness'] 
    data_names = ['nyudepthv2', 'kitti', 'kitti_eigen'] 
    from models import Decoder
    decoder_names = Decoder.names
    print(decoder_names)
    modality_names = ['rgb_flow', 'rgb_flownet', 'yuv_flow', 'rgb_flow_edges', 'yuv_flow_edges', 'rgb', 'flow', 'flownet', 'flow_edges']

    import argparse
    parser = argparse.ArgumentParser(description='Parallax-Depth')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18skipadd', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')

    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb_flownet', choices=modality_names,
                        help='modality: ' + ' | '.join(modality_names) + ' (default: rgb_flow)')

    parser.add_argument('--data', metavar='DATA', default='kitti_eigen',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: kitti_eigen)')

    parser.add_argument('--decoder', '-d', metavar='DECODER', default='nnconv5dw', choices=decoder_names,
                        help='decoder: ' + ' | '.join(decoder_names) + ' (default: nnconv5dw)')

    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')

    parser.add_argument('--epochs', default=12, type=int, metavar='N',
                        help='number of total epochs to run (default: 5)')

    parser.add_argument('-c', '--criterion', metavar='LOSS', default='smoothl1', choices=loss_names,
                        help='loss function: ' + ' | '.join(loss_names) + ' (default: l2)')

    parser.add_argument('-b', '--batch-size', default=8, type=int, help='mini-batch size (default: 8)')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate (default 0.1)')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')

    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='',
                        help='evaluate model on validation set')

    parser.add_argument('--no-pretrain', dest='pretrained', action='store_false',
                        help='not to use ImageNet pre-trained weights')

    parser.add_argument('--min_depth', default=1e-5, type=float)

    parser.add_argument('--max_depth', default=80.0, type=float)

    parser.set_defaults(pretrained=False)

    args = parser.parse_args()
    return args

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def adjust_learning_rate(optimizer, epoch, lr_init):
    """Sets the learning rate to the initial LR decayed by 1/10 every 3 epochs"""
    lr = lr_init * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_output_directory(args):
    output_directory = os.path.join('results',
        '{}.modality={}.arch={}.decoder={}.criterion={}.lr={}.bs={}.pretrained={}'.
        format(args.data, args.modality, \
            args.arch, args.decoder, args.criterion, args.lr, args.batch_size, \
            args.pretrained))
    return output_directory

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
   
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.vstack([rgb, depth_target_col, depth_pred_col])    
    return img_merge

def merge_into_row_with_gt(input, flow, depth_target, depth_pred, pretrained, modality):
    if not modality == 'flownet':
        rgb = np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    else:
        rgb = input

    #revert normalization
    if pretrained:
        rgb[:,:,0] = rgb[:,:,0] * 0.229
        rgb[:,:,0] = rgb[:,:,0] + 0.485
        rgb[:,:,1] = rgb[:,:,1] * 0.224
        rgb[:,:,1] = rgb[:,:,1] + 0.456
        rgb[:,:,2] = rgb[:,:,2] * 0.225
        rgb[:,:,2] = rgb[:,:,2] + 0.406

    rgb = 255 *rgb

    if modality == 'rgb':
        flow_cpu = flow
        flow_input_col = colored_depthmap(flow_cpu, 0, 80)
    elif (modality == 'rgb_flownet' or modality == 'flownet'):
        flow_cpu = np.squeeze(flow.cpu().numpy())
        flow_cpu = cv2.normalize(flow_cpu,None,0,255,cv2.NORM_MINMAX)
        flow_input_col = cv2.cvtColor(flow_cpu,cv2.COLOR_GRAY2RGB)
    else:
        flow_cpu = 255 * np.squeeze(flow.cpu().numpy())
        flow_input_col = cv2.cvtColor(flow_cpu,cv2.COLOR_GRAY2RGB)

    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    #d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    #d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    d_min = 0
    d_max = 80
    #flow_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    #flow_input_col   = cv2.applyColorMap(flow_cpu.astype('uint8'), cv2.COLORMAP_JET)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col   = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.vstack([rgb, flow_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)
