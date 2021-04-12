import os
import time
import csv
import numpy as np
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True
from torchvision import transforms

import models
from metrics import AverageMeter, Result
import criteria
import utils
import kornia
import dataloaders.read_flow_lib as read_flow_lib_

args = utils.parse_command()
print(args)

#Metrics
fieldnames = ['epochs', 'mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time', 'irmse', 'imae', 'silog']
#Init results               
best_result = Result()
best_result.set_to_worst()

def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    traindir = '/media/cras4/Server/Pedro/External_DIsk/KITTI_eigen_split/train'
    valdir   = '/media/cras4/Server/Pedro/External_DIsk/KITTI_eigen_split/valid'
    train_loader = None
    val_loader   = None

    args.modality = 'rgb_flownet'
    
    if (args.data == 'kitti'):
        from dataloaders.kitti_dataloader import KITTIDataloader
        if not args.evaluate:
            train_dataset = KITTIDataloader(traindir, type='train', modality=args.modality) 
        val_dataset = KITTIDataloader(valdir, type='val', modality=args.modality)

    elif (args.data == 'kitti_eigen'):
        from dataloaders.kitti_eigen_dataloader import KITTI_Eigen_Dataloader
        if not args.evaluate:
             train_dataset = KITTI_Eigen_Dataloader(traindir, type='train', modality=args.modality) 
        val_dataset = KITTI_Eigen_Dataloader(valdir, type='val', modality=args.modality)

    else:
        raise RuntimeError('Dataset not found.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True, sampler=None,
                                                   worker_init_fn=lambda work_id:np.random.seed(work_id))

    print("=> data loaders created.")
    return train_loader, val_loader

def main():
    global args, best_result, output_directory, train_csv, test_csv

    # evaluation mode
    start_epoch = 0
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no best model found at '{}'".format(args.evaluate)
        print("=> loading best model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        output_directory = os.path.dirname(args.evaluate)
        args = checkpoint['args']
        print(args)
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        _, val_loader = create_data_loaders(args)
        _, _ = validate(val_loader, model, checkpoint['epoch'], write_to_file=False)       
        return

    # optionally resume from a checkpoint
    elif args.resume:
        chkpt_path = args.resume
        assert os.path.isfile(chkpt_path), \
            "=> no checkpoint found at '{}'".format(chkpt_path)
        print("=> loading checkpoint '{}'".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']        
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        optimizer = torch.optim.SGD(model.parameters(), args.lr, 0, weight_decay=args.weight_decay)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #optimizer = torch.optim.Adam(model.parameters(),args.lr,weight_decay = args.weight_decay)
        #optimizer = torch.optim.Adadelta(model.parameters(),args.lr,weight_decay = args.weight_decay)
        output_directory = os.path.dirname(os.path.abspath(chkpt_path))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        train_loader, val_loader = create_data_loaders(args)

    # create new model
    else:
        train_loader, val_loader = create_data_loaders(args)
        print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))

        #set input channels
        if (args.modality == 'rgb_flow') or (args.modality == 'rgb_flownet') or (args.modality == 'yuv_flow'):
            in_channels = 4
        elif (args.modality == 'rgb_flow_edges') or (args.modality == 'yuv_flow_edges'):
            in_channels = 5
        elif (args.modality == 'rgb'):
            in_channels = 3
        elif (args.modality == 'flow_edges'):
            in_channels = 2
        elif (args.modality == 'flow' or args.modality == 'flownet'):
            in_channels = 1
        else:
            print("Redefine the modality")

        #create model
        if (args.arch == 'resnet50'):
            model = models.ResNet(layers=50, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif (args.arch == 'resnet18'):
            model = models.ResNet(layers=18, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif (args.arch == 'mobilenet'):
            model = models.MobileNet(decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif (args.arch == 'resnet18skipadd'):
            model = models.ResNetSkipAdd(layers=18, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif (args.arch == 'resnet18skipadd_dw'):
            model = models.ResNetSkipAdd(layers=18, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif (args.arch == 'resnet18skipconcat'):
            model = models.ResNetSkipConcat(layers=18, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif (args.arch == 'mobilenetskipadd'):
            model = models.MobileNetSkipAdd(output_size=train_loader.dataset.output_size, pretrained=args.pretrained)
        elif (args.arch == 'mobilenetskipconcat'):
            model = models.MobileNetSkipConcat(output_size=train_loader.dataset.output_size, pretrained=args.pretrained)
        elif (args.arch == 'densenet121'):
            model = models.DenseNet(layers=121, decoder=args.decoder, output_size = (192, 1120), in_channels=4)
        elif (args.arch == 'densenet121_skipadd'):
            model = models.DenseNet_SkipAdd(layers=121, output_size = (192, 1120), in_channels=4)
        elif (args.arch == 'vgg11'):
            model = models.VGG(layers=11, decoder=args.decoder, output_size = (192, 1120), in_channels=4)
        elif (args.arch == 'shufflenetv2'):
            model = models.ShuffleNetV2(decoder=args.decoder, output_size = (192, 1120), in_channels=4)
        elif (args.arch == 'squeezenet'):
            model = models.SqueezeNet(decoder=args.decoder, output_size = (192, 1120), in_channels=4)
        else:
            print("Redefine the Architecture")


        print("=> model created.")
        #set optimizer
        #optimizer = torch.optim.Adadelta(model.parameters(),1,weight_decay = args.weight_decay)
        #optimizer = torch.optim.Adam(model.parameters(),args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.SGD(model.parameters(),args.lr,weight_decay=args.weight_decay, momentum=0)

        # model = torch.nn.DataParallel(model).cuda() # for multi-gpu training
        # select defined gpu
        #os.environ['CUDA_VISIBLE_DEVICES'] = 0
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.cuda()

    # define loss function (criterion) 
    if args.criterion == 'l2':
        criterion = criteria.MaskedMSELoss().cuda()
    elif args.criterion == 'l1':
        criterion = criteria.MaskedL1Loss().cuda()
    elif args.criterion == 'custom':
        criterion = criteria.CustomLoss().cuda()
    elif args.criterion == 'smoothl1':
        criterion = torch.nn.SmoothL1Loss().cuda()
    elif args.criterion == 'inversedepthsmoothness':
        criterion = kornia.losses.InverseDepthSmoothnessLoss().cuda()

    # create results folder, if not already exists
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')

    # create new csv files with only header
    if not args.resume:
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for epoch in range(start_epoch, args.epochs):
        utils.adjust_learning_rate(optimizer, epoch, args.lr)   #uncomment if optimizer is not adadelta
        train(train_loader, model, criterion, optimizer, epoch) # train for one epoch
        result, img_merge = validate(val_loader, model, epoch) # evaluate on validation set



        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\nirmse={:.3f}\nimae={:.3f}\nsilog={:.4f}\n".
                    format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time, result.irmse, result.imae, result.silog))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)

        #print(optimizer)

        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'arch': args.arch,
            'model': model,
            'best_result': best_result,
            'optimizer' : optimizer.state_dict(),
        }, is_best, epoch, output_directory)

def train(train_loader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    model.train() # switch to train mode
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        #assert target limits
        target.data[target.data < args.min_depth] = args.min_depth
        target.data[target.data > args.max_depth] = args.max_depth
                
        # compute pred
        end = time.time()
        pred = model(input)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward() # compute gradient and do step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end    

        #assert pred limits
        pred.data[pred.data < args.min_depth] = args.min_depth
        pred.data[pred.data > args.max_depth] = args.max_depth
       

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '
                  'iRMSE={result.irmse:.3f}({average.irmse:.3f})'
                  'iMAE={result.imae:.3f}({average.imae:.3f})'
                  'SIlog={result.silog:.4f}({average.silog:.4f})'.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'epochs': epoch, 'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time, 'irmse':avg.irmse, 'imae':avg.imae, 'silog':avg.silog})


def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):        
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end
                
        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        #assert limits
        pred.data[pred.data < args.min_depth] = args.min_depth
        pred.data[pred.data > args.max_depth] = args.max_depth
        target.data[target.data < args.min_depth] = args.min_depth
        target.data[target.data > args.max_depth] = args.max_depth

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()
             
        # save 8 images for visualization
        skip = 5
        #only preped for rgb/yuv _ flow
        if not args.modality == 'flownet':
            rgb   = input[:,:3,:,:]

        if args.modality == 'rgb':
            flow = np.zeros((192,1120))
        elif args.modality == 'flownet':
            rgb = np.zeros((192,1120,3))
            flow = input
        elif args.modality == 'rgb_flow' or args.modality == 'rgb_flownet' or args.modality == 'yuv_flow':
            flow  = input[:,3:,:,:]    
        elif args.modality == 'rgb_flow_edges':
            flow  = input[:,3:4,:,:] 
            edges = input[:,4:,:,:]  

        gt   = np.squeeze(target.cpu().numpy())
        pred_ = np.squeeze(pred.data.cpu().numpy())
        
        valid_mask = gt > 0
        error = np.zeros(gt.shape)
        error[valid_mask] = gt[valid_mask] - pred_[valid_mask]
        error = np.abs(error)
        error = error ** 2
        error = np.sqrt(error)
        error_color  = utils.colored_depthmap(error.astype('uint8'),np.min(error[valid_mask]),np.max(error[valid_mask])) 
       
        #save plots
        output_directory = utils.get_output_directory(args)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        #print(output_directory)

     
        #save plots
        save_figures = False
        if (i == 29 or i == 39 or i == 44 or i == 173 or i == 187 or i == 285 or i == 493 or i == 581) and save_figures:
            import matplotlib.pyplot as plt
           
            plt.imshow(gt, vmin=0, vmax=80, cmap='jet', aspect='equal')
            plt.colorbar(orientation="horizontal", pad=0.005)
            plt.axis('off')
            filename  = output_directory + '/pred_colorbar_' + str(i) + '.png'
            plt.savefig(filename, bbox_inches='tight', dpi=1200)   
            plt.clf()

            plt.imshow(error, vmin=np.min(error[valid_mask]), vmax=np.max(error[valid_mask]), cmap='jet', aspect='equal')
            plt.colorbar(orientation="horizontal", pad=0.005)
            plt.axis('off')
            filename  = output_directory + '/error_colorbar_' + str(i) + '.png'
            plt.savefig(filename, bbox_inches='tight',dpi=1200)  
            plt.clf()

        if (i%skip == 0) and (result.rmse < best_result.rmse) :
            img_merge = utils.merge_into_row_with_gt(rgb, flow, target, pred, args.pretrained, args.modality)

            if args.modality == 'rgb_flow_edges':
                edges = 255 * np.squeeze(edges.cpu().numpy())
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                img_merge = np.vstack([img_merge,edges])

            img_merge = np.vstack([img_merge, error_color])
            filename  = output_directory + '/comparison_' + str(i) + '.png'
            utils.save_image(img_merge, filename)
                

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '
                  'iRMSE={result.irmse:.3f}({average.irmse:.3f})'
                  'iMAE={result.imae:.3f}({average.imae:.3f})'
                  'SIlog={result.silog:.4f}({average.silog:.4f})'.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        'iRMSE={average.irmse:.3f}\n'
        'iMAE={average.imae:.3f}\n'
        'SIlog={average.silog:.4f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'epochs': epoch, 'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time, 'irmse':avg.irmse, 'imae':avg.imae, 'silog':avg.silog})
    return avg, img_merge

if __name__ == '__main__':
    main()
