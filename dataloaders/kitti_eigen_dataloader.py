import os
import os.path
import numpy as np
import torch.utils.data as data
import cv2
import glob2
import dataloaders.transforms as transforms_
import dataloaders.read_flow_lib as read_flow_lib_
from PIL import Image
import random
from torchvision import transforms
from scipy.ndimage.interpolation import rotate
import albumentations


to_tensor = transforms_.ToTensor()

def enhance_edges(img,flag):
    if flag == False:
        return img
    else:
        img = cv2.cvtColor(img,cv2.COLOR_YUV2RGB)
        edges = get_canny_edge(img)
        Y = cv2.addWeighted(img[:,:,0],0.9,edges,0.3,0)
        img[:,:,0] =  Y
        return img

def get_canny_edge(img):
    #edges = np.zeros(img.shape)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5 * high_thresh
    edges = cv2.Canny(img, lowThresh, high_thresh)
    return edges

def random_rotation(img, flow, gt, edges):
    angle = np.random.uniform(-5.0, 5.0)
    img_rot = rotate(img, angle=angle, reshape=False)
    flow_rot = rotate(flow, angle=angle, reshape=False)
    gt_rot = rotate(gt, angle=angle, reshape=False)
    edges_rot = rotate(edges, angle=angle, reshape=False)
    return img_rot, flow_rot, gt_rot, edges_rot

def flip_h(img, flow, gt, edges):
    img_flip   = (img[:, ::-1]).copy()
    flow_flip  = (flow[:, ::-1]).copy()
    gt_flip    = (gt[:, ::-1]).copy()
    edges_flip = (edges[:, ::-1]).copy()
    return img_flip, flow_flip, gt_flip, edges_flip

def image_augment(p = 0.85):
    return albumentations.Compose([ 
       
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=3, p=0.4),
            albumentations.MedianBlur(blur_limit=3, p=0.2),
            albumentations.Blur(blur_limit=3, p=0.4),
        ], p=0.4),

        albumentations.OneOf([
            #albumentations.IAASharpen(p = 0.3),
            #albumentations.CLAHE(clip_limit=2, p=0.1),
            albumentations.RandomBrightnessContrast(brightness_limit=0.3,p=0.4),
            albumentations.RandomGamma(p=0.2),
        ], p=0.5),
        
        albumentations.OneOf([
            albumentations.RGBShift(p=0.5),
            albumentations.ChannelShuffle(p=0.5),
        ], p=0.1),
       
        albumentations.GaussNoise(p=0.1)    

    ], p= p)

def random_crop(img, flow, depth, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == depth.shape[0] == flow.shape[0]
    assert img.shape[1] == depth.shape[1] == flow.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img   = img[y:y + height, x:x + width]
    flow  = flow[y:y + height, x:x + width]
    depth = depth[y:y + height, x:x + width]        
    return img, flow, depth

def crop_center(img,cropx,cropy):
    print(img.shape)
    y = img.shape[1]
    x = img.shape[0]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)       
    return img[startx:startx+cropx,starty:starty+cropy]

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth

def make_dataset(type,modality):
    imgs_dir = '/media/cras4/Server/Pedro/External_DIsk/KITTI_eigen_split/imgs/'
    gt_dir   = '/media/cras4/Server/Pedro/External_DIsk/KITTI_eigen_split/gt_dense/'
    gt_sparse_dir   = '/media/cras4/Server/Pedro/External_DIsk/KITTI_eigen_split/gt/'
    
    if (modality == 'rgb_flownet' or modality == 'flownet'):
        stereo_flow_dir = '/media/cras4/Server/Pedro/External_DIsk/KITTI_eigen_split/flownet/'
    else:
        stereo_flow_dir = '/media/cras4/Server/Pedro/External_DIsk/KITTI_eigen_split/stereo_flow/'

    if(type == 'train'):        
        f = open('/home/cras4/Pytorch_ws/Parallax-Depth/NEON_github/dataloaders/eigen_train_split.txt', 'r')
        x = f.readlines()
    elif (type == 'val'):     
        f = open('/home/cras4/Pytorch_ws/Parallax-Depth/NEON_github/dataloaders/eigen_test_split.txt', 'r')
        x = f.readlines()

    item_vec = []
    for i in range(len(x)):
       vec = x[i].split(' ')
       img = imgs_dir + vec[0]
       if (modality == 'rgb_flownet' or modality == 'flownet'):
            stereo_flow = stereo_flow_dir + vec[1][:-4] + '.flo'
       else:
            stereo_flow = stereo_flow_dir + vec[1]

       #for dense gt:
       gt  = gt_dir + vec[2][:-5] + '.npy'

       #for sparse gt:
       #gt  = gt_sparse_dir + vec[2][:-1]
       #print(gt)

       #for sparse gt
       #if('None' in gt):          
       #   continue

       #for dense gt:
       if not (os.path.exists(gt)):
           continue


       item = img, stereo_flow, gt       
       item_vec.append(item) 

    return item_vec[:]
    
def load_kitti_eigen(item, modality):
    img_path, flow_path, gt_path = item
    
    img  = np.array(cv2.imread(img_path,cv2.IMREAD_COLOR))
    img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if ('yuv' in modality):
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) 
        img  = enhance_edges(img, True) 

    img  = np.array(img)

    if (modality == 'rgb_flownet' or modality == 'flownet'):
        flow = read_flow_lib_.read_flow(flow_path)
        flowx = flow[:,:,0]
        flowy = flow[:,:,1] 
        #obtain the magnitude
        flow_mag, flow_ang = cv2.cartToPolar(flowx,flowy)
        flow = flow_mag
    else:
        flow = np.array(cv2.imread(flow_path,cv2.IMREAD_COLOR))   
        flow = cv2.cvtColor(flow,cv2.COLOR_RGB2GRAY)
        flow = np.array(flow)

    #for sparse gt:
    #gt   = depth_read(gt_path)
    #for dense gt:
    gt   = np.load(gt_path)

    gt_height, gt_width = img.shape[0], img.shape[1]

    #eigen crop
    img_crop  = np.array(img[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)])
    if (modality == 'rgb_flow'):
        flow_crop = np.array(flow[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)])
    else:
        flow_crop = flow
    
    #for sparse gt:
    #gt_crop   = np.array(gt[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)])

    #assert dimensions to be divisible by 32
    img_crop  = img_crop[-192:,:1120]
    flow_crop = flow_crop[-192:,:1120]
    gt_crop   = gt[-192:,:1120]

    return img_crop, flow_crop, gt_crop


class KITTI_Eigen_Dataloader(data.Dataset):
    modality_names = ['rgb_flow','rgb_flownet' ,'yuv_flow', 'rgb_flow_edges', 'yuv_flow_edges', 'rgb', 'flow','flownet', 'flow_edges']

    def __init__(self, root, type, modality='rgb_flow', loader=load_kitti_eigen):
        items = make_dataset(type,modality)
        #print(items)
        assert len(items)>0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(items), type))
        
        self.root = root
        self.items = items
   
        if type == 'train':
            self.transform = self.train_transform
        elif type == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))
        self.loader = loader
       
        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality
        self.pretrained = False
        self.output_size = (192, 1120)

    
    def train_transform(self,img,flow,gt):
        
        do_horizontal_flip = np.random.uniform(0.0, 1.0) > 0.4
        do_img_augment     = True
        do_random_crops    = False
        do_random_rotation = np.random.uniform(0.0, 1.0) > 0.4

        img_np  = np.array(img)
        flow_np = np.array(flow)
        gt_np   = np.array(gt) 
        edges_np = np.zeros(flow.shape)

        if (do_random_crops):
            img_np, flow_np, gt_np = random_crop(img_np,flow_np,gt_np,192,704)

        if(self.modality == 'rgb_flow_edges'):
            edges_np = get_canny_edge(img_np)

        if(do_horizontal_flip):
            img_np, flow_np, gt_np, edges_np = flip_h(img_np,flow_np,gt_np,edges_np)
        
        if(do_img_augment):
            aug_ = image_augment(p=0.85)
            data = {"image":img_np}
            img_ = aug_(**data)
            img_np = img_['image']  

        if(do_random_rotation):
            img_np, flow_np, gt_np, edges_np = random_rotation(img_np,flow_np,gt_np,edges_np)     

        #normalize image
        img_np = np.asfarray(img_np, dtype='float') / 255

        if (self.modality == 'rgb_flownet' or self.modality == 'flownet'):
            #flow_np = cv2.normalize(flow_np,None,0,1,cv2.NORM_MINMAX)
            flow_np = (flow_np - 4.1362284e-5)/(220 - 4.1362284e-5)
        else:
            flow_np  = np.asfarray(flow,dtype='float') / 255
        
        edges_np = np.array(edges_np, dtype='int') / 255

        return img_np, flow_np, gt_np, edges_np

    def val_transform(self,img,flow,gt):

        edges_np = np.zeros(flow.shape)
        do_center_crop = False

        if(do_center_crop):
            img  = crop_center(img,192,704)
            flow = crop_center(flow,192,704)
            gt   = crop_center(gt,192,704)

        if(self.modality == 'rgb_flow_edges'):
            edges_np = get_canny_edge(img)
        
        edges_np = np.array(edges_np, dtype='int') / 255
        img_np   = np.asfarray(img, dtype='float') / 255
        if (self.modality == 'rgb_flownet'):
            #flow_np = cv2.normalize(flow,None,0,1,cv2.NORM_MINMAX)
            flow_np = (flow - 4.1362284e-5)/(220 - 4.1362284e-5)
        else:
            flow_np  = np.asfarray(flow,dtype='float') / 255
        
       
        return img_np, flow_np, gt, edges_np
      
    def add_channel(self, img, flow):
        new_input = np.append(img, np.expand_dims(flow, axis=2), axis=2)
        return new_input   

    def __getraw__(self, index):     
        item = self.items[index]
        #print("i =",index, "path: ",item[0])
        img, flow, gt = self.loader(item,self.modality)
        return img, flow, gt

    def __getitem__(self, index):
       	img, flow, gt = self.__getraw__(index)
        
        img_np, flow_np, gt_np, edges_np = self.transform(img,flow,gt)
        

        if self.modality == 'rgb_flow' or self.modality == 'rgb_flownet' or self.modality == 'yuv_flow':
            input_np = self.add_channel(img_np,flow_np)
        elif self.modality == 'rgb_flow_edges':
            input_np = self.add_channel(img_np,flow_np)
            input_np = self.add_channel(input_np,edges_np)
        elif self.modality == 'rgb':
            input_np = img_np
        elif self.modality == 'flownet':
            input_np = flow_np
  
        input_tensor = to_tensor(input_np)

        if (self.modality == 'rgb') and (self.pretrained):
            preproc = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            input_tensor = preproc(input_tensor)

        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        gt_tensor = to_tensor(gt_np)
        gt_tensor = gt_tensor.unsqueeze(0)          

        return input_tensor, gt_tensor

    def __len__(self):
        return len(self.items)

