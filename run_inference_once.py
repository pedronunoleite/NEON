import numpy as np 
import cv2
import torch
import kornia
import dataloaders.transforms as transforms_
import time
import open3d as o3d
import dataloaders.read_flow_lib as read_flow_lib_
import utils
from PIL import Image



to_tensor = transforms_.ToTensor()

def add_channel(img, flow):
    new_input = np.append(img, np.expand_dims(flow, axis=2), axis=2)
    return new_input   

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



input_img  = "/media/cras4/Server/Pedro/External_DIsk/KITTI_eigen_split/imgs/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000800.png"
input_flow = "/home/cras4/Pytorch_ws/Parallax-Depth/parallax-net/visual_abstract.flo"

#sparse_gt_path = "/media/cras4/Server/Pedro/External_DIsk/KITTI_eigen_split/gt/2011_09_26_drive_0029_sync/proj_depth/groundtruth/image_02/0000000126.png"
#sparse_gt = depth_read(sparse_gt_path)


img  = np.array(cv2.imread(input_img,cv2.IMREAD_COLOR))
img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img  = np.array(img)

flow = read_flow_lib_.read_flow(input_flow)
flowx = flow[:,:,0]
flowy = flow[:,:,1] 
#obtain the magnitude
flow_mag, flow_ang = cv2.cartToPolar(flowx,flowy)
flow = flow_mag

#crop (full size atm)
#img_crop = img[-320:,:1216]
#flow_crop = flow[-320:,:1216]

#padding for full img
#diff_H = 1280 - img.shape[1]
#diff_V = 384  - img.shape[0]
#img = cv2.copyMakeBorder(img, diff_V, 0, 0, diff_H, cv2.BORDER_CONSTANT)

gt_height, gt_width = img.shape[0], img.shape[1]
img = np.array(img[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)])
#flow = np.array(flow[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)])

#for sparse gt:
#gt_crop   = np.array(sparse_gt[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)])
#gt_crop   = gt_crop[-192:,:1120]


img  = img[-192:,:1120]
flow = flow[-192:,:1120]

print(img.shape)
print(flow.shape)

img_np   = np.asfarray(img, dtype='float') / 255
flow_np  = (flow - 4.1362284e-5)/(220 - 4.1362284e-5)

#Make 4 channel input
input_np = add_channel(img_np, flow_np)


#transform to tensor
input_tensor = to_tensor(input_np)

while input_tensor.dim() < 4:
    input_tensor = input_tensor.unsqueeze(0)

#load model
checkpoint = torch.load('model_best.pth.tar')
model = checkpoint['model']

#infer depth map
model.eval()
with torch.no_grad():
    depth_map = model(input_tensor.cuda())

#assert limits
depth_map.data[depth_map.data < 1e-5] = 1e-5
depth_map.data[depth_map.data > 80] = 80
#depth_map.data = depth_map.data[:,:,-370:,:1226]
print(depth_map.data.shape)

pred_ = np.squeeze(depth_map.data.cpu().numpy())
pred_ = pred_[-192:,:1120]
rgb   = input_tensor[:,:3,:,:]
rgb = np.transpose(np.squeeze(rgb.cpu().numpy()), (1,2,0))
rgb = rgb *255
rgb = rgb[-192:,:1120]
print(rgb.shape)
flow  = input_tensor[:,3:,:,:]
flow_cpu = np.squeeze(flow.cpu().numpy())
flow_cpu = cv2.normalize(flow_cpu,None,0,255,cv2.NORM_MINMAX)
flow_input_col = cv2.cvtColor(flow_cpu,cv2.COLOR_GRAY2RGB)
flow_cpu = flow_cpu[-192:,:1120]
filename = "/home/cras4/Pytorch_ws/Parallax-Depth/parallax-net/visual_abstract_flow.png"
utils.save_image(flow_cpu,filename)
print(flow_cpu.shape)

import matplotlib.pyplot as plt

depth_pred = utils.colored_depthmap(pred_, 0, 80)
filename   = "/home/cras4/Pytorch_ws/Parallax-Depth/parallax-net/visual_abstract_depth.png"
utils.save_image(depth_pred,filename)

       
plt.imshow(pred_, vmin=0, vmax=80, cmap='jet', aspect='equal')
plt.colorbar(orientation="horizontal", pad=0.005)
plt.axis('off')
filename  = "/home/cras4/Pytorch_ws/Parallax-Depth/parallax-net/full_size_test3.png"
#plt.savefig(filename, bbox_inches='tight', dpi=1200)   
plt.clf()

from PIL import Image
img_pil = Image.fromarray(rgb.astype('uint8'))
img_pil.show()

flow_pil = Image.fromarray(flow_cpu.astype('uint8'))
#flow_pil.show()

#to 3D

#Intrinsic matrix
K = [[9.569475e+02, 0.000000, 6.939767e+02],
     [0.000000, 9.522352e+02, 2.386081e+02],
     [0.000000e+00, 0.000000, 1.000000e+00]]
K = np.array(K).reshape(3,3)
K_tensor = to_tensor(K)
K_tensor = K_tensor.unsqueeze_(0)

#gt sparse
#sparse_tensor = to_tensor(sparse_gt)
#sparse_tensor = sparse_tensor.unsqueeze_(0)
#sparse_tensor = sparse_tensor.unsqueeze_(0)

#map_3d = kornia.depth_to_3d(sparse_tensor.cuda(),K_tensor.cuda())
#map_3d = np.squeeze(map_3d.cpu().numpy())



#3D Projection:
end = time.time()

map_3d = kornia.depth_to_3d(depth_map,K_tensor.cuda())
map_3d = np.squeeze(map_3d.cpu().numpy())

_,rows,cols = map_3d.shape
points_vec = []

for i in range(rows):
    for j in range(cols):
        point_3d = map_3d[0][i][j], map_3d[1][i][j], map_3d[2][i][j]
        points_vec.append(point_3d)


points_vec = np.array(points_vec)
print(points_vec[0])


torch.cuda.synchronize()
gpu_time = time.time() - end
print(gpu_time)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_vec)
o3d.visualization.draw_geometries([pcd])



