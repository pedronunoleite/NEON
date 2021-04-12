import numpy as np 
import cv2
import torch
import kornia
import dataloaders.transforms as transforms_
import time
import open3d as o3d

to_tensor = transforms_.ToTensor()

def add_channel(img, flow):
    new_input = np.append(img, np.expand_dims(flow, axis=2), axis=2)
    return new_input   

imgs = '/media/cras4/3FD800AF4F722DA9/KITTI_eigen_split/imgs/'
#gt   = '/media/cras4/3FD800AF4F722DA9/KITTI_eigen_split/gt/'
stereo_flow = '/media/cras4/3FD800AF4F722DA9/KITTI_eigen_split/stereo_flow/'
gt_dense = '/media/cras4/3FD800AF4F722DA9/KITTI_eigen_split/gt_dense/'

#read files
f = open('dataloaders/eigen_test_split.txt', 'r')
x = f.readlines()
vec = x[0].split(' ')


#read images
img  = cv2.imread(imgs + vec[0], cv2.IMREAD_COLOR)
flow = cv2.imread(stereo_flow + vec[0], cv2.IMREAD_GRAYSCALE)
gt   = np.load(gt_dense + vec[2][:-5] + '.npy')

gt_height, gt_width = flow.shape

#eigen crop
img_crop  = np.array(img[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)])
flow_crop = np.array(flow[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)])
#gt_crop   = np.array(gt[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)])

#assert dimensions to be divisible by 32
img_crop  = img_crop[-192:,:1120]
flow_crop = flow_crop[-192:,:1120]
gt_crop   = gt[-192:,:1120]

img_crop   = np.asfarray(img_crop, dtype='float') / 255
flow_crop  = np.asfarray(flow_crop,dtype='float') / 255



#Make 4 channel input
input_np = add_channel(img_crop, flow_crop)

#transform to tensor
input_tensor = to_tensor(input_np)

while input_tensor.dim() < 4:
    input_tensor = input_tensor.unsqueeze(0)

gt_tensor = to_tensor(gt_crop)
gt_tensor = gt_tensor.unsqueeze_(0)
gt_tensor = gt_tensor.unsqueeze_(0)


#load model
checkpoint = torch.load('rn18skip_bilinear_model_best.pth.tar')
model = checkpoint['model']

#infer depth map
model.eval()
with torch.no_grad():
    depth_map = model(input_tensor.cuda())
print(depth_map.data)

#Intrinsic matrix
K = [[ 9.597910e+02, 0.000000, 6.960217e+02],
     [0.000000, 9.569251e+02, 2.241806e+02],
     [0.000000e+00, 0.000000, 1.000000e+00]]

K = np.array(K).reshape(3,3)
K_tensor = to_tensor(K)
K_tensor = K_tensor.unsqueeze_(0)
#print(K_tensor)

#3D Projection:
end = time.time()

map_3d = kornia.depth_to_3d(depth_map,K_tensor.cuda())
map_3d = np.squeeze(map_3d.cpu().numpy())

print(map_3d[0][0][0])
print(map_3d[1][0][0])
print(map_3d[2][0][0])

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




