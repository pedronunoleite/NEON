# Exploiting Motion Perception in Depth Estimation through a Lightweight Convolutional Neural Network
<p align="justify">
This repository contains the code basis for the NEON network proposed in "Exploiting Motion Perception in Depth Estimation through a Lightweight Convolutional Neural Network", as well as the instructions on how to reproduce its results.

If you use this repository, please cite:
<\p>
```
//TODO
@article{
  author = {}
  doi = {},
  journal = {},
  month = {},
  number = {},
  pages = {},
  title = {},
  url = {}
  volume = {},
  year = {}
}
```
<p align="justify">
This article introduces a lightweight Convolutional Neural Network (CNN) for depth estimation, NEON, designed for balancing both accuracy and inference times.
Instead of solely focusing on visual features, the proposed methodology exploits the Motion-Parallax effect to combine the apparent motion of pixels with texture. 
This research demonstrates that motion perception provides crucial insight about the magnitude of movement for each pixel, which also encodes cues about depth since large displacements usually occur when objects are closer to the imaging sensor. 

NEON’s performance is compared to relevant networks in terms of Root Mean Squared Error (RMSE), the percentage of correctly predicted pixels (δ1) and inference times, using the KITTI dataset. Experiments prove that NEON is significantly more efficient than the current top ranked network, estimating predictions 12 times faster; while achieving an average RMSE of 3.118 m and a δ1 of 94.5%. Ablation studies demonstrate the relevance of tailoring the network to use motion perception principles in estimating depth from image
sequences, considering that the effectiveness and quality of the estimated depth map is similar to more computational demanding state-of-the-art networks. 
<\p>

![alt text](https://github.com/pedronunoleite/NEON/blob/main/media/visual_abstract.png)
