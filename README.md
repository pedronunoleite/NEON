# Exploiting Motion Perception in Depth Estimation through a Lightweight Convolutional Neural Network
<p align="justify">
This repository contains the code basis for the NEON network proposed in "Exploiting Motion Perception in Depth Estimation through a Lightweight Convolutional Neural Network", as well as the instructions on how to reproduce its results.
  
<p align="justify">
This article introduces a lightweight Convolutional Neural Network (CNN) for depth estimation, NEON, designed for balancing both accuracy and inference times.
Instead of solely focusing on visual features, the proposed methodology exploits the Motion-Parallax effect to combine the apparent motion of pixels with texture. 
This research demonstrates that motion perception provides crucial insight about the magnitude of movement for each pixel, which also encodes cues about depth since large displacements usually occur when objects are closer to the imaging sensor. 
</p>

If you use this repository, please cite:
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
## Table of Contents

[***Contributions***](https://github.com/pedronunoleite/NEON#Contributions)

[***Architecture***](https://github.com/pedronunoleite/NEON#Architecture)

[***Results***](https://github.com/pedronunoleite/NEON#Results)

[***Implementation***](https://github.com/pedronunoleite/NEON#Implementation)

[***References***](https://github.com/pedronunoleite/NEON#References)


## Contributions

<p align="justify">
The ability to comprehend visual changes in a sequence of images can be referred to as motion perception. It plays a crucial part in autonomous systems, providing relevant
information about the speed and direction of any moving object in a scene [7]. Additionally, information that directly translates motion can also be valuable for 3D perception.
</p>

<p align="justify">
This article proposes a lightweight encoder-decoder architecture, NEON (deNse dEpth from mOtioN-parallax), that exploits the notion of Motion-Parallax - objects closer to an observer are perceived to move faster than objects further away [8] - by making use of the magnitude of pixel motion in a flow field. This information is combined with texture-based features from an RGB image, to obtain a dense and accurate depth map of the scene. A visual representation of the proposed pipeline is presented in Fig.1.
</p>

![alt text](https://github.com/pedronunoleite/NEON/blob/main/media/visual_abstract.png)

<p align="justify">
By introducing the magnitude of motion as part of the network’s input, the learning task shifts focus from solely learning how to predict depth from texture, shadows and
occlusions; to also weighing in which magnitude of motion corresponds to what depth value. This allows the network’s complexity to be reduced, being capable of estimating depth
maps with low-latency, while maintaining high levels of accuracy.
</p>

In summary, the contributions of this work include:
<ul>
  <li><p align="justify"> 
A lightweight Convolutional Neural Network (CNN), designed to balance performance and inference times. Near State-of-the-Art results are achieved - the average absolute difference in terms of Root Mean Squared Error is 36 cm. NEON takes a fraction of the time to estimate predictions, being 12 times faster than the currently top ranked network. </p></li>
  <li><p align="justify"> 
     A thorough study on the impact of the introduction of the motion perception features as an additional input to the network. Ablation studies indicate a significant performance gain when the Motion-Parallax effect is exploited; and reveal that the network gives more weight to the newly introduced information than to texture-based features.
    </p></li>
  <li><p align="justify"> 
    An extensive benchmark analysis of the results obtained from evalutating NEON’s performance on Eigen’s [9] split of the KITTI dataset. Being composed of real world data, this dataset allows NEON to be tested in a multitude of scenarios, under various lighting conditions and phenomena. Finally, NEON is compared to the most relevant networks to date
   </p></li></ul>

## Architecture
<p align="justify"> 
The proposed network is based on an encoder-decoder structure. The encoder is responsible for extracting high-level characteristics from the input. These become a set of low-resolution feature maps that need to be merged and upsampled, so that the network is able to output a dense and full resolution depth map.  
</p>

<p align="justify">  
The encoder is composed of a ResNet18 followed by a 1x1 convolutional layer that asserts the number of channels to the decoder, which is comprised of five upsampling layers, each followed by a bilinear interpolation with a scale factor of two. Skip additive connections are implemented between both. Feature maps are presented as Channels x Height x Width.

![alt text](https://github.com/pedronunoleite/NEON/blob/main/media/neon_arch.png)


## Results

<p align="justify"> 
NEON’s performance is compared to relevant networks in terms of Root Mean Squared Error (RMSE), the percentage of correctly predicted pixels (δ1) and inference times, using the KITTI dataset. Experiments prove that NEON is significantly more efficient than the current top ranked network, estimating predictions 12 times faster; while achieving an average RMSE of 3.118 m and a δ1 of 94.5%. 
</p>
<p align="justify"> 
Ablation studies demonstrate the relevance of tailoring the network to use motion perception principles in estimating depth from image sequences, considering that the effectiveness and quality of the estimated depth map is similar to more computational demanding state-of-the-art networks.
</p>

<p align="justify"> 
For a thorough discussion of the experiments and results obtained with the NEON network, please refer to our article.
</p>

![alt text](https://github.com/pedronunoleite/NEON/blob/main/media/sota_comp.png)
![alt text](https://github.com/pedronunoleite/NEON/blob/main/media/sota_comp_img.png)




