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

[***Instructions***](https://github.com/pedronunoleite/NEON#Instructions)

[***References***](https://github.com/pedronunoleite/NEON#References)

[***Contact***](https://github.com/pedronunoleite/NEON#Contact)

## Contributions

<p align="justify">
The ability to comprehend visual changes in a sequence of images can be referred to as motion perception. It plays a crucial part in autonomous systems, providing relevant
information about the speed and direction of any moving object in a scene [1]. Additionally, information that directly translates motion can also be valuable for 3D perception.
</p>

<p align="justify">
This article proposes a lightweight encoder-decoder architecture, NEON (deNse dEpth from mOtioN-parallax), that exploits the notion of Motion-Parallax - objects closer to an observer are perceived to move faster than objects further away [2] - by making use of the magnitude of pixel motion in a flow field. This information is combined with texture-based features from an RGB image, to obtain a dense and accurate depth map of the scene. A visual representation of the proposed pipeline is presented in Fig.1.
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
    An extensive benchmark analysis of the results obtained from evalutating NEON’s performance on Eigen’s [3] split of the KITTI dataset. Being composed of real world data, this dataset allows NEON to be tested in a multitude of scenarios, under various lighting conditions and phenomena. Finally, NEON is compared to the most relevant networks to date.
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

![alt text](https://github.com/pedronunoleite/NEON/blob/main/media/sota_comp.PNG)


## Implementation

<p align="justify"> 
The code available in this repository has been developed with:
</p>

<ul>
  <li><p align="justify"> CUDA 11.0 </p></li>
  <li><p align="justify"> Pytorch 1.6.0 </p></li>
  <li><p align="justify"> Python 3.7.9 </p></li>
</ul>
<p align="justify"> 
All the experiments discussed within the article were run with an NVIDIA GeForce RTX 2080 SUPER with 8GB of VRAM, and the Intel i5-8600K CPU @ 3.60GHz with 6 cores.
</p>

<p align="justify"> 
We use Eigen's [3] split of the KITTI Dataset. The ground-truth is infilled by interpolating the missing depth values according to the colorization algorithm proposed by Levin et al. [4]. The optical flows are estimated with the Pytorch implementation of the Flownet2.0 [5] that can be found in: https://github.com/NVIDIA/flownet2-pytorch.
</p>

## Instructions
<ul>
  <li><p align="justify"> Reproduce the results presented in the paper: </p></li></ul>
  
```
python3 main.py --evaluate [path_to_model]
```
The path to our model can be found in: https://drive.google.com/file/d/1eiL3qJ3g8AhebnBj9N3DL4Vr5rV-pKXG/view?usp=sharing.
This should be the output:

```
*
RMSE=3.118
MAE=1.324
Delta1=0.945
REL=0.064
Lg10=0.027
iRMSE=10.397
iMAE=0.053
SIlog=4.8857
t_GPU=0.0064
```
Note: The t_GPU metric obviously depends on the specs of computer running the experiment.

<ul>
  <li><p align="justify"> To retrain our model: </p></li></ul>
    
```
python3 main.py --resume [path_to_model] --epochs nr_epochs -c loss_function -b batch_size -lr learning_rate 
```

<ul>
  <li><p align="justify"> To train from scratch: </p></li></ul>
    
```
python3 main.py --arch encoder_name --decoder decoder_name--epochs nr_epochs -c loss_function -b batch_size -lr learning_rate --modality modality_name
```
--arch options:

['squeezenet','shufflenetv2','vgg11','densenet121','densenet121_skipadd','resnet18', 'resnet50', 'resnet18skipadd', 'resnet18skipadd_dw', 'resnet18skipconcat','mobilenet', 'mobilenetskipadd', 'mobilenetskipconcat']

--decoder options:

['deconv3', 'deconv3dw', 'deconv5', 'deconv5dw', 'deconv7', 'deconv7dw', 'deconv9', 'deconv9dw', 'upconv', 'upproj', 'nnconv3', 'blconv3', 'shuffle3', 'nnconv3dw', 'blconv3dw', 'shuffle3dw', 'nnconv5', 'blconv5', 'shuffle5', 'nnconv5dw', 'blconv5dw', 'shuffle5dw', 'nnconv7', 'blconv7', 'shuffle7', 'nnconv7dw', 'blconv7dw', 'shuffle7dw', 'nnconv9', 'blconv9', 'shuffle9', 'nnconv9dw', 'blconv9dw', 'shuffle9dw']

--modality options:

['rgb_flow', 'rgb_flownet', 'yuv_flow', 'rgb_flow_edges', 'yuv_flow_edges', 'rgb', 'flow', 'flownet', 'flow_edges']

--loss options:

['l1', 'l2', 'custom', 'smoothl1', 'inversedepthsmoothness'] 


## References
<p align="justify">
[1] A. M. Pinto, A. P. Moreira, M. V. Correia, and P. G. Costa, “A flow-based motion perception technique for an autonomous robot system,” J. Intell. Robot. Syst. Theory Appl., vol. 75, no. 3-4, pp. 475–492, 2014.

[2] A. M. Pinto, “Visual motion analysis based on a robotic moving system,” Ph.D. dissertation, Faculty of Engineering of the University of Porto, FEUP, 2014. [Online]. Available: https://hdl.handle.net/10216/73552.

[3] D. Eigen, C. Puhrsch, and R. Fergus, “Depth map prediction from a single image using a multi-scale deep network,” Adv. Neural Inf. Process. Syst., vol. 3, no. January, pp. 2366–2374, 2014.

[4] A. Levin, D. Lischinski, and Y. Weiss, “Colorization using optimization,” ACM SIGGRAPH 2004 Pap. SIGGRAPH 2004, pp. 689–694, 2004.

[5] E. Ilg, N. Mayer, T. Saikia, M. Keuper, A. Dosovitskiy, and T. Brox, “FlowNet 2.0: Evolution of optical flow estimation with deep networks,” Proc. - 30th IEEE Conf. Comput. Vis. Pattern Recognition, CVPR 2017, vol. 2017-January, pp. 1647–1655, 2017.
</p>

The code basis for this repository has been adapted from: https://github.com/dwofk/fast-depth.

The figure depicting NEON's architecture has been generated with: https://github.com/HarisIqbal88/PlotNeuralNet.

The number of MAC and floating point operations have been estimated with: https://github.com/sovrasov/flops-counter.pytorch.

Data augmentation was conducted with: https://github.com/albumentations-team/albumentations.

## Contact
If you have any inquiry about this work please email me at: pedro.nuno@inesctec.pt
