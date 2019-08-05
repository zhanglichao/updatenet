## Learning the Model Update for Siamese Trackers [[paper]](https://arxiv.org/pdf/1908.00855.pdf)

## Instructions
In general, in Siamese trackers the template is linearly combined with the accumulated template from the previous frame, resulting in an exponential decay of information over time. While such an approach to updating has led to improved results, its simplicity limits the potential gain likely to be obtained by learning to update. Therefore, in this paper we propose to replace the handcrafted update function with a method which learns to update. We use a convolutional neural network, called UpdateNet, which given the initial template, the accumulated template and the template of the current frame aims to estimate the optimal template for the next frame. The UpdateNet is compact and can easily be integrated into existing Siamese trackers, e.g. SiamFC [1] and DaSiamRPN [2].


## Visualization of learned accumulated templates and response maps.
<br>
<p align="center">
  <img width="80%" height='80%'src="figs/fig3_reb.png" />
</p>


## Pre-trained models

The pre-trained [models](https://github.com/zhanglichao/updatenet/tree/master/models) are available to download.

## Raw results

The [results](https://drive.google.com/drive/folders/1GhYh-R5u5paLH2Qw1MwSUW4M_kGg0Z82?usp=sharing) are available for comparison.

[1] Luca Bertinetto, Jack Valmadre, João F. Henriques, Andrea Vedaldi, Philip H. S. Torr.
    Fully-convolutional siamese networks for object tracking.
    In ECCV workshop 2016.

[2] Zheng Zhu, Qiang Wang, Bo Li, Wei Wu, Junjie Yan, Weiming Hu.
    Distractor-aware siamese networks for visual object tracking. 
    In ECCV 2018.
