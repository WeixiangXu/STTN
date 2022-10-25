# STTN
Paper: [Soft Threshold Ternary Networks](https://arxiv.org/abs/2204.01234)
## Install Ternary Activation CUDA Extension
  - This repo uses Pytorch 1.7.1+cu101
  - Our CUDA extension are build with gcc version 5.3.1
  ```
  cd TerActivateFunc
  sh install.sh
  ```

## Training Time: Train STTN (ternarized W and A) on ImageNet

```
python main.py --data path_to_ImageNet --gpu-id 0,1,2,3 -j16 --arch resnet18_bwx2 --save=res18_bwx2_ta
```
After training, two parallel binary weights are obtained.

## Inference Time: Merge and Evaluate
Previous ternary networks introduces a hard threshold <img src="https://latex.codecogs.com/svg.image?\Delta" title="https://latex.codecogs.com/svg.image?\Delta" />  to divide quantization intervals, which sets an additional constraint and limits the ability of ternary networks:

<div align=center><img src="https://latex.codecogs.com/svg.image?\Delta&space;&space;&space;&space;T_i=&space;&space;&space;&space;\begin{cases}&space;&space;&space;&space;&plus;1,&&space;\text{if&space;$W_i>\Delta$}\\&space;&space;&space;&space;0,&&space;\text{if&space;$\left|W_i\right|\le&space;\Delta$}\\&space;&space;&space;&space;-1,&&space;\text{if&space;$W_i<-\Delta$}&space;&space;&space;&space;\end{cases}" title="https://latex.codecogs.com/svg.image?\Delta T_i= \begin{cases} +1,& \text{if $W_i>\Delta$}\\ 0,& \text{if $\left|W_i\right|\le \Delta$}\\ -1,& \text{if $W_i<-\Delta$} \end{cases}" /></div>

STTN enables the model to automatically determine which weights to be -1/0/1, avoiding the hard threshold  <img src="https://latex.codecogs.com/svg.image?\Delta" title="https://latex.codecogs.com/svg.image?\Delta" />. Concretely, at training time, we replace the ternary convolution filter <img src="https://latex.codecogs.com/svg.image?T" title="https://latex.codecogs.com/svg.image?T" /> with two parallel binary convolution filters <img src="https://latex.codecogs.com/svg.image?B_1" title="https://latex.codecogs.com/svg.image?B_1" /> and <img src="https://latex.codecogs.com/svg.image?B_1" title="https://latex.codecogs.com/svg.image?B_2" />. They are both binary-valued and have the same shape with ternary filter. Due to the additivity of convolutions with the same kernel sizes, a new kernel can be obtained by: <img src="https://latex.codecogs.com/svg.image?T&space;=&space;B_1&space;&plus;&space;B_2" title="https://latex.codecogs.com/svg.image?T = B_1 + B_2" />. During inference, the ternary weights are used.

- We provide a simple script to merge the two parallel binary weights into an equivalent ternary weight.
```
python merge_from_bwx2_to_ternary.py
```
- You can test the accuracy with the merged ternary weights. It should be the same as the result of two parallel binary weights during training, because they are equivalent.
```
python evaluate.py --data path_to_ImageNet -j16 --arch resnet18_ter --resume ./ternary/resnet18_ter.pth.tar --evaluate
```

## Results on ResNet-18

|  Model              | Weight   | Activation | Accuracy  | Download   |
| :------:            | :------: | :--------: |:-------:  | :------: |
|  Float              | FP32     | FP32       | 69.3      |          |
|  STTN (paper report)| ternary    | ternary  | 66.2%     |          |
|  STTN (this repo)   | ternary    | ternary  | **68.2%** |[google cloud](https://drive.google.com/file/d/1d_O46kxc5Fq8zl5XI2vVM_mJNTOhNATL/view?usp=sharing) |


## Citation
```
@inproceedings{xu2021soft,
  title={Soft threshold ternary networks},
  author={Xu, Weixiang and He, Xiangyu and Zhao, Tianli and Hu, Qinghao and Wang, Peisong and Cheng, Jian},
  booktitle={Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence},
  pages={2298--2304},
  year={2021}
}
```

```
@ARTICLE{9927185,
  author={Xu, Weixiang and Li, Fanrong and Jiang, Yingying and Yong, A and He, Xiangyu and Wang, Peisong and Cheng, Jian},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Improving Extreme Low-bit Quantization with Soft Threshold}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3216389}}
```
