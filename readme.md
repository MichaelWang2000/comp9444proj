# Timing information
* Our model is trained on two NVIDIA GeForce RTX 4090 GPUs for approximately 10h.
* Ubuntu (22s.04)
* Nvidia-smi version: 535.154.05
* Driver Version: 535.154.05
* Cuda version: 12.2.91
# How to set up the env
## install python 3.10
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10
sudo apt install python3.10-venv
```
## Using virtual env
```
pip install virtualenv
virtualenv -p python3.10 myenv
source comp9444_2/bin/activate
```
## install all package
```
pip install -r requirments.txt
```
## install cuda
```
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh ./cuda_12.2.0_535.54.03_linux.run --toolkit --silent --override
```
# Open the vitural env
```
source comp9444_2/bin/activate
```
# How to download the train data
## Style biedermeir:
```
gdown --folder https://drive.google.com/drive/folders/1kxXM1yA-vuFwiZR_T9n0g5KyCcmpwXxl
```
## Style ink_and_wash_painting: 
```
gdown --folder https://drive.google.com/drive/folders/1PSo_w7Y7V5HW5yj8TwLYZb74uilcL0Rh
```
## Style impressionism: 
```
gdown --folder https://drive.google.com/drive/folders/15r_gAL9foFcNuzqiZMKtdU2-miRDnCRd
```
## Style Contemporary Realism: 
```
gdown --folder https://drive.google.com/drive/folders/1LcxYkAqIGY93NTO8-cM--y3ugEwn4moU
```
## Train data: 
```
gdown https://drive.google.com/uc?id=1-v2UX6NF6a7A2xX9P4Azp5B1r3LQ4sQP/view?usp=sharing
```
# How to test
```
python test.py  --content_dir input/content/ --style_dir input/style/    --output out
```
# How to train the model
```
python train.py --style_dir ./datasets/Images/ --content_dir ./datasets/train2014 --save_dir models/ --batch_size 4
```

# StyTr^2 : Image Style Transfer with Transformers（CVPR2022）
*Authors: [Yingying Deng](https://diyiiyiii.github.io/), Fan Tang, XingjiaPan, Weiming Dong, Chongyang Ma, Changsheng Xu*

This paper is proposed to achieve unbiased image style transfer based on the transformer model. We can promote the stylization effect compared with state-of-the-art methods.
This repository is the official implementation of [SyTr^2 : Image Style Transfer with Transformers](https://arxiv.org/abs/2105.14576).

## Results presentation 
<p align="center">
<img src="https://github.com/diyiiyiii/StyTR-2/blob/main/Figure/Unbiased.png" width="90%" height="90%">
</p>
Compared with some state-of-the-art algorithms, our method has a strong ability to avoid content leakage and has better feature representation ability.  <br>


## Framework
<p align="center">
<img src="https://github.com/diyiiyiii/StyTR-2/blob/main/Figure/network.png" width="100%" height="100%">
</p> 
The overall pipeline of our StyTr^2 framework. We split the content and style images into patches, and use a linear projection to obtain image sequences. Then the content sequences added with CAPE are fed into the content transformer encoder, while the style sequences are fed into the style transformer encoder. Following the two transformer encoders, a multi-layer transformer decoder is adopted to stylize the content sequences according to the style sequences. Finally, we use a progressive upsampling decoder to obtain the stylized images with high-resolution.



## Experiment
### Requirements
* python 3.6
* pytorch 1.4.0
* PIL, numpy, scipy
* tqdm  <br> 

---

Another possible setup was tested using Python 3.7:

      certifi==2024.2.2
      charset-normalizer==3.3.2
      cycler==0.11.0
      fonttools==4.38.0
      future==1.0.0
      idna==3.7
      kiwisolver==1.4.5
      matplotlib==3.5.3
      numpy==1.21.6
      nvidia-cublas-cu11==11.10.3.66
      nvidia-cuda-nvrtc-cu11==11.7.99
      nvidia-cuda-runtime-cu11==11.7.99
      nvidia-cudnn-cu11==8.5.0.96
      packaging==24.0
      Pillow==9.5.0
      pyparsing==3.1.2
      python-dateutil==2.9.0.post0
      requests==2.31.0
      scipy==1.7.3
      six==1.16.0
      torch==1.6.0
      torchvision==0.7.0
      typing_extensions==4.7.1
      urllib3==2.0.7

> **NOTE:** On newer Ubuntu (24.04) it might be easier to install Python 3.7 because it can be added and installedfrom deadsnakes PPA using the following commands:
>```
>sudo add-apt-repository ppa:deadsnakes/ppa
>sudo apt update
>sudo apt install python3.7
>sudo apt install python3.7-venv
>```
### Testing 
Pretrained models: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  [vit_embedding](https://drive.google.com/file/d/1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B/view?usp=sharing), [decoder](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing), [Transformer_module](https://drive.google.com/file/d/1dnobsaLeE889T_LncCkAA2RkqzwsfHYy/view?usp=sharing)   <br> 
Please download them and put them into the floder  ./experiments/  <br> 
```
python test.py  --content_dir input/content/ --style_dir input/style/    --output out
```
### Training  
Style dataset is WikiArt collected from [WIKIART](https://www.wikiart.org/)  <br>  
content dataset is COCO2014  <br>  
```
python train.py --style_dir ../../datasets/Images/ --content_dir ../../datasets/train2014 --save_dir models/ --batch_size 8
```
### Reference
If you find our work useful in your research, please cite our paper using the following BibTeX entry ~ Thank you ^ . ^. Paper Link [pdf](https://arxiv.org/abs/2105.14576)<br> 
```
@inproceedings{deng2021stytr2,
      title={StyTr^2: Image Style Transfer with Transformers}, 
      author={Yingying Deng and Fan Tang and Weiming Dong and Chongyang Ma and Xingjia Pan and Lei Wang and Changsheng Xu},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022},
}
```
