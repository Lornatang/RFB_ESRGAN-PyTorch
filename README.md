# RFB_ESRGAN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation of [Perceptual Extreme Super Resolution Network with Receptive Field Block](https://arxiv.org/abs/2005.12597v1).

## Table of contents

- [RFB_ESRGAN-PyTorch](#rfb_esrgan-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [Test](#test)
    - [Train](#train)
        - [Train RFBESRNet model](#train-rfbesrnet-model)
        - [Train RFBESRGAN model](#train-rfbesrgan-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Perceptual Extreme Super Resolution Network with Receptive Field Block](#perceptual-extreme-super-resolution-network-with-receptive-field-block)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

## Test

Modify the contents of the `config.py` file as follows.

- line 30: `upscale_factor` change to the magnification you need to enlarge.
- line 32: `mode` change Set to valid mode.
- line 110: `model_path` change weight address after training.

## Train

Modify the contents of the `config.py` file as follows.

- line 30: `upscale_factor` change to the magnification you need to enlarge.
- line 32: `mode` change Set to train mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.

### Train RFBESRNet model

- line 48: `start_epoch` change number of RFB_ESRNet training iterations in the previous round.
- line 49: `resume` change to RFB_ESRNet weight address that needs to be loaded.

### Train RFBESRGAN model

- line 76: `start_epoch` change number of RFB_ESRGAN training iterations in the previous round.
- line 77: `resume` change to RFB_ESRNet weight address that needs to be loaded.
- line 78: `resume_d` change to Discriminator weight address that needs to be loaded.
- line 79: `resume_g` change to Generator weight address that needs to be loaded.

### Result

Source of original paper results: [https://arxiv.org/pdf/2005.12597v1.pdf](https://arxiv.org/pdf/2005.12597v1.pdf)

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale | RFBNet (PSNR) | RFB_ESRGAN (PSNR) |
|:-------:|:-----:|:-------------:|:-----------------:|
|  DIV8K  |  16   |    (**-**)    |   23.38(**-**)    |

Low resolution / Recovered High Resolution / Ground Truth
<span align="center"><img src="assets/result.png"/></span>

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Perceptual Extreme Super Resolution Network with Receptive Field Block

_Taizhang Shang, Qiuju Dai, Shengchen Zhu, Tong Yang, Yandong Guo_ <br>

**Abstract** <br>
Perceptual Extreme Super-Resolution for single image is extremely difficult, because the texture details of different images vary greatly. To tackle
this difficulty, we develop a super resolution network with receptive field block based on Enhanced SRGAN. We call our network RFB-ESRGAN. The key
contributions are listed as follows. First, for the purpose of extracting multi-scale information and enhance the feature discriminability, we applied
receptive field block (RFB) to super resolution. RFB has achieved competitive results in object detection and classification. Second, instead of using
large convolution kernels in multi-scale receptive field block, several small kernels are used in RFB, which makes us be able to extract detailed
features and reduce the computation complexity. Third, we alternately use different upsampling methods in the upsampling stage to reduce the high
computation complexity and still remain satisfactory performance. Fourth, we use the ensemble of 10 models of different iteration to improve the
robustness of model and reduce the noise introduced by each individual model. Our experimental results show the superior performance of RFB-ESRGAN.
According to the preliminary results of NTIRE 2020 Perceptual Extreme Super-Resolution Challenge, our solution ranks first among all the participants.

[[Paper]](https://arxiv.org/pdf/2005.12597v1.pdf)

```bibtex
@misc{2005.12597,
    Author = {Taizhang Shang and Qiuju Dai and Shengchen Zhu and Tong Yang and Yandong Guo},
    Title = {Perceptual Extreme Super Resolution Network with Receptive Field Block},
    Year = {2020},
    Eprint = {arXiv:2005.12597},
}
```
