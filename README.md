# RFB_ESRGAN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Perceptual Extreme Super Resolution Network with Receptive Field Block](https://arxiv.org/abs/2005.12597v1).

## Table of contents

- [RFB_ESRGAN-PyTorch](#rfb_esrgan-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
        - [Download train dataset](#download-train-dataset)
        - [Download valid dataset](#download-valid-dataset)
    - [Test](#test)
    - [Train](#train)
        - [Train RRDBNet model](#train-rrdbnet-model)
        - [Train ESRGAN model](#train-esrgan-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Perceptual Extreme Super Resolution Network with Receptive Field Block](#perceptual-extreme-super-resolution-network-with-receptive-field-block)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/1PbvPyhbhTUXmuMn7eRwDGupOA-qoyjbp?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1UbOacCc6i8nbxEmLzkFtjQ?pwd=llot) access:`llot`

## Download datasets

### Download train dataset

#### DFO2K

- Image format
    - [Baidu Driver](https://pan.baidu.com/s/1RwQ_x6_CEJpfmh9IT8EJIA?pwd=llot) access: `llot`

### Download valid dataset

#### Set5

- Image format
    - [Google Driver](https://drive.google.com/file/d/1GtQuoEN78q3AIP8vkh-17X90thYp_FfU/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1dlPcpwRPUBOnxlfW5--S5g) access:`llot`

#### Set14

- Image format
    - [Google Driver](https://drive.google.com/file/d/1CzwwAtLSW9sog3acXj8s7Hg3S7kr2HiZ/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1KBS38UAjM7bJ_e6a54eHaA) access:`llot`

#### BSD200

- Image format
    - [Google Driver](https://drive.google.com/file/d/1cdMYTPr77RdOgyAvJPMQqaJHWrD5ma5n/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1xahPw4dNNc3XspMMOuw1Bw) access:`llot`

## Test

Modify the contents of the file as follows.

- line 28: `upscale_factor` change to the magnification you need to enlarge.
- line 30: `mode` change Set to valid mode.
- line 119: `model_path` change weight address after training.

## Train

Modify the contents of the file as follows.

- line 28: `upscale_factor` change to the magnification you need to enlarge.
- line 30: `mode` change Set to train mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.

### Train RRDBNet model

- line 47: `resume` change to `True`.
- line 48: `strict` Transfer learning is set to `False`, incremental learning is set to `True`.
- line 49: `start_epoch` change number of training iterations in the previous round.
- line 50: `resume_weight` the weight address that needs to be loaded.

### Train ESRGAN model

- line 79: `resume` change to `True`.
- line 80: `strict` Transfer learning is set to `False`, incremental learning is set to `True`.
- line 81: `start_epoch` change number of training iterations in the previous round.
- line 82: `resume_d_weight` the discriminator weight address that needs to be loaded.
- line 83: `resume_g_weight` the generator weight address that needs to be loaded.

### Result

Source of original paper results: [https://arxiv.org/pdf/2005.12597v1.pdf](https://arxiv.org/pdf/2005.12597v1.pdf)

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale | RRDBNet (PSNR) | ESRGAN (PSNR) |
|:-------:|:-----:|:--------------:|:-------------:|
|  Set5   |  16   |    -(**-**)    | -(**29.45**)  |
|  Set14  |  16   |    -(**-**)    | -(**25.88**)  |

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
