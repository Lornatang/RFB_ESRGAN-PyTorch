# RFB_ESRGAN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation of
[Perceptual Extreme Super Resolution Network with Receptive Field Block](https://arxiv.org/abs/2005.12597).

### Table of contents

- [RFB_ESRGAN-PyTorch](#rfb_esrgan-pytorch)
- [Overview](#overview)
- [Table of contents](#table-of-contents)
- [About Perceptual Extreme Super Resolution Network with Receptive Field Block](#about-perceptual-extreme-super-resolution-network-with-receptive-field-block)
- [Model Description](#model-description)
- [Installation](#installation)
    - [Clone and install requirements](#clone-and-install-requirements)
    - [Download dataset](#download-dataset)
- [Test](#test)
    - [Test benchmark](#test-benchmark)
    - [Test image](#test-image)
    - [Test video](#test-video)
    - [Test model performance](#test-model-performance)
- [Train (e.g DIV2K)](#train-eg-div2k)
- [Contributing](#contributing)
- [Credit](#credit)
    - [Perceptual Extreme Super Resolution Network with Receptive Field Block](#perceptual-extreme-super-resolution-network-with-receptive-field-block)

### About Perceptual Extreme Super Resolution Network with Receptive Field Block

If you're new to RFB-ESRGAN, here's an abstract straight from the paper:

Perceptual Extreme Super-Resolution for single image is extremely difficult, because the texture details of different
images vary greatly. To tackle this difficulty, we develop a super resolution network with receptive field block based
on Enhanced SRGAN. We call our network RFB-ESRGAN. The key contributions are listed as follows. First, for the purpose
of extracting multi-scale information and enhance the feature discriminability, we applied receptive field block (RFB)
to super resolution. RFB has achieved competitive results in object detection and classification. Second, instead of
using large convolution kernels in multi-scale receptive field block, several small kernels are used in RFB, which makes
us be able to extract detailed features and reduce the computation complexity. Third, we alternately use different
upsampling methods in the upsampling stage to reduce the high computation complexity and still remain satisfactory
performance. Fourth, we use the ensemble of 10 models of different iteration to improve the robustness of model and
reduce the noise introduced by each individual model. Our experimental results show the superior performance of
RFB-ESRGAN. According to the preliminary results of NTIRE 2020 Perceptual Extreme Super-Resolution Challenge, our
solution ranks first among all the participants.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives
a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that
discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that
x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/Lornatang/RFB_ESRGAN-PyTorch.git
$ cd RFB_ESRGAN-PyTorch/
$ pip3 install -r requirements.txt
```

#### Download dataset

```bash
$ cd data/
$ bash download_dataset.sh
```

### Test

#### Test benchmark

```text
usage: test_benchmark.py [-h] [--arch {rfb_esrgan}] [-j WORKERS]
                         [-b BATCH_SIZE] [--image-size IMAGE_SIZE]
                         [--upscale-factor {16}] [--model-path MODEL_PATH]
                         [--pretrained] [--seed SEED] [--gpu GPU]
                         DIR

positional arguments:
  DIR                   Path to dataset.

optional arguments:
  -h, --help            show this help message and exit
  --arch {rfb_esrgan}   Model architecture: rfb_esrgan. (Default:
                        `rfb_esrgan`)
  -j WORKERS, --workers WORKERS
                        Number of data loading workers. (Default: 4)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size of the dataset. (Default: 64)
  --image-size IMAGE_SIZE
                        Image size of high resolution image. (Default: 512)
  --upscale-factor {16}
                        Low to high resolution scaling factor. Optional: [16].
                        (Default: 16)
  --model-path MODEL_PATH
                        Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --seed SEED           Seed for initializing testing.
  --gpu GPU             GPU id to use.
    
# Example
$ python3 test_benchmark.py --arch rfb_esrgan --pretrained --gpu 0 [image-folder with train and val folders]
```

#### Test image

```text
usage: test_image.py [-h] [--arch {rfb_esrgan}] --lr LR [--hr HR]
                     [--upscale-factor {16}] [--model-path MODEL_PATH]
                     [--pretrained] [--seed SEED] [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --arch {rfb_esrgan}   Model architecture: rfb_esrgan. (Default: `rfb_esrgan`)
  --lr LR               Test low resolution image name.
  --hr HR               Raw high resolution image name.
  --upscale-factor {16}
                        Low to high resolution scaling factor. Optional: [16].
                        (Default: 16)
  --model-path MODEL_PATH
                        Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.
  
# Example
$ python3 test_image.py --arch rfb_esrgan --lr [path-to-lr-image] --hr [Optional, path-to-hr-image] --pretrained --gpu 0
```

#### Test video

```text
usage: test_video.py [-h] [--arch {rfb_esrgan}] --file FILE
                     [--upscale-factor {16}] [--model-path MODEL_PATH]
                     [--pretrained] [--seed SEED] [--gpu GPU] [--view]

optional arguments:
  -h, --help            show this help message and exit
  --arch {rfb_esrgan}   Model architecture: rfb_esrgan. (Default:
                        `rfb_esrgan`)
  --file FILE           Test low resolution video name.
  --upscale-factor {16}
                        Low to high resolution scaling factor. Optional: [16].
                        (Default: 16)
  --model-path MODEL_PATH
                        Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.
  --view                Do you want to show SR video synchronously.

# Example
$ python3 test_video.py --arch rfb_esrgan --file [path-to-video] --pretrained --gpu 0 --view 
```

#### Test model performance

|      Model      | Params |  FLOPs | CPU Speed | GPU Speed |
|:---------------:|:------:|:------:|:---------:|:---------:|
|       rfb       | 21.50M | 100.7G |    665ms  |    150ms  |

```text
usage: calculate_model_complexity.py [-h] [--image-size IMAGE_SIZE]
                                     [--batch-size BATCH_SIZE] [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --image-size IMAGE_SIZE
                        Image size of low-resolution. (Default: 32)
  --batch-size BATCH_SIZE
                        In order to ensure the fairness test, many experiments
                        are carried out. (Default: 128)
  --gpu GPU             GPU id to use.
  
# Example (CPU: Intel i9-10900X/GPU: Nvidia GeForce RTX 2080Ti)
$ python3 setup.py install --user --prefix=""
$ python3 scripts/cal_model_complexity.py --gpu 0
```

Low resolution / Recovered High Resolution / Ground Truth

<span align="center"><img src="assets/result.png" alt="">
</span>

### Train (e.g DIV2K)

```text
usage: train.py [-h] [--arch {rfb_esrgan}] [-j WORKERS]
                [--psnr-epochs PSNR_EPOCHS]
                [--start-psnr-epoch START_PSNR_EPOCH]
                [--gan-epochs GAN_EPOCHS] [--start-gan-epoch START_GAN_EPOCH]
                [-b BATCH_SIZE] [--psnr-lr PSNR_LR] [--gan-lr GAN_LR]
                [--image-size IMAGE_SIZE] [--upscale-factor {16}]
                [--netD NETD] [--netG NETG] [--pretrained]
                [--world-size WORLD_SIZE] [--rank RANK] [--dist-url DIST_URL]
                [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
                [--multiprocessing-distributed]
                DIR

positional arguments:
  DIR                   Path to dataset.

optional arguments:
  -h, --help            show this help message and exit
  --arch {rfb_esrgan}   Model architecture: rfb_esrgan. (Default:
                        `rfb_esrgan`)
  -j WORKERS, --workers WORKERS
                        Number of data loading workers. (Default: 4)
  --psnr-epochs PSNR_EPOCHS
                        Number of total psnr epochs to run. (Default: 128)
  --start-psnr-epoch START_PSNR_EPOCH
                        Manual psnr epoch number (useful on restarts).
                        (Default: 0)
  --gan-epochs GAN_EPOCHS
                        Number of total gan epochs to run. (Default: 64)
  --start-gan-epoch START_GAN_EPOCH
                        Manual gan epoch number (useful on restarts).
                        (Default: 0)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size of the dataset. (Default: 16)
  --psnr-lr PSNR_LR     Learning rate for psnr-oral. (Default: 0.0002)
  --gan-lr GAN_LR       Learning rate for gan-oral. (Default: 0.0001)
  --image-size IMAGE_SIZE
                        Image size of high resolution image. (Default: 512)
  --upscale-factor {16}
                        Low to high resolution scaling factor. Optional: [16].
                        (Default: 16)
  --netD NETD           Path to Discriminator checkpoint.
  --netG NETG           Path to Generator checkpoint.
  --pretrained          Use pre-trained model.
  --world-size WORLD_SIZE
                        Number of nodes for distributed training.
  --rank RANK           Node rank for distributed training. (Default: -1)
  --dist-url DIST_URL   url used to set up distributed training. (Default:
                        `tcp://59.110.31.55:12345`)
  --dist-backend DIST_BACKEND
                        Distributed backend. (Default: `nccl`)
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training.
                                       
# Example (e.g DIV2K)
$ python3 train.py --arch rfb_esrgan [image-folder with train and val folders]
# Multi-processing Distributed Data Parallel Training
$ python3 train.py --arch rfb_esrgan [image-folder with train and val folders] \
                   --dist-url 'tcp://127.0.0.1:12345' \
                   --dist-backend 'nccl' \
                   --multiprocessing-distributed \
                   --world-size 1 \
                   --rank 0
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py --arch rfb_esrgan [image-folder with train and val folders] \
                   --start-psnr-epoch 10 \
                   --netG weights/PSNR_epoch10.pth \
                   --gpu 0
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Perceptual Extreme Super Resolution Network with Receptive Field Block

_Taizhang Shang, Qiuju Dai, Shengchen Zhu, Tong Yang, Yandong Guo_ <br>

**Abstract** <br>
Perceptual Extreme Super-Resolution for single image is extremely difficult, because the texture details of different
images vary greatly. To tackle this difficulty, we develop a super resolution network with receptive field block based
on Enhanced SRGAN. We call our network RFB-ESRGAN. The key contributions are listed as follows. First, for the purpose
of extracting multi-scale information and enhance the feature discriminability, we applied receptive field block (RFB)
to super resolution. RFB has achieved competitive results in object detection and classification. Second, instead of
using large convolution kernels in multi-scale receptive field block, several small kernels are used in RFB, which makes
us be able to extract detailed features and reduce the computation complexity. Third, we alternately use different
upsampling methods in the upsampling stage to reduce the high computation complexity and still remain satisfactory
performance. Fourth, we use the ensemble of 10 models of different iteration to improve the robustness of model and
reduce the noise introduced by each individual model. Our experimental results show the superior performance of
RFB-ESRGAN. According to the preliminary results of NTIRE 2020 Perceptual Extreme Super-Resolution Challenge, our
solution ranks first among all the participants.

[[Paper]](https://arxiv.org/pdf/2005.12597)

```
@misc{2005.12597,
    Author = {Taizhang Shang and Qiuju Dai and Shengchen Zhu and Tong Yang and Yandong Guo},
    Title = {Perceptual Extreme Super Resolution Network with Receptive Field Block},
    Year = {2020},
    Eprint = {arXiv:2005.12597},
}
```
