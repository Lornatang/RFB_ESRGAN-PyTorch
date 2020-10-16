# RFB_ESRGAN-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of 
[Perceptual Extreme Super Resolution Network with Receptive Field Block](https://arxiv.org/abs/2005.12597).

### Table of contents
1. [Perceptual Extreme Super Resolution Network with Receptive Field Block](#about-perceptual-extreme-super-resolution-network-with-receptive-field-block)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download dataset](#download-dataset)
4. [Test](#test)
    * [Test benchmark](#test-benchmark)
    * [Test image](#test-image)
    * [Test video](#test-video)
4. [Train](#train-eg-div2k)
5. [Contributing](#contributing) 
6. [Credit](#credit)

### About Perceptual Extreme Super Resolution Network with Receptive Field Block

If you're new to RFB-ESRGAN, here's an abstract straight from the paper:

Perceptual Extreme Super-Resolution for single image is extremely difficult, because the texture details of 
different images vary greatly. To tackle this difficulty, we develop a super resolution network with 
receptive field block based on Enhanced SRGAN. We call our network RFB-ESRGAN. The key contributions are 
listed as follows. First, for the purpose of extracting multi-scale information and enhance the feature 
discriminability, we applied receptive field block (RFB) to super resolution. RFB has achieved competitive 
results in object detection and classification. Second, instead of using large convolution kernels in 
multi-scale receptive field block, several small kernels are used in RFB, which makes us be able to 
extract detailed features and reduce the computation complexity. Third, we alternately use different 
upsampling methods in the upsampling stage to reduce the high computation complexity and still remain 
satisfactory performance. Fourth, we use the ensemble of 10 models of different iteration to improve 
the robustness of model and reduce the noise introduced by each individual model. Our experimental results 
show the superior performance of RFB-ESRGAN. According to the preliminary results of 
NTIRE 2020 Perceptual Extreme Super-Resolution Challenge, our solution ranks first among all the participants.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. 
It receives a random noise z and generates images from this noise, which is called G(z).Discriminator is 
a discriminant network that discriminates whether an image is real. The input is x, x is a picture, 
and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, 
and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/Lornatang/ESRGAN-PyTorch.git
$ cd ESRGAN-PyTorch/
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
usage: test_benchmark.py [-h] [--dataroot DATAROOT] [-j N]
                         [--upscale-factor {2,4}] [--model-path PATH]
                         [--device DEVICE]

Perceptual Extreme Super Resolution Network with Receptive Field Block.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data`)
  -j N, --workers N     Number of data loading workers. (default:4)
  --upscale-factor {2,4}
                        Low to high resolution scaling factor. (default:4).
  --model-path PATH     Path to latest checkpoint for model. (default:
                        ``./weights/RFB_ESRGAN_4x.pth``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``CUDA:0``).


# Example
$ python test_benchmark.py --dataroot ./data/DIV2K --upscale-factor 4 --model-path ./weight/RFB_ESRGAN_X4.pth --device 0
```

#### Test image

```text
usage: test_image.py [-h] [--lr LR] [--hr HR] [--upscale-factor {2,4}]
                     [--model-path PATH] [--device DEVICE]

Perceptual Extreme Super Resolution Network with Receptive Field Block.

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Test low resolution image name.
  --hr HR               Raw high resolution image name.
  --upscale-factor {2,4}
                        Low to high resolution scaling factor. (default:4).
  --model-path PATH     Path to latest checkpoint for model. (default:
                        ``./weight/RFB_ESRGAN_4x.pth``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``CUDA:0``).

# Example
$ python test_image.py --lr ./lr.png --hr ./hr.png --upscale-factor 4 --model-path ./weight/RFB_ESRGAN_X4.pth --device 0
```

#### Test video

```text
usage: test_video.py [-h] --file FILE [--upscale-factor {2,4}]
                     [--model-path PATH] [--device DEVICE] [--view]

RFB_ESRGAN algorithm is applied to video files.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution video name.
  --upscale-factor {2,4}
                        Low to high resolution scaling factor. (default:4).
  --model-path PATH     Path to latest checkpoint for model. (default:
                        ``./weight/RFB_ESRGAN_4x.pth``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``CUDA:0``).
  --view                Super resolution real time to show.

# Example
$ python test_video.py --file ./lr.mp4 --upscale-factor 4 --model-path ./weight/RFB_ESRGAN_X4.pth --device 0
```

Low resolution / Recovered High Resolution / Ground Truth

<span align="center"><img src="assets/result.png" alt="">
</span>

### Train (e.g DIV2K)

```text
usage: train.py [-h] [--dataroot DATAROOT] [-j N] [--start-epoch N]
                [--psnr-iters N] [--iters N] [-b N] [--psnr-lr PSNR_LR]
                [--lr LR] [--upscale-factor {2,4}] [--resume_PSNR] [--resume]
                [--manualSeed MANUALSEED] [--device DEVICE]

Perceptual Extreme Super Resolution Network with Receptive Field Block.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data`)
  -j N, --workers N     Number of data loading workers. (default:4)
  --start-epoch N       manual epoch number (useful on restarts)
  --psnr-iters N        The number of iterations is needed in the training of
                        PSNR model. (default:1e6)
  --iters N             The training of srgan model requires the number of
                        iterations. (default:4e5)
  -b N, --batch-size N  mini-batch size (default: 16), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --psnr-lr PSNR_LR     Learning rate for PSNR model. (default:2e-4)
  --lr LR               Learning rate. (default:1e-4)
  --upscale-factor {2,4}
                        Low to high resolution scaling factor. (default:4).
  --resume_PSNR         Path to latest checkpoint for PSNR model.
  --resume              Path to latest checkpoint for Generator.
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:10000)
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default: ``).

# Example (e.g DIV2K)
$ python train.py --dataroot ./data/DIV2K --upscale-factor 4
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python train.py --dataroot ./data/DIV2K \
                  --upscale-factor 4        \
                  --resume_PSNR \
                  --resume
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 

### Credit

#### Perceptual Extreme Super Resolution Network with Receptive Field Block
_Taizhang Shang, Qiuju Dai, Shengchen Zhu, Tong Yang, Yandong Guo_ <br>

**Abstract** <br>
Perceptual Extreme Super-Resolution for single image is extremely difficult, because the texture details of 
different images vary greatly. To tackle this difficulty, we develop a super resolution network with 
receptive field block based on Enhanced SRGAN. We call our network RFB-ESRGAN. The key contributions are 
listed as follows. First, for the purpose of extracting multi-scale information and enhance the feature 
discriminability, we applied receptive field block (RFB) to super resolution. RFB has achieved competitive 
results in object detection and classification. Second, instead of using large convolution kernels in 
multi-scale receptive field block, several small kernels are used in RFB, which makes us be able to 
extract detailed features and reduce the computation complexity. Third, we alternately use different 
upsampling methods in the upsampling stage to reduce the high computation complexity and still remain 
satisfactory performance. Fourth, we use the ensemble of 10 models of different iteration to improve 
the robustness of model and reduce the noise introduced by each individual model. Our experimental results 
show the superior performance of RFB-ESRGAN. According to the preliminary results of 
NTIRE 2020 Perceptual Extreme Super-Resolution Challenge, our solution ranks first among all the participants.

[[Paper]](https://arxiv.org/pdf/2005.12597)

```
@misc{2005.12597,
    Author = {Taizhang Shang and Qiuju Dai and Shengchen Zhu and Tong Yang and Yandong Guo},
    Title = {Perceptual Extreme Super Resolution Network with Receptive Field Block},
    Year = {2020},
    Eprint = {arXiv:2005.12597},
}
```
