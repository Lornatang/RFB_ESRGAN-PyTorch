# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Realize the parameter configuration function of dataset, model, training and verification code."""
import torch
from torch.backends import cudnn

# ==============================================================================
# General configuration
# ==============================================================================
# Random seed to maintain reproducible results
torch.manual_seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor = 16
# Current configuration parameter method
mode = "train_rfbnet"
# Experiment name, easy to save weights and log files
exp_name = "Train_RFBNet_baseline"

# ==============================================================================
# Training RRDBNet model configuration
# ==============================================================================
if mode == "train_rfbnet":
    # Dataset address
    train_image_dir = "data/DFO2K/RFB_ESRGAN/train"
    valid_image_dir = "data/DFO2K/RFB_ESRGAN/valid"

    image_size = 512
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    resume = False
    strict = False
    start_epoch = 0
    resume_weight = ""

    # Total num epochs
    epochs = 300

    # Adam optimizer parameter for RRDBNet(p)
    model_lr = 2e-4
    model_betas = (0.9, 0.99)

    # StepLR scheduler
    step_size = epochs // 4
    gamma = 0.5

    # Print the training log every one hundred iterations
    print_frequency = 500

# ==============================================================================
# Training RFB_ESRGAN model configuration
# ==============================================================================
if mode == "train_rfb_esrgan":
    # Dataset address
    train_image_dir = "data/DFO2K/RFB_ESRGAN/train"
    valid_image_dir = "data/DFO2K/RFB_ESRGAN/valid"

    image_size = 512
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    resume = True
    strict = False
    start_epoch = 0
    resume_d_weight = ""
    resume_g_weight = "results/Train_RFBNet_baseline/g-last.pth"

    # Total num epochs
    epochs = 120

    # Loss function weight
    pixel_weight = 10.0
    content_weight = 1.0
    adversarial_weight = 0.005

    # Adam optimizer parameter for Discriminator
    d_model_lr = 1e-4
    d_model_betas = (0.9, 0.99)

    # Adam optimizer parameter for Generator
    g_model_lr = 1e-4
    g_model_betas = (0.9, 0.99)

    # MultiStepLR scheduler parameter for RFB-ESRGAN
    d_optimizer_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    g_optimizer_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    d_optimizer_gamma = 0.5
    g_optimizer_gamma = 0.5

    # Print the training log every one hundred iterations
    print_frequency = 500

# ==============================================================================
# Verify configuration
# ==============================================================================
if mode == "valid":
    # Test data address
    lr_dir = f"data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set5/GTmod12"

    model_path = f"results/{exp_name}/g-last.pth"
