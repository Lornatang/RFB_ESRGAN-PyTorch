# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
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
import argparse
import csv
import logging
import os
import random

import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
from tqdm import tqdm
import torch.optim as optim

from rfb_RFB_esrgan_pytorch import DatasetFromFolder
from rfb_RFB_esrgan_pytorch import Discriminator
from rfb_RFB_esrgan_pytorch import VGG34Loss
from rfb_RFB_esrgan_pytorch import Generator
from rfb_RFB_esrgan_pytorch import init_torch_seeds
from rfb_RFB_esrgan_pytorch import load_checkpoint
from rfb_RFB_esrgan_pytorch import select_device

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Perceptual Extreme Super Resolution Network with Receptive Field Block.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to datasets. (default:`./data`)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="Number of data loading workers. (default:4)")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                    help="manual epoch number (useful on restarts)")
parser.add_argument("--psnr-iters", default=1e6, type=int, metavar="N",
                    help="The number of iterations is needed in the training of PSNR model. (default:1e6)")
parser.add_argument("--iters", default=4e5, type=int, metavar="N",
                    help="The training of srgan model requires the number of iterations. (default:4e5)")
parser.add_argument("-b", "--batch-size", default=16, type=int, metavar="N",
                    help="mini-batch size (default: 16), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--psnr-lr", type=float, default=2e-4,
                    help="Learning rate for PSNR model. (default:2e-4)")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate. (default:1e-4)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--resume_PSNR", action="store_true",
                    help="Path to latest checkpoint for PSNR model.")
parser.add_argument("--resume", action="store_true",
                    help="Path to latest checkpoint for Generator.")
parser.add_argument("--manualSeed", type=int, default=10000,
                    help="Seed for initializing training. (default:10000)")
parser.add_argument("--device", default="",
                    help="device id i.e. `0` or `0,1` or `cpu`. (default: ``).")

args = parser.parse_args()
print(args)

output_lr_dir = f"./output/{args.upscale_factor}x/lr"
output_hr_dir = f"./output/{args.upscale_factor}x/hr"
output_sr_dir = f"./output/{args.upscale_factor}x/sr"

try:
    os.makedirs(output_lr_dir)
    os.makedirs(output_hr_dir)
    os.makedirs(output_sr_dir)
    os.makedirs("weight")
except OSError:
    pass

# Set random initialization seed, easy to reproduce.
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
init_torch_seeds(args.manualSeed)

# Selection of appropriate treatment equipment
device = select_device(args.device, batch_size=args.batch_size)

dataset = DatasetFromFolder(input_dir=f"{args.dataroot}/{args.upscale_factor}x/train/input",
                            target_dir=f"{args.dataroot}/{args.upscale_factor}x/train/target")

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         pin_memory=True,
                                         num_workers=int(args.workers))

# Construct network architecture model of generator and discriminator.
netG = Generator(upscale_factor=args.upscale_factor).to(device)
netD = Discriminator().to(device)

# Define PSNR model optimizers
psnr_epochs = int(args.psnr_iters // len(dataloader))
epoch_indices = int(psnr_epochs // 4)
optimizer = optim.Adam(netG.parameters(), lr=args.psnr_lr, betas=(0.9, 0.99))
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                           T_0=epoch_indices,
                                                           T_mult=1,
                                                           eta_min=1e-7)

# Loading PSNR pre training model
if args.resume_PSNR:
    args.start_epoch = load_checkpoint(netG, optimizer, f"./weight/RRFDBNet_PSNR_{args.upscale_factor}x_checkpoint.pth")

# We use vgg34 as our feature extraction method by default.
vgg_criterion = VGG34Loss().to(device)
# Loss = perceptual_loss + 0.005 * adversarial_loss + 0.1 * l1_loss
content_criterion = nn.L1Loss().to(device)
adversarial_criterion = nn.BCEWithLogitsLoss().to(device)

# Set the all model to training mode
netG.train()
netD.train()

# Pre-train generator using raw MSE loss
logger.info(f"[*] Start training RRFDBNet for PSNR model based on L1 loss.")
logger.info(f"[*] Generator pre-training for {psnr_epochs} epochs.")
logger.info(f"[*] Searching RRFDBNet for PSNR pretrained model weights.")

# Save the generator model based on MSE pre training to speed up the training time
if os.path.exists(f"./weight/RRFDBNet_PSNR_{args.upscale_factor}x.pth"):
    print("[*] Found RRFDBNet for PSNR pretrained model weights. Skip pre-train.")
    netG.load_state_dict(torch.load(f"./weight/RRFDBNet_PSNR_{args.upscale_factor}x.pth", map_location=device))
else:
    # Writer train RRFDBNet PSNR model log.
    if args.start_epoch == 0:
        with open(f"RRFDBNet_PSNR_{args.upscale_factor}x_Loss.csv", "w+") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "L1 Loss"])
    print("[!] Not found pretrained weights. Start training RRFDBNet for PSNR model.")
    for epoch in range(args.start_epoch, psnr_epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        avg_loss = 0.
        for i, (input, target) in progress_bar:
            # Set generator gradients to zero
            netG.zero_grad()
            # Generate data
            lr = input.to(device)
            hr = target.to(device)

            # Generating fake high resolution images from real low resolution images.
            sr = netG(lr)
            # The MSE of the generated fake high-resolution image and real high-resolution image is calculated.
            l1_loss = content_criterion(sr, hr)
            # Calculate gradients for generator
            l1_loss.backward()
            # Update generator weights
            optimizer.step()

            avg_loss += l1_loss.item()

            progress_bar.set_description(f"[{epoch + 1}/{psnr_epochs}][{i + 1}/{len(dataloader)}] "
                                         f"L1 loss: {l1_loss.item():.6f}")

            # record iter.
            total_iter = len(dataloader) * epoch + i

            # The image is saved every 5000 iterations.
            if (total_iter + 1) % 5000 == 0:
                vutils.save_image(lr,
                                  os.path.join(output_lr_dir, f"RRFDBNet_PSNR_{total_iter + 1}.bmp"), normalize=True)
                vutils.save_image(hr,
                                  os.path.join(output_hr_dir, f"RRFDBNet_PSNR_{total_iter + 1}.bmp"), normalize=True)
                vutils.save_image(sr,
                                  os.path.join(output_sr_dir, f"RRFDBNet_PSNR_{total_iter + 1}.bmp"), normalize=True)

        # The model is saved every 1 epoch.
        torch.save({"epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(),
                    "state_dict": netG.state_dict()
                    }, f"./weight/RRFDBNet_PSNR_{args.upscale_factor}x_checkpoint.pth")

        # Writer training log
        with open(f"RRFDBNet_PSNR_{args.upscale_factor}x_Loss.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss / len(dataloader)])

    torch.save(netG.state_dict(), f"./weight/RRFDBNet_PSNR_{args.upscale_factor}x.pth")
    print(f"[*] Training RRFDBNet for PSNR model done! Saving RRFDBNet for PSNR model weight to "
          f"`./weight/RRFDBNet_PSNR_{args.upscale_factor}x.pth`.")

# After training the RRFDBNet for PSNR model, set the initial iteration to 0.
args.start_epoch = 0

# Alternating training RFB-ESRGAN network.
epochs = int(args.iters // len(dataloader))
base_epoch = int(epochs // 8)
epoch_indices = [base_epoch, base_epoch * 2, base_epoch * 4, base_epoch * 6]
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9, 0.99))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.9, 0.99))
schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=epoch_indices, gamma=0.5)
schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=epoch_indices, gamma=0.5)

# Loading RFB-ESRGAN checkpoint
if args.resume:
    args.start_epoch = load_checkpoint(netD, optimizerD, f"./weight/netD_{args.upscale_factor}x_checkpoint.pth")
    args.start_epoch = load_checkpoint(netG, optimizerG, f"./weight/netG_{args.upscale_factor}x_checkpoint.pth")

# Train RFB-ESRGAN model.
logger.info(f"[*] Staring training RFB-ESRGAN model!")
logger.info(f"[*] Training for {epochs} epochs.")
# Writer train RFB-ESRGAN model log.
if args.start_epoch == 0:
    with open(f"RFB_ESRGAN_{args.upscale_factor}x_Loss.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "D Loss", "G Loss"])

for epoch in range(args.start_epoch, epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    g_avg_loss = 0.
    d_avg_loss = 0.
    for i, (input, target) in progress_bar:
        lr = input.to(device)
        hr = target.to(device)
        batch_size = lr.size(0)
        real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype, device=device)
        fake_label = torch.full((batch_size, 1), 0, dtype=lr.dtype, device=device)

        ##############################################
        # (1) Update G network: maximize - E(lr)[log(D(hr, sr))] - E(sr)[1- log(D(sr, hr))]
        ##############################################
        # Set generator gradients to zero.
        netG.zero_grad()

        # Generator first generates high resolution graph.
        sr = netG(lr)

        # According to the feature map, the root mean square error is regarded as the content loss.
        perceptual_loss = vgg_criterion(sr, hr)
        # Train with fake high resolution image.
        hr_output = netD(hr)  # No train real fake image.
        sr_output = netD(sr)  # Train fake image.
        errG_hr = adversarial_criterion(hr_output - torch.mean(sr_output), fake_label)
        errG_sr = adversarial_criterion(sr_output - torch.mean(hr_output), real_label)
        adversarial_loss = (errG_hr + errG_sr) / 2
        # Pixel level loss between two images.
        l1_loss = content_criterion(sr, hr)
        errG = perceptual_loss + 0.005 * adversarial_loss + 0.1 * l1_loss
        errG.backward()
        optimizerG.step()
        D_G_z = sr_output.mean().item()

        ##############################################
        # (2) Update D network: maximize - E(lr)[1- log(D(hr, sr))] - E(sr)[log(D(sr, hr))]
        ##############################################
        # Set discriminator gradients to zero.
        netD.zero_grad()

        # Train with real high resolution image.
        hr_output = netD(hr)  # Train real image.
        sr_output = netD(sr.detach())  # No train fake image.
        errD_hr = adversarial_criterion(hr_output - torch.mean(sr_output), real_label)
        errD_sr = adversarial_criterion(sr_output - torch.mean(hr_output), fake_label)
        errD = (errD_sr + errD_hr) / 2
        errD.backward()
        D_x = hr_output.mean().item()
        optimizerD.step()

        # Dynamic adjustment of learning rate
        schedulerD.step()
        schedulerG.step()

        d_avg_loss += errD.item()
        g_avg_loss += errG.item()

        progress_bar.set_description(f"[{epoch + 1}/{epochs}][{i + 1}/{len(dataloader)}] "
                                     f"Loss_D: {errD:.6f} Loss_G: {errG.item():.6f} "
                                     f"D(HR): {D_x:.6f} D(G(LR)): {D_G_z:.6f}")

        # record iter.
        total_iter = len(dataloader) * epoch + i

        # The image is saved every 5000 iterations.
        if (total_iter + 1) % 5000 == 0:
            vutils.save_image(lr, os.path.join(output_lr_dir, f"RFB_ESRGAN_{total_iter + 1}.bmp"), normalize=True)
            vutils.save_image(hr, os.path.join(output_hr_dir, f"RFB_ESRGAN_{total_iter + 1}.bmp"), normalize=True)
            vutils.save_image(sr, os.path.join(output_sr_dir, f"RFB_ESRGAN_{total_iter + 1}.bmp"), normalize=True)

    # The model is saved every 1 epoch.
    torch.save({"epoch": epoch + 1,
                "optimizer": optimizerD.state_dict(),
                "state_dict": netD.state_dict()
                }, f"./weight/netD_{args.upscale_factor}x_checkpoint.pth")
    torch.save({"epoch": epoch + 1,
                "optimizer": optimizerG.state_dict(),
                "state_dict": netG.state_dict()
                }, f"./weight/netG_{args.upscale_factor}x_checkpoint.pth")

    # Writer training log
    with open(f"RFB_ESRGAN_{args.upscale_factor}x_Loss.csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, d_avg_loss / len(dataloader), g_avg_loss / len(dataloader)])

torch.save(netG.state_dict(), f"./weight/RFB_ESRGAN_{args.upscale_factor}x.pth")
logger.info(f"[*] Training RFB-ESRGAN model done! Saving RFB-ESRGAN model weight "
            f"to `./weight/RFB_ESRGAN_{args.upscale_factor}x.pth`.")
