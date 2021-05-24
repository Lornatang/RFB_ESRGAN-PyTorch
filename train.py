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
import argparse
import logging
import math
import os
import random
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tensorboardX import SummaryWriter

import rfb_esrgan_pytorch.models as models
from rfb_esrgan_pytorch.dataset import BaseTestDataset
from rfb_esrgan_pytorch.dataset import BaseTrainDataset
from rfb_esrgan_pytorch.loss import VGGLoss
from rfb_esrgan_pytorch.models.discriminator import discriminator_for_vgg
from rfb_esrgan_pytorch.utils.common import AverageMeter
from rfb_esrgan_pytorch.utils.common import ProgressMeter
from rfb_esrgan_pytorch.utils.common import configure
from rfb_esrgan_pytorch.utils.common import create_folder
from rfb_esrgan_pytorch.utils.estimate import test

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Perceptual Extreme Super Resolution Network with Receptive Field Block.")
parser.add_argument("data", metavar="DIR",
                    help="Path to dataset.")
parser.add_argument("-a", "--arch", metavar="ARCH", default="rfb",
                    choices=model_names,
                    help="Model architecture: " +
                         " | ".join(model_names) +
                         ". (Default: `rfb`)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="Number of data loading workers. (Default: 4)")
parser.add_argument("--psnr-epochs", default=1435, type=int, metavar="N",
                    help="Number of total psnr epochs to run. (Default: 1435)")
parser.add_argument("--start-psnr-epoch", default=0, type=int, metavar='N',
                    help="Manual psnr epoch number (useful on restarts). (Default: 0)")
parser.add_argument("--gan-epochs", default=574, type=int, metavar="N",
                    help="Number of total gan epochs to run. (Default: 574)")
parser.add_argument("--start-gan-epoch", default=0, type=int, metavar="N",
                    help="Manual gan epoch number (useful on restarts). (Default: 0)")
parser.add_argument("-b", "--batch-size", default=16, type=int,
                    metavar="N",
                    help="Mini-batch size (default: 16), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--sampler-frequency", default=1, type=int, metavar="N",
                    help="If there are many datasets, this method can be used "
                         "to increase the number of epochs. (Default:1)")
parser.add_argument("--psnr-lr", type=float, default=0.0002,
                    help="Learning rate for psnr-oral. (Default: 0.0002)")
parser.add_argument("--gan-lr", type=float, default=0.0001,
                    help="Learning rate for gan-oral. (Default: 0.0001)")
parser.add_argument("--image-size", type=int, default=512,
                    help="Image size of high resolution image. (Default: 512)")
parser.add_argument("--upscale-factor", type=int, default=16, choices=[4, 16],
                    help="Low to high resolution scaling factor. Optional: [4, 16]. (Default: 16)")
parser.add_argument("--model-path", default="", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model.")
parser.add_argument("--resume-psnr", default="", type=str, metavar="PATH",
                    help="Path to latest psnr-oral checkpoint.")
parser.add_argument("--resume-d", default="", type=str, metavar="PATH",
                    help="Path to latest -oral checkpoint.")
parser.add_argument("--resume-g", default="", type=str, metavar="PATH",
                    help="Path to latest psnr-oral checkpoint.")
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="Use pre-trained model.")
parser.add_argument("--world-size", default=-1, type=int,
                    help="Number of nodes for distributed training.")
parser.add_argument("--rank", default=-1, type=int,
                    help="Node rank for distributed training. (Default: -1)")
parser.add_argument("--dist-url", default="tcp://59.110.31.55:12345", type=str,
                    help="url used to set up distributed training. (Default: `tcp://59.110.31.55:12345`)")
parser.add_argument("--dist-backend", default="nccl", type=str,
                    help="Distributed backend. (Default: `nccl`)")
parser.add_argument("--seed", default=None, type=int,
                    help="Seed for initializing training.")
parser.add_argument("--gpu", default=None, type=int,
                    help="GPU id to use.")
parser.add_argument("--multiprocessing-distributed", action="store_true",
                    help="Use multi-processing distributed training to launch "
                         "N processes per node, which has N GPUs. This is the "
                         "fastest way to use PyTorch for either single node or "
                         "multi node data parallel training.")

best_psnr = 0.0
best_ssim = 0.0
# Load base low-resolution image.
base_image = transforms.ToTensor()(Image.open(os.path.join("assets", "butterfly.png")))
base_image = base_image.unsqueeze(0)
logger.info("Loaded `butterfly.png` successful.")


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed training. "
                      "This will turn on the CUDNN deterministic setting, "
                      "which can slow down your training considerably! "
                      "You may see unexpected behavior when restarting "
                      "from checkpoints.")

    if args.gpu is not None:
        logger.warning("You have chosen a specific GPU. This will completely disable data parallelism.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_psnr, best_ssim, base_image
    args.gpu = gpu

    if args.gpu is not None:
        logger.info(f"Use GPU: {args.gpu} for training.")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    # create model
    generator = configure(args)
    discriminator = discriminator_for_vgg(image_size=args.image_size)

    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            discriminator.cuda(args.gpu)
            generator.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            discriminator = nn.parallel.DistributedDataParallel(module=discriminator, device_ids=[args.gpu])
            generator = nn.parallel.DistributedDataParallel(module=generator, device_ids=[args.gpu])
        else:
            discriminator.cuda()
            generator.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            discriminator = nn.parallel.DistributedDataParallel(discriminator)
            generator = nn.parallel.DistributedDataParallel(generator)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        discriminator = discriminator.cuda(args.gpu)
        generator = generator.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            discriminator.features = torch.nn.DataParallel(discriminator.features)
            generator.features = torch.nn.DataParallel(generator.features)
            discriminator.cuda()
            generator.cuda()
        else:
            discriminator = torch.nn.DataParallel(discriminator).cuda()
            generator = torch.nn.DataParallel(generator).cuda()

    # Loss = 10 * pixel loss + content loss + 0.005 * adversarial loss
    pixel_criterion = nn.L1Loss().cuda(args.gpu)
    content_criterion = VGGLoss().cuda(args.gpu)
    adversarial_criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
    logger.info(f"Losses function information:\n"
                f"\tPixel:       L1Loss\n"
                f"\tContent:     VGG19_35th\n"
                f"\tAdversarial: BCEWithLogitsLoss")

    # All optimizer function and scheduler function.
    psnr_optimizer = torch.optim.Adam(generator.parameters(), lr=args.psnr_lr, betas=(0.9, 0.99))
    psnr_epoch_indices = math.floor(args.psnr_epochs // 4)
    psnr_scheduler = torch.optim.lr_scheduler.StepLR(psnr_optimizer, step_size=psnr_epoch_indices, gamma=0.5)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.gan_lr, betas=(0.9, 0.99))
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.9, 0.99))
    interval_epoch = math.ceil(args.gan_epochs // 8)
    gan_epoch_indices = [interval_epoch, interval_epoch * 2, interval_epoch * 4, interval_epoch * 6]
    discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_optimizer, milestones=gan_epoch_indices, gamma=0.5)
    generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(generator_optimizer, milestones=gan_epoch_indices, gamma=0.5)
    logger.info(f"Optimizer information:\n"
                f"\tPSNR learning rate:          {args.psnr_lr}\n"
                f"\tDiscriminator learning rate: {args.gan_lr}\n"
                f"\tGenerator learning rate:     {args.gan_lr}\n"
                f"\tPSNR optimizer:              Adam, [betas=(0.9,0.99)]\n"
                f"\tDiscriminator optimizer:     Adam, [betas=(0.9,0.99)]\n"
                f"\tGenerator optimizer:         Adam, [betas=(0.9,0.99)]\n"
                f"\tPSNR scheduler:              StepLR, [step_size=psnr_epoch_indices, gamma=0.5]\n"
                f"\tDiscriminator scheduler:     MultiStepLR, [milestones=epoch_indices, gamma=0.5]\n"
                f"\tGenerator scheduler:         MultiStepLR, [milestones=epoch_indices, gamma=0.5]")

    logger.info("Load training dataset")
    # Selection of appropriate treatment equipment.
    train_dataset = BaseTrainDataset(os.path.join(args.data, "train"), args.image_size, args.upscale_factor)
    test_dataset = BaseTestDataset(os.path.join(args.data, "test"), args.image_size, args.upscale_factor)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   pin_memory=True,
                                                   sampler=train_sampler,
                                                   num_workers=args.workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=args.workers)

    logger.info(f"Dataset information:\n"
                f"\tTrain Path:              {os.getcwd()}/{args.data}/train\n"
                f"\tTest Path:               {os.getcwd()}/{args.data}/test\n"
                f"\tNumber of train samples: {len(train_dataset)}\n"
                f"\tNumber of test samples:  {len(test_dataset)}\n"
                f"\tNumber of train batches: {len(train_dataloader)}\n"
                f"\tNumber of test batches:  {len(test_dataloader)}\n"
                f"\tShuffle of train:        True\n"
                f"\tShuffle of test:         False\n"
                f"\tSampler of train:        {bool(train_sampler)}\n"
                f"\tSampler of test:         None\n"
                f"\tWorkers of train:        {args.workers}\n"
                f"\tWorkers of test:         {args.workers}")

    # optionally resume from a checkpoint
    if args.resume_psnr:
        if os.path.isfile(args.resume_psnr):
            logger.info(f"Loading checkpoint '{args.resume_psnr}'.")
            if args.gpu is None:
                checkpoint = torch.load(args.resume_psnr)
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(args.resume_psnr, map_location=f"cuda:{args.gpu}")
            args.start_psnr_epoch = checkpoint["epoch"]
            best_psnr = checkpoint["best_psnr"]
            if args.gpu is not None:
                # best_psnr may be from a checkpoint from a different GPU
                best_psnr = best_psnr.to(args.gpu)
            generator.load_state_dict(checkpoint["state_dict"])
            psnr_optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"Loaded checkpoint '{args.resume_psnr}' (epoch {checkpoint['epoch']}).")
        else:
            logger.info(f"No checkpoint found at '{args.resume_psnr}'.")

    if args.resume_d or args.resume_g:
        if os.path.isfile(args.resume_d) or os.path.isfile(args.resume_g):
            logger.info(f"Loading checkpoint '{args.resume_d}'.")
            logger.info(f"Loading checkpoint '{args.resume_g}'.")
            if args.gpu is None:
                checkpoint_d = torch.load(args.resume_d)
                checkpoint_g = torch.load(args.resume_g)
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint_d = torch.load(args.resume_d, map_location=f"cuda:{args.gpu}")
                checkpoint_g = torch.load(args.resume_g, map_location=f"cuda:{args.gpu}")
            args.start_gan_epoch = checkpoint_g["epoch"]
            best_psnr = checkpoint_g["best_psnr"]
            if args.gpu is not None:
                # best_psnr may be from a checkpoint from a different GPU
                best_psnr = best_psnr.to(args.gpu)
            discriminator.load_state_dict(checkpoint_d["state_dict"])
            discriminator_optimizer.load_state_dict(checkpoint_d["optimizer"])
            generator.load_state_dict(checkpoint_g["state_dict"])
            generator_optimizer.load_state_dict(checkpoint_g["optimizer"])
            logger.info(f"Loaded checkpoint '{args.resume_d}' (epoch {checkpoint_d['epoch']}).")
            logger.info(f"Loaded checkpoint '{args.resume_g}' (epoch {checkpoint_g['epoch']}).")
        else:
            logger.info(f"No checkpoint found at '{args.resume_d}' or '{args.resume_g}'.")

    cudnn.benchmark = True

    # The mixed precision training is used in PSNR-oral.
    scaler = amp.GradScaler()
    logger.info("Turn on mixed precision training.")

    # Create a SummaryWriter at the beginning of training.
    psnr_writer = SummaryWriter(f"runs/{args.arch}_psnr_logs")
    gan_writer = SummaryWriter(f"runs/{args.arch}_gan_logs")

    logger.info(f"Train information:\n"
                f"\tPSNR-oral epochs: {args.psnr_epochs}\n"
                f"\tGAN-oral epochs:  {args.gan_epochs}")

    for epoch in range(args.start_psnr_epoch, args.psnr_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_psnr(dataloader=train_dataloader,
                   model=generator,
                   criterion=pixel_criterion,
                   optimizer=psnr_optimizer,
                   epoch=epoch,
                   scaler=scaler,
                   writer=psnr_writer,
                   args=args)

        psnr_scheduler.step()

        # Test for every epoch.
        psnr, ssim, lpips, gmsd = test(dataloader=test_dataloader, model=generator, gpu=args.gpu)
        psnr_writer.add_scalar("PSNR_Test/PSNR", psnr, epoch + 1)
        psnr_writer.add_scalar("PSNR_Test/SSIM", ssim, epoch + 1)
        psnr_writer.add_scalar("PSNR_Test/LPIPS", lpips, epoch + 1)
        psnr_writer.add_scalar("PSNR_Test/GMSD", gmsd, epoch + 1)

        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            torch.save({"epoch": epoch + 1,
                        "arch": args.arch,
                        "best_psnr": best_psnr,
                        "state_dict": generator.state_dict(),
                        "optimizer": psnr_optimizer.state_dict(),
                        }, os.path.join("weights", f"PSNR_epoch{epoch}.pth"))
            if is_best:
                torch.save(generator.state_dict(), os.path.join("weights", f"PSNR.pth"))

    # Load best model weight.
    generator.load_state_dict(torch.load(os.path.join("weights", f"PSNR.pth"), map_location=f"cuda:{args.gpu}"))

    for epoch in range(args.start_gan_epoch, args.gan_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_gan(dataloader=train_dataloader,
                  discriminator=discriminator,
                  discriminator_optimizer=discriminator_optimizer,
                  generator=generator,
                  generator_optimizer=generator_optimizer,
                  pixel_criterion=pixel_criterion,
                  content_criterion=content_criterion,
                  adversarial_criterion=adversarial_criterion,
                  epoch=epoch,
                  scaler=scaler,
                  writer=gan_writer,
                  args=args)

        discriminator_scheduler.step()
        generator_scheduler.step()

        # Test for every epoch.
        psnr, ssim, lpips, gmsd = test(dataloader=test_dataloader, model=generator, gpu=args.gpu)
        gan_writer.add_scalar("GAN_Test/PSNR", psnr, epoch + 1)
        gan_writer.add_scalar("GAN_Test/SSIM", ssim, epoch + 1)
        gan_writer.add_scalar("GAN_Test/LPIPS", lpips, epoch + 1)
        gan_writer.add_scalar("GAN_Test/GMSD", gmsd, epoch + 1)

        is_best = ssim > best_ssim
        best_ssim = max(ssim, best_ssim)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            torch.save({"epoch": epoch + 1,
                        "arch": "vgg",
                        "state_dict": discriminator.state_dict(),
                        "optimizer": discriminator_optimizer.state_dict()
                        }, os.path.join("weights", f"Discriminator_epoch{epoch}.pth"))
            torch.save({"epoch": epoch + 1,
                        "arch": args.arch,
                        "best_psnr": best_psnr,
                        "state_dict": generator.state_dict(),
                        "optimizer": generator_optimizer.state_dict()
                        }, os.path.join("weights", f"Generator_epoch{epoch}.pth"))
            if is_best:
                torch.save(generator.state_dict(), os.path.join("weights", f"GAN.pth"))


def train_psnr(dataloader: torch.utils.data.DataLoader,
               model: nn.Module,
               criterion: nn.L1Loss,
               optimizer: torch.optim.Adam,
               epoch: int,
               scaler: amp.GradScaler,
               writer: SummaryWriter,
               args: argparse.ArgumentParser.parse_args):
    batch_time = AverageMeter("Time", ":6.4f")
    losses = AverageMeter("Loss", ":.6f")
    progress = ProgressMeter(num_batches=len(dataloader),
                             meters=[batch_time, losses],
                             prefix=f"Epoch: [{epoch}]")

    # switch to train mode
    model.train()

    end = time.time()
    for i, (lr, hr) in enumerate(dataloader):
        # Move data to special device.
        if args.gpu is not None:
            lr = lr.cuda(args.gpu, non_blocking=True)
            hr = hr.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad()

        with amp.autocast():
            sr = model(lr)
            loss = criterion(sr, hr)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        losses.update(loss.item(), lr.size(0))

        iters = i + epoch * len(dataloader) + 1
        writer.add_scalar("PSNR_Train/MSE_Loss", loss.item(), iters)

        # Output results every 100 batches.
        if i % 100 == 0:
            progress.display(i)

    # Each Epoch validates the model once.
    sr = model(base_image)
    vutils.save_image(sr.detach(), os.path.join("runs", f"PSNR_epoch_{epoch}.png"))


def train_gan(dataloader: torch.utils.data.DataLoader,
              discriminator: nn.Module,
              discriminator_optimizer: torch.optim.Adam,
              generator: nn.Module,
              generator_optimizer: torch.optim.Adam,
              pixel_criterion: nn.L1Loss,
              content_criterion: VGGLoss,
              adversarial_criterion: nn.BCEWithLogitsLoss,
              epoch: int,
              scaler: amp.GradScaler,
              writer: SummaryWriter,
              args: argparse.ArgumentParser.parse_args):
    batch_time = AverageMeter("Time", ":.4f")
    d_losses = AverageMeter("D Loss", ":.6f")
    g_losses = AverageMeter("G Loss", ":.6f")
    pixel_losses = AverageMeter("Pixel Loss", ":6.4f")
    content_losses = AverageMeter("Content Loss", ":6.4f")
    adversarial_losses = AverageMeter("Adversarial Loss", ":6.4f")

    progress = ProgressMeter(num_batches=len(dataloader),
                             meters=[batch_time, d_losses, g_losses, pixel_losses, content_losses, adversarial_losses],
                             prefix=f"Epoch: [{epoch}]")

    # switch to train mode
    discriminator.train()
    generator.train()

    end = time.time()
    for i, (lr, hr) in enumerate(dataloader):
        # Move data to special device.
        if args.gpu is not None:
            lr = lr.cuda(args.gpu, non_blocking=True)
            hr = hr.cuda(args.gpu, non_blocking=True)
        batch_size = lr.size(0)

        # The real sample label is 1, and the generated sample label is 0.
        real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype).cuda(args.gpu, non_blocking=True)
        fake_label = torch.full((batch_size, 1), 0, dtype=lr.dtype).cuda(args.gpu, non_blocking=True)

        ##############################################
        # (1) Update D network: E(hr)[fake(C(D(hr) - E(sr)C(sr)))] + E(sr)[fake(C(fake) - E(real)C(real))]
        ##############################################
        discriminator_optimizer.zero_grad()

        with amp.autocast():
            sr = generator(lr)
            # It makes the discriminator distinguish between real sample and fake sample.
            real_output = discriminator(hr)
            fake_output = discriminator(sr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            d_loss_real = adversarial_criterion(real_output - torch.mean(fake_output), real_label)
            d_loss_fake = adversarial_criterion(fake_output - torch.mean(real_output), fake_label)

            # Count all discriminator losses.
            d_loss = (d_loss_real + d_loss_fake) / 2

        scaler.scale(d_loss).backward()
        scaler.step(discriminator_optimizer)
        scaler.update()

        ##############################################
        # (2) Update G network: E(hr)[sr(C(D(hr) - E(sr)C(sr)))] + E(sr)[sr(C(fake) - E(real)C(real))]
        ##############################################
        generator_optimizer.zero_grad()

        with amp.autocast():
            sr = generator(lr)
            # It makes the discriminator unable to distinguish the real samples and fake samples.
            real_output = discriminator(hr.detach())
            fake_output = discriminator(sr)

            # Calculate the absolute value of pixels with L1 loss.
            pixel_loss = pixel_criterion(sr, hr.detach())
            # The 35th layer in VGG19 is used as the feature extractor by default.
            content_loss = content_criterion(sr, hr.detach())
            # Adversarial loss for real and fake images (relativistic average GAN)
            adversarial_loss = adversarial_criterion(fake_output - torch.mean(real_output), real_label)

            # Count all generator losses.
            g_loss = 10 * pixel_loss + 1 * content_loss + 0.005 * adversarial_loss

        scaler.scale(g_loss).backward()
        scaler.step(generator_optimizer)
        scaler.update()

        # Set generator gradients to zero.
        generator.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        d_losses.update(d_loss.item(), lr.size(0))
        g_losses.update(g_loss.item(), lr.size(0))
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))

        iters = i + epoch * len(dataloader) + 1
        writer.add_scalar("GAN_Train/D_Loss", d_loss.item(), iters)
        writer.add_scalar("GAN_Train/G_Loss", g_loss.item(), iters)
        writer.add_scalar("GAN_Train/Pixel_Loss", pixel_loss.item(), iters)
        writer.add_scalar("GAN_Train/Content_Loss", content_loss.item(), iters)
        writer.add_scalar("GAN_Train/Adversarial_Loss", adversarial_loss.item(), iters)

        # Output results every 100 batches.
        if i % 100 == 0:
            progress.display(i)

    # Each Epoch validates the model once.
    sr = generator(base_image)
    vutils.save_image(sr.detach(), os.path.join("runs", f"GAN_epoch_{epoch}.png"))


if __name__ == "__main__":
    print("##################################################\n")
    print("Run Training Engine.\n")

    create_folder("runs")
    create_folder("weights")

    logger.info("TrainingEngine:")
    print("\tAPI version .......... 0.2.0")
    print("\tBuild ................ 2021.05.20")
    print("##################################################\n")
    main()
    logger.info("All training has been completed successfully.\n")
