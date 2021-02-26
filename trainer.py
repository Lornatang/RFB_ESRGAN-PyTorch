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
import logging
import math
import os
import time

import lpips
import torch.cuda.amp as amp
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import esrgan_pytorch.models as models
from esrgan_pytorch.dataset import CustomTestDataset
from esrgan_pytorch.dataset import CustomTrainDataset
from esrgan_pytorch.loss import VGGLoss
from esrgan_pytorch.models.discriminator import discriminator
from esrgan_pytorch.utils.common import init_torch_seeds
from esrgan_pytorch.utils.common import save_checkpoint
from esrgan_pytorch.utils.device import select_device
from esrgan_pytorch.utils.estimate import test_gan
from esrgan_pytorch.utils.estimate import test_psnr

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def train_psnr(epoch: int,
               total_epoch: int,
               total_iters: int,
               dataloader: torch.utils.data.DataLoader,
               model: nn.Module,
               pixel_criterion: nn.L1Loss,
               psnr_criterion: nn.MSELoss,
               optimizer: torch.optim.Adam,
               scheduler: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
               scaler: amp.GradScaler,
               writer: SummaryWriter,
               device: torch.device):
    # switch train mode.
    model.train()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (input, target) in progress_bar:
        # Move data to special device.
        lr = input.to(device)
        hr = target.to(device)

        optimizer.zero_grad()
        # Runs the forward pass with autocasting.
        with amp.autocast():
            # Generating fake high resolution images from real low resolution images.
            sr = model(lr)
            # The L1 Loss of the generated fake high-resolution image and real high-resolution image is calculated.
            pixel_loss = pixel_criterion(sr, hr)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(pixel_loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        progress_bar.set_description(f"[{epoch + 1}/{total_epoch}][{i + 1}/{len(dataloader)}] "
                                     f"L1 Loss: {pixel_loss.item():.6f}")

        iters = i + epoch * len(dataloader) + 1
        writer.add_scalar("Train/L1 Loss", pixel_loss.item(), iters)
        writer.add_scalar("Train/PSNR", 10 * math.log10(1. / psnr_criterion(sr, hr).item()), iters)

        # The image is saved every 1000 epoch.
        if iters % 1000 == 0:
            vutils.save_image(hr, os.path.join("runs", "hr", f"RRDBNet_{iters}.bmp"))
            sr = model(lr)
            vutils.save_image(sr.detach(), os.path.join("runs", "sr", f"RRDBNet_{iters}.bmp"))

        if iters == int(total_iters):  # If the iteration is reached, exit.
            break

    scheduler.step()


def train_gan(epoch: int,
              total_epoch: int,
              total_iters: int,
              dataloader: torch.utils.data.DataLoader,
              discriminator: nn.Module,
              generator: nn.Module,
              pixel_criterion: nn.L1Loss,
              perceptual_criterion: VGGLoss,
              adversarial_criterion: nn.BCEWithLogitsLoss,
              discriminator_optimizer: torch.optim.Adam,
              generator_optimizer: torch.optim.Adam,
              discriminator_scheduler: torch.optim.lr_scheduler,
              generator_scheduler: torch.optim.lr_scheduler,
              scaler: amp.GradScaler,
              writer: SummaryWriter,
              device: torch.device):
    # switch train mode.
    generator.train()
    discriminator.train()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (input, target) in progress_bar:
        lr = input.to(device)
        hr = target.to(device)
        batch_size = lr.size(0)

        # The real sample label is 1, and the generated sample label is 0.
        real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype, device=device)
        fake_label = torch.full((batch_size, 1), 0, dtype=lr.dtype, device=device)

        ##############################################
        # (1) Update D network: E(x~real)[fake(D(x))] + E(x~fake)[fake(D(x))]
        ##############################################
        # Set discriminator gradients to zero.
        discriminator_optimizer.zero_grad()
        # Runs the forward pass with autocasting.
        with amp.autocast():
            # Generating fake high resolution images from real low resolution images.
            sr = generator(lr)

            # Train with real high resolution image.
            real_output = discriminator(hr)
            fake_output = discriminator(sr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            errD_real = adversarial_criterion(real_output - torch.mean(fake_output), real_label)
            errD_fake = adversarial_criterion(fake_output - torch.mean(real_output), fake_label)

            errD = errD_fake + errD_real
            D_x = real_output.mean().item()
            D_G_z1 = fake_output.mean().item()

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(errD).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(discriminator_optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        ##############################################
        # (2) Update G network: E(x~real)[g(D(x))] + E(x~fake)[g(D(x))]
        ##############################################
        # Set discriminator gradients to zero.
        generator_optimizer.zero_grad()
        # Runs the forward pass with autocasting.
        with amp.autocast():
            # The pixel-wise MSE loss is calculated.
            pixel_loss = pixel_criterion(sr, hr)
            # According to the feature map, the root mean square error is regarded as the content loss.
            perceptual_loss = perceptual_criterion(sr, hr)
            # Train with fake high resolution image.
            real_output = discriminator(hr.detach())  # No train real fake image.
            fake_output = discriminator(sr)  # Train fake image.
            # Adversarial loss (relativistic average GAN)
            adversarial_loss = adversarial_criterion(fake_output - torch.mean(real_output), real_label)
            errG = perceptual_loss + 0.005 * adversarial_loss + 0.01 * pixel_loss
            D_G_z2 = fake_output.mean().item()

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(errG).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(generator_optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        progress_bar.set_description(f"[{epoch + 1}/{total_epoch}][{i + 1}/{len(dataloader)}] "
                                     f"D Loss: {errD.item():.6f} "
                                     f"Pixel Loss: {pixel_loss.item():.6f} "
                                     f"Perceptual Loss: {perceptual_loss.item():.6f} "
                                     f"Adversarial Loss: {adversarial_loss.item():.6f} "
                                     f"D(HR): {D_x:.6f} "
                                     f"D(G(SR)): {D_G_z1:.6f}/{D_G_z2:.6f}")

        iters = i + epoch * len(dataloader) + 1
        writer.add_scalar("Train/D Loss", pixel_loss.item(), iters)
        writer.add_scalar("Train/Pixel Loss", pixel_loss.item(), iters)
        writer.add_scalar("Train/Perceptual Loss", perceptual_loss.item(), iters)
        writer.add_scalar("Train/Adversarial Loss", adversarial_loss.item(), iters)
        writer.add_scalar("Train/D(x)", D_x, iters)
        writer.add_scalar("Train/D(G(SR1))", D_G_z1, iters)
        writer.add_scalar("Train/D(G(SR2))", D_G_z2, iters)

        # The image is saved every 1000 epoch.
        if iters % 1000 == 0:
            vutils.save_image(hr, os.path.join("runs", "hr", f"ESRGAN_{iters}.bmp"))
            sr = generator(lr)
            vutils.save_image(sr.detach(), os.path.join("runs", "sr", f"ESRGAN_{iters}.bmp"))

        if iters == int(total_iters):  # If the iteration is reached, exit.
            break

    # Dynamic adjustment of learning rate
    discriminator_scheduler.step()
    generator_scheduler.step()


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Set random initialization seed, easy to reproduce.
        init_torch_seeds(args.manualSeed)

        logger.info("Load training dataset")
        # Selection of appropriate treatment equipment.
        train_dataset = CustomTrainDataset(root=os.path.join(args.data, "train"),
                                           sampler_frequency=args.sampler_frequency)
        test_dataset = CustomTestDataset(root=os.path.join(args.data, "test"),
                                         image_size=args.image_size,
                                         sampler_frequency=args.sampler_frequency)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            pin_memory=False,
                                                            num_workers=int(args.workers))
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                           batch_size=args.batch_size,
                                                           shuffle=False,
                                                           pin_memory=False,
                                                           num_workers=int(args.workers))

        logger.info(f"Train Dataset information:\n"
                    f"\tTrain Dataset dir is `{os.getcwd()}/{args.data}/train`\n"
                    f"\tBatch size is {args.batch_size}\n"
                    f"\tWorkers is {int(args.workers)}\n"
                    f"\tLoad dataset to CUDA")
        logger.info(f"Test Dataset information:\n"
                    f"\tTest Dataset dir is `{os.getcwd()}/{args.data}/test`\n"
                    f"\tBatch size is {args.batch_size}\n"
                    f"\tWorkers is {int(args.workers)}\n"
                    f"\tLoad dataset to CUDA")

        # Construct network architecture model of generator and discriminator.
        self.device = select_device(args.device, batch_size=1)
        if args.pretrained:
            logger.info(f"Using pre-trained model `{args.arch}`")
            self.generator = models.__dict__[args.arch](pretrained=True).to(self.device)
        else:
            logger.info(f"Creating model `{args.arch}`")
            self.generator = models.__dict__[args.arch]().to(self.device)
        logger.info(f"Creating discriminator model")
        self.discriminator = discriminator().to(self.device)

        # Parameters of pre training model.
        self.start_psnr_epoch = math.floor(args.start_psnr_iter / len(self.train_dataloader))
        self.psnr_epochs = math.ceil(args.psnr_iters / len(self.train_dataloader))
        psnr_epoch_indices = math.floor(self.psnr_epochs / 4)
        self.psnr_optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.psnr_lr, betas=(0.9, 0.999))
        self.psnr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.psnr_optimizer,
                                                                                   T_0=psnr_epoch_indices,
                                                                                   T_mult=1,
                                                                                   eta_min=1e-7)

        logger.info(f"Pre-training model training parameters:\n"
                    f"\tIters is {args.psnr_iters}\n"
                    f"\tEpoch is {self.psnr_epochs}\n"
                    f"\tOptimizer Adam\n"
                    f"\tLearning rate {args.lr}\n"
                    f"\tBetas (0.9, 0.999)")

        # Create a SummaryWriter at the beginning of training.
        self.psnr_writer = SummaryWriter(f"runs/RRDBNet_{int(time.time())}_logs")
        self.gan_writer = SummaryWriter(f"runs/ESRGAN_{int(time.time())}_logs")

        # Creates a GradScaler once at the beginning of training.
        self.scaler = amp.GradScaler()
        logger.info(f"Turn on mixed precision training.")

        # Parameters of GAN training model.
        self.start_epoch = math.floor(args.start_iter / len(self.train_dataloader))
        self.epochs = math.ceil(args.iters / len(self.train_dataloader))
        interval_epoch = math.ceil(self.epochs / 8)
        epoch_indices = [interval_epoch, interval_epoch * 2, interval_epoch * 4, interval_epoch * 6]
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
        self.discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.discriminator_optimizer,
                                                                            milestones=epoch_indices,
                                                                            gamma=0.5)
        self.generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.generator_optimizer,
                                                                        milestones=epoch_indices,
                                                                        gamma=0.5)
        logger.info(f"All model training parameters:\n"
                    f"\tIters is {args.iters}\n"
                    f"\tEpoch is {self.epochs}\n"
                    f"\tOptimizer is Adam\n"
                    f"\tLearning rate is {args.lr}\n"
                    f"\tBetas is (0.9, 0.999)\n"
                    f"\tScheduler is MultiStepLR")

        # We use VGG5.4 as our feature extraction method by default.
        self.perceptual_criterion = VGGLoss().to(self.device)
        # Loss = perceptual loss + 0.005 * adversarial loss + 0.01 * pixel loss
        self.pixel_criterion = nn.L1Loss().to(self.device)
        self.adversarial_criterion = nn.BCEWithLogitsLoss().to(self.device)
        # LPIPS Evaluating.
        self.lpips_criterion = lpips.LPIPS(net="vgg", verbose=False).to(self.device)
        # PSNR Evaluating
        self.psnr_criterion = nn.MSELoss().to(self.device)
        logger.info(f"Loss function:\n"
                    f"\tPixel loss is L1Loss\n"
                    f"\tPerceptual loss is VGGLoss\n"
                    f"\tAdversarial loss is BCEWithLogitsLoss")

    def run(self):
        args = self.args
        best_psnr = 0.
        best_lpips = 1.

        # Loading PSNR pre training model.
        if args.netP != "":
            checkpoint = torch.load(args.netP)
            self.args.start_psnr_iter = checkpoint["iter"]
            best_psnr = checkpoint["best_psnr"]
            self.generator.load_state_dict(checkpoint["state_dict"])

        # Start train PSNR model.
        logger.info("Staring training PSNR model")
        logger.info(f"Training for {args.psnr_iters} iters")

        if args.start_psnr_iter < args.psnr_iters:
            for psnr_epoch in range(self.start_psnr_epoch, self.psnr_epochs):
                # Train epoch.
                train_psnr(epoch=psnr_epoch,
                           total_epoch=self.psnr_epochs,
                           total_iters=args.psnr_iters,
                           dataloader=self.train_dataloader,
                           model=self.generator,
                           pixel_criterion=self.pixel_criterion,
                           psnr_criterion=self.psnr_criterion,
                           optimizer=self.psnr_optimizer,
                           scheduler=self.psnr_scheduler,
                           scaler=self.scaler,
                           writer=self.psnr_writer,
                           device=self.device)

                # Test for every epoch.
                psnr = test_psnr(model=self.generator,
                                 psnr_criterion=self.psnr_criterion,
                                 dataloader=self.test_dataloader,
                                 device=self.device)
                iters = (psnr_epoch + 1) * len(self.train_dataloader)
                self.psnr_writer.add_scalar("Test/PSNR", psnr, psnr_epoch)

                # remember best psnr and save checkpoint
                is_best = psnr > best_psnr
                best_psnr = max(psnr, best_psnr)

                # The model is saved every 1 epoch.
                save_checkpoint(
                    {"iter": iters,
                     "state_dict": self.generator.state_dict(),
                     "best_psnr": best_psnr,
                     "optimizer": self.psnr_optimizer.state_dict()
                     }, is_best,
                    os.path.join("weights", f"RRDBNet_iter_{iters}.pth"),
                    os.path.join("weights", f"RRDBNet.pth"))
        else:
            logger.info("The weight of pre training model is found.")

        # Load best generator model weight.
        self.generator.load_state_dict(torch.load(os.path.join("weights", f"RRDBNet.pth"), self.device))

        # Loading SRGAN training model.
        if args.netG != "":
            checkpoint = torch.load(args.netG)
            self.args.start_psnr_iter = checkpoint["iter"]
            best_lpips = checkpoint["best_lpips"]
            self.generator.load_state_dict(checkpoint["state_dict"])

        if args.start_iter < args.iters:
            for epoch in range(self.start_epoch, self.epochs):
                # Train epoch.
                train_gan(epoch=epoch,
                          total_epoch=self.epochs,
                          total_iters=args.iters,
                          dataloader=self.train_dataloader,
                          discriminator=self.discriminator,
                          generator=self.generator,
                          pixel_criterion=self.pixel_criterion,
                          perceptual_criterion=self.perceptual_criterion,
                          adversarial_criterion=self.adversarial_criterion,
                          discriminator_optimizer=self.discriminator_optimizer,
                          generator_optimizer=self.generator_optimizer,
                          discriminator_scheduler=self.discriminator_scheduler,
                          generator_scheduler=self.generator_scheduler,
                          scaler=self.scaler,
                          writer=self.gan_writer,
                          device=self.device)
                # Test for every epoch.
                psnr, lpips = test_gan(model=self.generator,
                                       psnr_criterion=self.psnr_criterion,
                                       lpips_criterion=self.lpips_criterion,
                                       dataloader=self.test_dataloader,
                                       device=self.device)
                iters = (epoch + 1) * len(self.train_dataloader)
                self.gan_writer.add_scalar("Test/PSNR", psnr, epoch)
                self.gan_writer.add_scalar("Test/LPIPS", lpips, epoch)

                # remember best psnr and save checkpoint
                is_best = lpips < best_lpips
                best_psnr = max(psnr, best_psnr)
                best_lpips = min(lpips, best_lpips)

                # The model is saved every 1 epoch.
                torch.save(self.discriminator.state_dict(), os.path.join("weights", "Discriminator.pth"))
                save_checkpoint(
                    {"iter": iters,
                     "state_dict": self.generator.state_dict(),
                     "best_psnr": best_psnr,
                     "best_lpips": best_lpips,
                     "optimizer": self.generator_optimizer.state_dict()
                     }, is_best,
                    os.path.join("weights", f"ESRGAN_iter_{iters}.pth"),
                    os.path.join("weights", f"ESRGAN.pth"))
        else:
            logger.info("The weight of GAN training model is found.")
