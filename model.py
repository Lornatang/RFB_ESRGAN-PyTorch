# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn

__all__ = [
    "ResidualDenseBlock", "ResidualInResidualDenseBlock",
    "ReceptiveFieldBlock", "ReceptiveFieldDenseBlock", "ResidualOfReceptiveFieldDenseBlock",
    "UpsamplingModule",
    "Discriminator", "Generator",
    "ContentLoss",
]


class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growths (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growths: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growths * 0, growths, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growths * 1, growths, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growths * 2, growths, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growths * 3, growths, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growths * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))

        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out


class ResidualInResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growths (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growths: int) -> None:
        super(ResidualInResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growths)
        self.rdb2 = ResidualDenseBlock(channels, growths)
        self.rdb3 = ResidualDenseBlock(channels, growths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)

        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out


class ReceptiveFieldBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Modules introduced in RFBNet paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        """

        super(ReceptiveFieldBlock, self).__init__()
        branch_channels = in_channels // 4

        # shortcut layer
        self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0))

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (1, 1), dilation=1),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (1, 3), (1, 1), (0, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (3, 3), dilation=3),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 1), (1, 1), (1, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (3, 3), dilation=3),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels // 2, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels // 2, (branch_channels // 4) * 3, (1, 3), (1, 1), (0, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d((branch_channels // 4) * 3, branch_channels, (3, 1), (1, 1), (1, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (5, 5), dilation=5),
        )

        self.conv_linear = nn.Conv2d(4 * branch_channels, out_channels, (1, 1), (1, 1), (0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        shortcut = torch.mul(shortcut, 0.2)

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], 1)
        out = self.conv_linear(out)
        out = torch.add(out, shortcut)

        return out


# Source code reference from `https://arxiv.org/pdf/2005.12597.pdf`.
class ReceptiveFieldDenseBlock(nn.Module):
    """Inspired by the multi-scale kernels and the structure of Receptive Fields (RFs) in human visual systems,
    RFB-SSD proposed Receptive Fields Block (RFB) for object detection
    """

    def __init__(self, channels: int, growths: int):
        """

        Args:
            channels (int): Number of channels in the input image.
            growths (int): how many filters to add each layer (`k` in paper).
        """

        super(ReceptiveFieldDenseBlock, self).__init__()
        self.rfb1 = ReceptiveFieldBlock(channels + 0 * growths, growths)
        self.rfb2 = ReceptiveFieldBlock(channels + 1 * growths, growths)
        self.rfb3 = ReceptiveFieldBlock(channels + 2 * growths, growths)
        self.rfb4 = ReceptiveFieldBlock(channels + 3 * growths, growths)
        self.rfb5 = ReceptiveFieldBlock(channels + 4 * growths, channels)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        rfb1 = self.leaky_relu(self.rfb1(x))
        rfb2 = self.leaky_relu(self.rfb2(torch.cat([x, rfb1], 1)))
        rfb3 = self.leaky_relu(self.rfb3(torch.cat([x, rfb1, rfb2], 1)))
        rfb4 = self.leaky_relu(self.rfb4(torch.cat([x, rfb1, rfb2, rfb3], 1)))
        rfb5 = self.identity(self.rfb5(torch.cat([x, rfb1, rfb2, rfb3, rfb4], 1)))

        out = torch.mul(rfb5, 0.2)
        out = torch.add(out, identity)

        return out


# Source code reference from `https://arxiv.org/pdf/2005.12597.pdf`.
class ResidualOfReceptiveFieldDenseBlock(nn.Module):
    def __init__(self, channels: int, growths: int):
        super(ResidualOfReceptiveFieldDenseBlock, self).__init__()
        self.rfdb1 = ReceptiveFieldDenseBlock(channels, growths)
        self.rfdb2 = ReceptiveFieldDenseBlock(channels, growths)
        self.rfdb3 = ReceptiveFieldDenseBlock(channels, growths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rfdb1(x)
        out = self.rfdb2(out)
        out = self.rfdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out


class UpsamplingModule(nn.Module):
    def __init__(self, channels: int) -> None:
        super(UpsamplingModule, self).__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ReceptiveFieldBlock(channels, channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels * 4, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            ReceptiveFieldBlock(channels, channels),
            nn.LeakyReLU(0.2, True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            ReceptiveFieldBlock(channels, channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels * 4, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            ReceptiveFieldBlock(channels, channels),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)

        return out


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 512 x 512
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 256 x 256
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 128 x 128
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 64 x 64
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 32 x 32
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 16 x 16
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 16 * 16, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))

        # Trunk-a backbone network.
        trunk_a = []
        for _ in range(16):
            trunk_a.append(ResidualInResidualDenseBlock(64, 32))
        self.trunk_a = nn.Sequential(*trunk_a)

        # Trunk-RFB backbone network.
        trunk_rfb = []
        for _ in range(8):
            trunk_rfb.append(ResidualOfReceptiveFieldDenseBlock(64, 32))
        self.trunk_rfb = nn.Sequential(*trunk_rfb)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = ReceptiveFieldBlock(64, 64)

        # Upsampling convolutional layer.
        self.upsampling = UpsamplingModule(64)

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out_a = self.trunk_a(out1)
        out_rfb = self.trunk_rfb(out_a)
        out2 = self.conv2(out_rfb)
        out = torch.add(out1, out2)
        out = self.conv2(out)
        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(self) -> None:
        super(ContentLoss, self).__init__()
        # Load the VGG19 model trained on the ImageNet dataset.
        vgg19 = models.vgg19(pretrained=True).eval()
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:35])
        # Freeze model parameters.
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        # The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Standardized operations
        sr = sr.sub(self.mean).div(self.std)
        hr = hr.sub(self.mean).div(self.std)

        # Find the feature map difference between the two images
        loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss
