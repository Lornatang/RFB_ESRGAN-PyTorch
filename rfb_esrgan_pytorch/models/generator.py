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
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    "rfb_esrgan": "https://github.com/Lornatang/RFB_ESRGAN-PyTorch/releases/download/0.1.0/RFB_ESRGAN_DF2K-e31a1b2e.pth"
}


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # 16 ResidualInResidualDenseBlock layer
        residual_residual_dense_blocks = []
        for _ in range(16):
            residual_residual_dense_blocks += [ResidualInResidualDenseBlock(64, 32, 0.2)]
        self.Trunk_a = nn.Sequential(*residual_residual_dense_blocks)

        # 8 ResidualInResidualDenseBlock layer
        residual_residual_fields_dense_blocks = []
        for _ in range(8):
            residual_residual_fields_dense_blocks += [ResidualOfReceptiveFieldDenseBlock(64, 32, 0.2)]
        self.Trunk_RFB = nn.Sequential(*residual_residual_fields_dense_blocks)

        # Second conv layer post residual field blocks
        self.RFB = ReceptiveFieldBlock(64, 64, non_linearity=False)

        # Upsampling layers
        upsampling = []
        for _ in range(2):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                ReceptiveFieldBlock(64, 64),
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.PixelShuffle(upscale_factor=2),
                ReceptiveFieldBlock(64, 64)
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Next layer after upper sampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(input)
        out = self.Trunk_a(out1)
        out2 = self.Trunk_RFB(out)
        out = torch.add(out1, out2)
        out = self.RFB(out)
        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out


class ReceptiveFieldBlock(nn.Module):
    r"""This structure is similar to the main building blocks in the GoogLeNet model.
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    """

    def __init__(self, in_channels, out_channels, scale_ratio=0.2, non_linearity=True):
        super(ReceptiveFieldBlock, self).__init__()
        channels = in_channels // 4
        # shortcut layer
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3, dilation=3)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3, dilation=3)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, channels // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, (channels // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d((channels // 4) * 3, channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=5, dilation=5)
        )

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True) if non_linearity else None

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(input)

        branch1 = self.branch1(input)
        branch2 = self.branch2(input)
        branch3 = self.branch3(input)
        branch4 = self.branch4(input)

        out = torch.cat((branch1, branch2, branch3, branch4), dim=1)
        out = self.conv1x1(out)

        out = out.mul(self.scale_ratio) + shortcut
        if self.lrelu is not None:
            out = self.lrelu(out)

        return out


class ReceptiveFieldDenseBlock(nn.Module):
    r""" Inspired by the multi-scale kernels and the structure of Receptive Fields (RFs) in human visual systems,
        RFB-SSD proposed Receptive Fields Block (RFB) for object detection
    """

    def __init__(self, in_channels, growth_channels, scale_ratio):
        """

        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ReceptiveFieldDenseBlock, self).__init__()
        self.RFB1 = ReceptiveFieldBlock(in_channels, growth_channels, scale_ratio)
        self.RFB2 = ReceptiveFieldBlock(in_channels + 1 * growth_channels, growth_channels, scale_ratio)
        self.RFB3 = ReceptiveFieldBlock(in_channels + 2 * growth_channels, growth_channels, scale_ratio)
        self.RFB4 = ReceptiveFieldBlock(in_channels + 3 * growth_channels, growth_channels, scale_ratio)
        self.RFB5 = ReceptiveFieldBlock(in_channels + 4 * growth_channels, in_channels, scale_ratio,
                                        non_linearity=False)

        self.scale_ratio = scale_ratio

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        rfb1 = self.RFB1(input)
        rfb2 = self.RFB2(torch.cat((input, rfb1), dim=1))
        rfb3 = self.RFB3(torch.cat((input, rfb1, rfb2), dim=1))
        rfb4 = self.RFB4(torch.cat((input, rfb1, rfb2, rfb3), dim=1))
        rfb5 = self.RFB5(torch.cat((input, rfb1, rfb2, rfb3, rfb4), dim=1))

        return rfb5.mul(self.scale_ratio) + input


class ResidualOfReceptiveFieldDenseBlock(nn.Module):
    r"""The residual block structure of traditional RFB-ESRGAN is defined"""

    def __init__(self, in_channels=64, growth_channels=32, scale_ratio=0.2):
        """

        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualOfReceptiveFieldDenseBlock, self).__init__()
        self.RFDB1 = ReceptiveFieldDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RFDB2 = ReceptiveFieldDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RFDB3 = ReceptiveFieldDenseBlock(in_channels, growth_channels, scale_ratio)

        self.scale_ratio = scale_ratio

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.RFDB1(input)
        out = self.RFDB2(out)
        out = self.RFDB3(out)

        return out.mul(self.scale_ratio) + input


class ResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional SRGAN and Dense model is defined"""

    def __init__(self, in_channels, growth_channels, scale_ratio):
        """

        Args:
            in_channels (int): Number of channels in the input image.
            growth_channels (int): how many filters to add each layer (`k` in paper).
            scale_ratio (float): Residual channel scaling column.
        """
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + 0 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels + 1 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels + 3 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        conv2 = self.conv2(torch.cat((input, conv1), dim=1))
        conv3 = self.conv3(torch.cat((input, conv1, conv2), dim=1))
        conv4 = self.conv4(torch.cat((input, conv1, conv2, conv3), dim=1))
        conv5 = self.conv5(torch.cat((input, conv1, conv2, conv3, conv4), dim=1))

        return conv5.mul(self.scale_ratio) + input


class ResidualInResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined"""

    def __init__(self, in_channels, growth_channels, scale_ratio):
        """

        Args:
            in_channels (int): Number of channels in the input image.
            growth_channels (int): how many filters to add each layer (`k` in paper).
            scale_ratio (float): Residual channel scaling column.
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RDB2 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RDB3 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)

        self.scale_ratio = scale_ratio

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(input)
        out = self.RDB2(out)
        out = self.RDB3(out)

        return out.mul(self.scale_ratio) + input


def _gan(arch, pretrained, progress):
    model = Generator()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress,
                                              map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def rfb_esrgan(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/2005.12597>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("rfb_esrgan", pretrained, progress)
