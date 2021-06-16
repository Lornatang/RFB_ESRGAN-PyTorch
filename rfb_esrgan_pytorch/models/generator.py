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
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from .utils import ReceptiveFieldBlock
from .utils import ResidualInResidualDenseBlock
from .utils import ResidualOfReceptiveFieldDenseBlock
from .utils import SubpixelConvolutionLayer

model_urls = {
    "rfb_esrgan": None
}


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # 16 ResidualInResidualDenseBlock layer.
        residual_residual_dense_blocks = []
        for _ in range(16):
            residual_residual_dense_blocks += [ResidualInResidualDenseBlock(64, 32, 0.2)]
        self.Trunk_a = nn.Sequential(*residual_residual_dense_blocks)

        # 8 ResidualOfReceptiveFieldDenseBlock layer.
        residual_residual_fields_dense_blocks = []
        for _ in range(8):
            residual_residual_fields_dense_blocks += [
                ResidualOfReceptiveFieldDenseBlock(64, 32, 0.1)]
        self.Trunk_RFB = nn.Sequential(*residual_residual_fields_dense_blocks)

        # Second conv layer post residual field blocks
        self.RFB = ReceptiveFieldBlock(64, 64, 0.1)

        # Sub-pixel convolution layers.
        self.subpixel_conv = nn.Sequential(
            SubpixelConvolutionLayer(64),
            SubpixelConvolutionLayer(64)
        )

        # Next conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer.
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First convolution layer.
        conv1 = self.conv1(x)

        # ResidualInResidualDenseBlock network with 16 layers.
        trunk_a = self.Trunk_a(conv1)
        trunk_rfb = self.Trunk_RFB(trunk_a)
        # First convolution and ResidualOfReceptiveFieldDenseBlock feature image fusion.
        out = torch.add(conv1, trunk_rfb)

        # First ReceptiveFieldBlock layer.
        out = self.RFB(out)

        # Using sub-pixel convolution layer to improve image resolution.
        out = self.subpixel_conv(out)
        # Second convolution layer.
        out = self.conv2(out)
        # Output RGB channel image.
        out = self.conv3(out)

        return out


def _gan(arch: str, pretrained: bool, progress: bool) -> Generator:
    model = Generator()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def rfb_esrgan(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from `<https://arxiv.org/pdf/2005.12597.pdf>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("rfb_esrgan", pretrained, progress)
