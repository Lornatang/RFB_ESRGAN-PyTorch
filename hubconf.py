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
"""File for accessing GAN via PyTorch Hub https://pytorch.org/hub/
Usage:
    import torch
    model = torch.hub.load("Lornatang/RFB_ESRGAN-PyTorch", "rfb", pretrained=True, progress=True, verbose=False)
"""
import torch
from torch.hub import load_state_dict_from_url

from rfb_esrgan_pytorch.models import Generator

model_urls = {
    "rfb": "https://github.com/Lornatang/RFB_ESRGAN-PyTorch/releases/download/0.1.0/RFB_ESRGAN_DF2K-e31a1b2e.pth"
}

dependencies = ["torch"]


def create(arch, pretrained, progress):
    model = Generator()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress,
                                              map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def rfb(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..."<https://arxiv.org/abs/2005.12597>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return create("rfb", pretrained, progress)

