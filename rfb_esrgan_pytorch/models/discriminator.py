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


# Source code reference from `https://github.com/xinntao/BasicSR/blob/master/basicsr/models/archs/discriminator_arch.py`.
class DiscriminatorForVGG(nn.Module):
    def __init__(self, image_size: int) -> None:
        super(DiscriminatorForVGG, self).__init__()

        feature_map_size = image_size // 32

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # input is (3) x 512 x 512
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 64, 3, 2, 1, bias=False),  # state size. (64) x 256 x 256
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 128, 3, 2, 1, bias=False),  # state size. (128) x 128 x 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 256, 3, 2, 1, bias=False),  # state size. (256) x 64 x 64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, 3, 2, 1, bias=False),  # state size. (512) x 32 x 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, 3, 2, 1, bias=False),  # state size. (512) x 16 x 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * feature_map_size * feature_map_size, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


def discriminator_for_vgg(image_size: int) -> DiscriminatorForVGG:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/2005.12597>` paper.
    """
    model = DiscriminatorForVGG(image_size)
    return model
