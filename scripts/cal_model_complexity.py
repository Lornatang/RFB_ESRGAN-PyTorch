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
import time

import prettytable as pt
import torch
from thop import profile

import rfb_esrgan_pytorch.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description="Perceptual Extreme Super Resolution Network with Receptive Field Block.")
parser.add_argument("-b", "--batch-size", type=int, default=16,
                    help="When calculating full memory inference speed. (default: 16)")
parser.add_argument("-i", "--image-size", type=int, default=32,
                    help="Image size of low-resolution. (Default: 32).")
parser.add_argument("--gpu", default=None, type=int,
                    help="GPU id to use.")


def inference(arch, cpu_data, cuda_data, args):
    cpu_model = models.__dict__[arch]()

    params = sum(x.numel() for x in cpu_model.parameters()) / 1E6

    # Cal flops and parameters.
    flops = profile(model=cpu_model, inputs=(cpu_data,), verbose=False)[0] / 1E9 * 2

    with torch.no_grad():
        if args.gpu is not None:
            start_time = time.time()
            _ = cpu_model(cpu_data)
            cpu_speed = int(1 / (time.time() - start_time) * args.batch_size)

            cuda_model = cpu_model.cuda(args.gpu)

            start_time = time.time()
            _ = cuda_model(cuda_data)
            cuda_speed = int(1 / (time.time() - start_time) * args.batch_size)

            return params, flops, cpu_speed, cuda_speed
        else:
            start_time = time.time()
            _ = cpu_model(cpu_data)
            cpu_speed = int(1 / (time.time() - start_time) * args.batch_size)

            return params, flops, cpu_speed


def main():
    args = parser.parse_args()
    tb = pt.PrettyTable()

    cpu_data = torch.randn([args.batch_size, 3, args.image_size, args.image_size])
    if args.gpu is not None:
        cuda_data = cpu_data.cuda(args.gpu)
    else:
        cuda_data = None

    print(f"|-------------------------------------------------------------|")
    print(f"|                           Summary                           |")
    tb.field_names = ["Model", "Params", "FLOPs", "CPU Speed", "GPU Speed"]

    for i in range(len(model_names)):
        value = inference(model_names[i], cpu_data, cuda_data, args)
        tb.add_row([f"{model_names[i].center(15)}",
                    f"{value[0]:4.2f} M",
                    f"{value[1]:4.1f} G",
                    f"{value[2]:4d} it/s",
                    f"{value[3]:4d} it/s"])

    print(tb)


if __name__ == "__main__":
    main()
