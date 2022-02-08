import os

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir ../data/DIV8K/original/train --output_dir ../data/DIV8K/RFB_ESRGAN/train --image_size 600 --step 300 --num_workers 10")
os.system("python ./prepare_dataset.py --images_dir ../data/DIV8K/original/valid --output_dir ../data/DIV8K/RFB_ESRGAN/valid --image_size 600 --step 300 --num_workers 10")
