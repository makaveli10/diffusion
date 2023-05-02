#!/bin/bash

apt install wget

pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers
wget -O train_dreambooth.py https://raw.githubusercontent.com/ShivamShrirao/diffusers/main/examples/dreambooth/train_dreambooth.py
git clone https://github.com/makaveli10/diffusion
mv diffusion/* .
rm -rf diffusion

# dirty hack to make bitsandbytes work on cuda
# cp /usr/local/lib/python3.8/dist-packages/bitsandbytes/libbitsandbytes_cuda112.so \
#     /usr/local/lib/python3.8/dist-packages/bitsandbytes/libbitsandbytes_cpu.so