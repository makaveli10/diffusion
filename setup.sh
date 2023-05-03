#!/bin/bash

apt install build-essential -y
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
apt install wget

pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers
# git clone https://github.com/makaveli10/diffusion
# mv diffusion/* .
# rm -rf diffusion

# dirty hack to make bitsandbytes work on cuda
# cp /usr/local/lib/python3.8/dist-packages/bitsandbytes/libbitsandbytes_cuda112.so \
#     /usr/local/lib/python3.8/dist-packages/bitsandbytes/libbitsandbytes_cpu.so

# codeformer setup
git clone https://github.com/sczhou/CodeFormer
cd CodeFormer
pip install -r requirements.txt
python basicsr/setup.py develop
pip install dlib
python scripts/download_pretrained_models.py CodeFormer
