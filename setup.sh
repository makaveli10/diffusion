#!/bin/bash

apt install build-essential -y
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
apt install wget

pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers

# dirty hack to make bitsandbytes work on cuda
cp /opt/conda/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda116.so /opt/conda/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cpu.so

# codeformer setup
git clone https://github.com/sczhou/CodeFormer
cd CodeFormer
pip install -r requirements.txt
python basicsr/setup.py develop
pip install dlib
python scripts/download_pretrained_models.py CodeFormer
