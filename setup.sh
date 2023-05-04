#!/bin/bash

apt install build-essential -y
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
apt install wget

pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers

# codeformer setup
git clone https://github.com/sczhou/CodeFormer
cd CodeFormer
pip install -r requirements.txt
python basicsr/setup.py develop
# pip install dlib
python scripts/download_pretrained_models.py facelib
cd ..
python CodeFormer/scripts/download_pretrained_models.py CodeFormer
