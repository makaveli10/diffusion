#!/bin/bash

# dirty hack to make bitsandbytes work on cuda
export cuda_ver=cuda"$(python -c 'import torch; print(torch.version.cuda)' | sed -e 's/\.//g')"
export python_version="$(python --version | sed 's/ //g' | tr '[:upper:]' '[:lower:]')"
export python_ver=${python_version:0:9}
cp /opt/conda/lib/$python_ver/site-packages/bitsandbytes/libbitsandbytes_$cuda_ver.so /opt/conda/lib/$python_ver/site-packages/bitsandbytes/libbitsandbytes_cpu.so

jupyter notebook --allow-root --port=8888 --no-browser --ip=0.0.0.0