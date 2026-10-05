#!/usr/bin/env bash
set -euo pipefail
cd /data/matengyu/geyugong/mindhier-reproduction
export PIP_CACHE_DIR="$PWD/.cache/pip"
export HF_HOME="$PWD/.cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONDONTWRITEBYTECODE=1
/data/matengyu/geyugong/neuroadapter-subject1-research/envs/neuroadapter/bin/python -m venv --system-site-packages .envs/mindhier
.envs/mindhier/bin/python -m pip install 'numpy==1.26.4' 'transformers==4.44.2' 'huggingface-hub==0.36.2' 'webdataset==0.2.111' 'accelerate==1.10.1' 'tensorboard==2.20.0' 'typed-argument-parser==1.11.0' 'safetensors==0.6.2' 'ftfy==6.3.1' 'einops==0.8.1'
.envs/mindhier/bin/python -m pip freeze > manifests/environment-freeze.txt
.envs/mindhier/bin/python -m pip check > runs/bootstrap/pip-check.txt || true
.envs/mindhier/bin/python -c 'import torch,torchvision,transformers,numpy,h5py,webdataset; print(torch.__version__,torchvision.__version__,transformers.__version__,numpy.__version__)'
printf 'success\n' > runs/bootstrap/environment.complete
