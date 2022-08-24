#!/bin/bash
echo "Installing required packages"
pip install --upgrade pip
pip install -r requirements.txt
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

