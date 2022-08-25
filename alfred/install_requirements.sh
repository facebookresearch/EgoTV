#!/bin/bash
echo "Installing required packages"
pip install --upgrade pip
pip install -r requirements.txt
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html

