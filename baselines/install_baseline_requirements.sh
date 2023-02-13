#!/bin/bash
echo "Installing required packages for running baselines"
pip install --upgrade pip
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html
sudo apt install graphviz
pip install -r baseline_requirements.txt
python -c "import nltk; nltk.download('punkt')"