# EgoTV: Egocentric Task Verification from Natural Language Task Descriptions
******************************************************
Official code of our ICCV 2023 paper.

<p align="center">
    <a href="https://rishihazra.github.io/EgoTV" target="_blank">
        <img src="//img.shields.io/website/https/rishihazra.github.io/EgoTV?down_color=red&down_message=offline&up_message=link">
    </a>
    <a href="https://arxiv.org/abs/2303.16975" target="_blank">
        <img src="//img.shields.io/badge/arXiv-2303.16975-red">
    </a>
    <a href="https://www.dropbox.com/s/6ac5yslze5mct1k/EgoTV.zip?dl=0">
        <img src="//img.shields.io/badge/Downloads-EgoTV Dataset-blue">
    </a>
    <a href="https://www.dropbox.com/s/6ac5yslze5mct1k/EgoTV.zip?dl=0">
        <img src="//img.shields.io/badge/Downloads-CTV Dataset-blue">
    </a>
</p>

<p align="center">
  <img src="nsg-pos.gif" alt="egoTV">
</p>

## Quickstart
Clone repo.
```shell
git clone https://github.com/facebookresearch/EgoTV.git
```
Download EgoTV dataset and set env paths.
```shell
export DATA_ROOT=<path to dataset>
export BASELINES=$(pwd)/EgoTV/baselines
```

Listed below are two ways of setting up the background requirements.

### 1. Using virtual environment
Install all requirements to run baselines. All baseline models are in the filepath: [baselines/all_train](https://github.com/facebookresearch/EgoTV/tree/main/baselines/all_train)
```shell
conda create -n <venv> python==3.10.0  # substitute with your own venv
source activate <venv>
bash $BASELINES/install_baseline_requirements.sh
```

### 2. Using Docker
Alternatively, we provide a [Dockerfile](https://github.com/facebookresearch/EgoTV/blob/main/Dockerfile) for an easier setup. Ensure you have [Docker](https://docs.docker.com/desktop/install/ubuntu/) installed.
```shell
docker pull rishihazra/alfred-dgx:torch-1.11.0
docker run -it rishihazra/alfred-dgx:torch-1.11.0
```

You can also generate your custom dataset. See [DATA_GENERATION.md](https://github.com/facebookresearch/EgoTV/tree/main/DATA_GENERATION.md) for details. 
**************************************************************

## EgoTV file structure
```
dataset/
├── test_splits
│   ├── abstraction
│   ├── novel scenes
│   ├── novel tasks
│   └── novel steps
└── train
|   ├── heat_then_clean_then_slice
|   │   └── Apple-None-None-27
|   │       └── trial_T20220917_235349_019133
|   │           ├── pddl_states
|   │           ├── traj_data.json
|   │           └── video.mp4
```

More details of datafiles can be found in [EgoTV/alfred/README.md](https://github.com/facebookresearch/EgoTV/tree/main/alfred).

## Play with EgoTV samples 
We provide a notebook to download and analyze the EgoTV samples. Refer [ablations/data_analysis.ipynb](https://github.com/facebookresearch/EgoTV/blob/main/ablations/data_analysis.ipynb)


![egoTV](egoTV.png)

**************************************************************

## Baselines

To run baselines. Here <baseline> can be replaced by clip4clip, coca, text2text, videoclip.

```shell
./run_scripts/run_<baseline>_train.sh  # for train
./run_scripts/run_<baseline>_test.sh  # for test
# if data split not preprocessed, specify "--preprocess" in the run instruction
# for attention-based models, specify "--attention" in the run instruction
# to resume training from a previously stored checkpoint, specify "--resume" in the run instruction
```

**1. VIOLIN**
* text encoders: GloVe, (Distil)-BERT uncased [10], CLIP [5]
* visual_encoders: ResNet18, I3D [4], S3D [7], MViT [6], CLIP [5]

Note: To run the I3D and S3D models, download the pretrained model (rgb_imagenet.pt, S3D_kinetics400.pt) from these repositories respectively: 
* [https://github.com/piergiaj/pytorch-i3d/tree/master/models](https://github.com/piergiaj/pytorch-i3d/tree/master/models)
* [https://github.com/kylemin/S3D](https://github.com/kylemin/S3D)
```shell
mkdir $BASELINES/i3d/models
wget -P $BASELINES/i3d/models "https://github.com/piergiaj/pytorch-i3d/tree/master/models/rgb_imagenet.pt" "https://github.com/piergiaj/pytorch-i3d/tree/master/models/rgb_charades.pt"
wget -P $BASELINES/s3d "https://drive.google.com/uc?export=download&id=1HJVDBOQpnTMDVUM3SsXLy0HUkf_wryGO"
```

**2. CoCa**

Download [CoCa model](https://github.com/mlfoundations/open_clip) from OpenCLIP (coca_ViT-B-32 finetuned on mscoco_finetuned_laion2B-s13B-b90k)

**3. VideoCLIP**

The VideoCLIP has conflicting packages with EgoTV, hence we setup a new environment for it.

* create a new conda env since the packages used are different from EgoTV packages
```shell
conda create -n videoclip python=3.8.8
source activate videoclip
```
* clone the repo and run the following installations
```shell
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install -e .  # also optionally follow fairseq README for apex installation for fp16 training.
export MKL_THREADING_LAYER=GNU  # fairseq may need this for numpy
cd examples/MMPT  # MMPT can be in any folder, not necessarily under fairseq/examples.
pip install -e .
pip install transformers==3.4
```
* download the checkpoint using
```
wget -P runs/retri/videoclip/ "https://dl.fbaipublicfiles.com/MMPT/retri/videoclip/checkpoint_best.pt"	
```

### Run proScript

**************************************************************

## NSG Model

Setup proScript. Details of proScript can be found in [baselines/proScript](https://github.com/facebookresearch/EgoTV/tree/main/baselines/proScript). <--output_type 'nl'> for natural language graph output; 
<--output_type 'dsl'> for domain-specific language graph output (default: dsl)
```shell
source activate alfred_env
export DATA_ROOT=<path to dataset>
export BASELINES=$(pwd)/EgoTV/baselines
cd $BASELINES/proScript

# train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train_supervised.py --num_workers 4 --batch_size 32 --preprocess --test_split <> --run_id <> --epochs 20
# test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 test.py --num_workers 4 --batch_size 32 --preprocess --test_split <> --run_id <>
```

Run NSG.
```shell
./run_scripts/run_nsg_train.sh  # for nsg train
./run_scripts/run_nsg_test.sh  # for nsg test
```
**************************************************************

## Citing EgoTV
If you find this codebase helpful for your work, please cite our paper:
```BibTeX
@InProceedings{Hazra_2023_ICCV,
    author    = {Hazra, Rishi and Chen, Brian and Rai, Akshara and Kamra, Nitin and Desai, Ruta},
    title     = {EgoTV: Egocentric Task Verification from Natural Language Task Descriptions},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {15417-15429}
}
```
**************************************************************


## References
[1] Jingzhou Liu, Wenhu Chen, Yu Cheng, Zhe Gan, Licheng Yu, Yiming Yang, Jingjing Liu ["VIOLIN: A Large-Scale Dataset for Video-and-Language Inference"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Violin_A_Large-Scale_Dataset_for_Video-and-Language_Inference_CVPR_2020_paper.pdf). In CVPR 2020  
[2] Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, Aniruddha Kembhavi, Abhinav Gupta, Ali Farhadi ["AI2-THOR: An Interactive 3D Environment for Visual AI"](https://arxiv.org/pdf/1712.05474.pdf)  
[3] Mohit Shridhar,	Jesse Thomason,	Daniel Gordon,	Yonatan Bisk, Winson Han, Roozbeh Mottaghi,	Luke Zettlemoyer, Dieter Fox ["ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks"](https://arxiv.org/abs/1912.01734) In CVPR 2020  
[4] Joao Carreira, Andrew Zisserman ["Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"](https://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf) In CVPR 2017  
[5] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever ["Learning Transferable Visual Models From Natural Language Supervision"](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf) In ICML 2021  
[6] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer ["Multiscale Vision Transformers"](https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_Multiscale_Vision_Transformers_ICCV_2021_paper.pdf) In ICCV 2021  
[7] Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu, Kevin Murphy ["Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification"](https://openaccess.thecvf.com/content_ECCV_2018/papers/Saining_Xie_Rethinking_Spatiotemporal_Feature_ECCV_2018_paper.pdf) In ECCV 2018  
[8] Keisuke Sakaguchi, Chandra Bhagavatula, Ronan Le Bras, Niket Tandon, Peter Clark, Yejin Choi ["proScript: Partially Ordered Scripts Generation"](https://aclanthology.org/2021.findings-emnlp.184/) In Findings of EMNLP 2021  
[9] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu ["Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"](https://jmlr.org/papers/volume21/20-074/20-074.pdf) In JMLR 2020  
[10] Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf ["DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"](https://arxiv.org/abs/1910.01108)  
[11] Jeffrey Pennington, Richard Socher, Christopher Manning ["GloVe: Global Vectors for Word Representation"](https://aclanthology.org/D14-1162/) In EMNLP 2014   
[12] Yu, Jiahui, et al. ["Coca: Contrastive captioners are image-text foundation models."](https://openreview.net/pdf?id=Ee277P3AYC), In Transactions on Machine Learning Research (2022)   
[13] Xu, Hu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. ["Videoclip: Contrastive pre-training for zero-shot video-text understanding."](https://aclanthology.org/2021.emnlp-main.544.pdf) In EMNLP 2021  
[14] Luo, Huaishao, et al. ["CLIP4Clip: An empirical study of CLIP for end to end video clip retrieval and captioning."](https://arxiv.org/pdf/2104.08860.pdf) Neurocomputing (2022)  
[15] Zeng, Andy, et al. ["Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language"](https://openreview.net/pdf?id=G2Q2Mh3avow) ICLR 2023

# License

The majority of EgoTV is licensed under CC-BY-NC, however, portions of the projects are available under separate license terms: Howto100M, I3D and HuggingFace Transformers are licensed under the Apache2.0 license; S3D and CLIP are licensed under the MIT license; CrossTask and MViT are licensed under the BSD-3.
