# Task Tracking and Grounding

## To set-up the AI2-THOR environment

### Clone the repository
```
$ git clone https://github.com/rutadesai/VisionLangaugeGrounding.git
$ export GENERATE_DATA=$(pwd)/VisionLangaugeGrounding/alfred
$ cd $GENERATE_DATA
```
We have build the dataset generation code on top of [ALFRED dataset [3] repository.](https://github.com/askforalfred/alfred)

### Install all requirements
```
$ conda create -n <virtual_env> python==3.10.0
$ source activate <virtual_env>
$ bash install_requirements.sh
```


## Data Generation

[comment]: <> (Get dependencies and compile the planner)

[comment]: <> (```)

[comment]: <> ($ sudo apt-get install ffmpeg flex bison)

[comment]: <> ($ cd $GENERATE_DATA/gen/ff_planner)

[comment]: <> ($ make)

[comment]: <> (```)

### Generate dataset
```
$ cd $GENERATE_DATA/gen
$ python scripts/generate_trajectories.py --save_path <your save path> --split_type <split_type>

# append the following to generate with multiprocessing for faster generation
# --num_threads <num_threads> --in_parallel 
```
The data is generated in: *save_path*  
Here, split_type can be one of the following [*"train", "sub_goal_composition", "verb_noun_composition",
                                 "context_goal_composition", "context_verb_noun_composition", "abstraction"*]

### Generate Layouts
If you want to generate new layouts (aside from the generated layouts in [alfred/gen/layouts](https://github.com/rutadesai/VisionLangaugeGrounding/tree/main/alfred/gen/layouts)),

```
$ cd $GENERATE_DATA/gen
$ python layouts/precompute_layout_locations.py 
```

### Define new goals and generate data corresponding to those goals

* Define the goal conditions in [alfred/gen/goal_library.py](https://github.com/rutadesai/VisionLangaugeGrounding/blob/main/alfred/gen/goal_library.py)
* Add the list of goals in [alfred/gen/constants.py](https://github.com/rutadesai/VisionLangaugeGrounding/blob/main/alfred/gen/constants.py)
* Add the goal_variables in [alfred/gen/scripts/generate_trajectories.py](https://github.com/rutadesai/VisionLangaugeGrounding/blob/main/alfred/gen/scripts/generate_trajectories.py)
* Run the following commands:
```
$ cd $GENERATE_DATA/gen
$ python scripts/generate_trajectories.py --save_path <your save path>
```

To simply run the fastforward planner on the generated pddl problem
```
$ cd $GENERATE_DATA/gen
$ ff_planner/ff -o planner/domains/PutTaskExtended_domain.pddl -s 3 -f logs_gen/planner/generated_problems/problem_<num>.pddl
```

### Generated dataset tree
```
dataset/
├── test_splits
│   ├── abstraction
│   ├── context_goal_composition
│   ├── context_verb_noun_composition
│   ├── sub_goal_composition
│   └── verb_noun_composition
└── train
|   ├── heat_then_clean_then_slice
|   │   └── Apple-None-None-27
|   │       └── trial_T20220917_235349_019133
|   │           ├── pddl_states
|   │           ├── traj_data.json
|   │           └── video.mp4
```

### Test Splits

[$GENERATE_DATA/gen/scripts/generate_trajectories.py](https://github.com/rutadesai/VisionLangaugeGrounding/blob/main/alfred/gen/scripts/generate_trajectories.py)  
Note no split (train or test) have overlapping examples.

#### [Scenes: 1-25]
1. **sub-goal composition**:
    >* all tasks not in train

2. **verb-noun composition**: 
    >* heat(egg)
    >* clean(plate)
    >* slice(lettuce)
    >* place(in, shelf)
    
3. **context-verb-noun composition**:
    >* heat(tomato) in scenes 1-5
    >* cool(cup) in scenes 6-10
    >* place(in, coutertop) in scenes 11-15
    >* slice(potato) in scenes 16-20
    >* clean(knife, fork, spoon) in scenes 21-25

4. **abstraction**:
    >* all train tasks with highly abstracted hypothesis ([$GENERATE_DATA/gen/goal_library_abstraction.py](https://github.com/rutadesai/VisionLangaugeGrounding/blob/main/alfred/gen/goal_library_abstraction.py))
    >* for the rest of the splits ([$GENERATE_DATA/gen/goal_library.py](https://github.com/rutadesai/VisionLangaugeGrounding/blob/main/alfred/gen/goal_library.py))
    
#### [Scenes: 26-30]
5. **context-goal composition**: 
    >* all train tasks in scenes in 26-30

## Dataset Stats 
For details: [ablations/data_analysis.ipynb](https://github.com/rutadesai/VisionLangaugeGrounding/blob/main/ablations/data_analysis.ipynb)

* Total hours: 170 hours
  * Train: 112 hours
  * Test: 58 h
* Average video-length > 1 min
* Tasks: 82
* Objects > 130 (with visual variations)
* Pickup objects: 32 (with visual variations)
* Receptacles: 13 (excluding movable receptacles)
* High-level Actions: 7 (GotoLocation, PickupObject, PutObject, SliceObject, CleanObject, HeatObject, CoolObject)
* Low-level Actions: 12 (LookUp, LookDown, MoveAhead, RotateRight, RotateLeft, OpenObject, CloseObject, ToggleObjectOn, ToggleObjectOff, SliceObject, PickupObject, PutObject)
* Average number of high-level actions (sub-goals) per sample: 10
* Average number of low-level actions per sample: > 50
* Scenes: 30 (Kitchens)

For additional details of the collected dataset trajectory, see: [alfred/README.md](https://github.com/rutadesai/VisionLangaugeGrounding/tree/main/alfred)

## Baselines

### Setup Baselines:

```
$ export DATA_ROOT=<path to dataset>
$ export BASELINES=$(pwd)/VisionLangaugeGrounding/baselines
$ cd $BASELINES
$ sudo apt install graphviz
$ pip install -r baseline_requirements.txt # install requirements
```

Also, 
``` python
import nltk
nltk.download('punkt')
```

### Run baselines
* [baselines/end2end](https://github.com/rutadesai/VisionLangaugeGrounding/tree/main/baselines/end2end): for training and testing baseline models
* [baselines/feature_extraction.py](https://github.com/rutadesai/VisionLangaugeGrounding/blob/main/baselines/feature_extraction.py): intialization and feature extractions for text and visual encoders
  * text encoders: GloVe, (Distil)-BERT [10], CLIP [5]
  * visual_encoders: ResNet18, I3D [4], S3D [7], MViT [6], CLIP [5]
```
$ cd $BASELINES/end2end

# train
$ CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --nproc_per_node=$NGPUS train_baseline.py --num_workers $NWorkers --split_type "train" --batch_size 64 --text_feature_extractor <"bert/glove"> --visual_feature_extractor <"i3d/resnet"> --run_id $run_id --sample_rate 3
# if data split not preprocessed, specify "--preprocess" in the previous step
# for attention-based models, specify "--attention" in the previous step

# test (split_type = {"sub_goal_composition", "verb_noun_composition", "context_verb_noun_composition", "context_goal_composition", "abstraction"})
$ CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --nproc_per_node=$NGPUS test_baseline.py --num_workers $NWorkers --split_type <split_type> --batch_size 128 --sample_rate 3 --run_id $run_id --text_feature_extractor <"bert/glove/clip"> --visual_feature_extractor <"i3d/resnet/mvit/clip">
# if data split not preprocessed, specify "--preprocess" in the run instruction
# for attention-based models, specify "--attention" in the run instruction
# to resume training from a previously stored checkpoint, specify "--resume" in the run instruction
```

Note: to run the I3D and S3D models, download the pretrained model (rgb_imagenet.pt, S3D_kinetics400.pt) from these repositories respectively: 
* [https://github.com/piergiaj/pytorch-i3d/tree/master/models](https://github.com/piergiaj/pytorch-i3d/tree/master/models)
* [https://github.com/kylemin/S3D](https://github.com/kylemin/S3D)
```
$ mkdir $BASELINES/i3d/models
$ wget -P $BASELINES/i3d/models "https://github.com/piergiaj/pytorch-i3d/tree/master/models/rgb_imagenet.pt" "https://github.com/piergiaj/pytorch-i3d/tree/master/models/rgb_charades.pt"
$ wget -P $BASELINES/s3d "https://drive.google.com/uc?export=download&id=1HJVDBOQpnTMDVUM3SsXLy0HUkf_wryGO"
```

Alternatively, modify and run from root
```
$ ./run_train.sh
# ./run_test.sh
```

![baselines.png](img.png)

### Run proScript
* for more details, see [baselines/proScript](https://github.com/rutadesai/VisionLangaugeGrounding/tree/main/baselines/proScript)
```
$ source activate alfred_env
$ export DATA_ROOT=<path to dataset>
$ export BASELINES=$(pwd)/VisionLangaugeGrounding/baselines
$ cd $BASELINES/proScript

# train
$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train_supervised.py --num_workers 4 --batch_size 32 --preprocess --test_split <> --run_id <> --epochs 20
# test
$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 test.py --num_workers 4 --batch_size 32 --preprocess --test_split <> --run_id <>
```
<--output_type 'nl'> for natural language graph output; 
<--output_type 'dsl'> for domain-specific language graph output (default: dsl)

## NeSy Model

```
$ source activate alfred_env
$ export DATA_ROOT=/fb-agios-acai-efs/dataset
$ export BASELINES=$(pwd)/VisionLangaugeGrounding/baselines
$ cd VisionLangaugeGrounding/nesy
$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29500 nesy_v1.py --num_workers 2 --split_type 'train' --batch_size 64 --sample_rate 2 --run_id 50 --fp_seg 12
```


### Ablations
[ablations/](https://github.com/rutadesai/VisionLangaugeGrounding/tree/main/ablations)
```
$ source activate alfred_env
$ export DATA_ROOT=/fb-agios-acai-efs/dataset
$ export BASELINES=$(pwd)/VisionLangaugeGrounding/baselines
$ export CKPTS=/fb-agios-acai-efs/rishi/best_model_ckpts
$ cd VisionLangaugeGrounding/ablations
$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 complexity_ordering.py --num_workers 0 --split_type --batch_size 16 --sample_rate 3 --visual_feature_extractor 'mvit' --text_feature_extractor 'bert' --run_id 1
```

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