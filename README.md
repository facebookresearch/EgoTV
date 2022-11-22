# Vision Language Grounding

## To set-up the AI2-THOR environment

### Clone the repository
```
$ git clone https://github.com/rutadesai/VisionLangaugeGrounding.git
$ export GENERATE_DATA=$(pwd)/VisionLangaugeGrounding/alfred
$ cd $GENERATE_DATA
```

### Install all requirements
```
$ conda create -n <virtual_env> python
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

Generate dataset
```
$ cd $GENERATE_DATA/gen
$ python scripts/generate_trajectories.py
```

The data is generated in: alfred/gen/dataset/

### Generate Layouts
If you want to generate new layouts (aside from the generated layouts in alfred/gen/layouts/),

```
$ cd $GENERATE_DATA/gen
$ python layouts/precompute_layout_locations.py 
```

### Define new goals and generate data corresponding to those goals

* Define the goal conditions in alfred/gen/goal_library.py
* Add the list of goals in alfred/gen/constants.py
* Add the goal_variables in alfred/gen/scripts/generate_trajectories.py
* Run the following commands:
```
$ cd $GENERATE_DATA/gen
$ python scripts/generate_trajectories.py
```

To simply run the fastforward planner on the generated pddl problem
```
$ cd $GENERATE_DATA/gen
$ ff_planner/ff -o planner/domains/PutTaskExtended_domain.pddl -s 3 -f logs_gen/planner/generated_problems/problem_<num>.pddl
```

Generated dataset tree
```
dataset/
├── test_splits
│   ├── abstraction
│   ├── context_goal_composition
│   ├── context_verb_noun_composition
│   ├── csv_files
│   ├── sub_goal_composition
│   └── verb_noun_composition
└── train
|   ├── heat_then_clean_then_slice
|   │   └── Apple-None-None-27
|   │       └── trial_T20220917_235349_019133
|   │           ├── pddl_states
|   │           ├── raw_images
|   │           ├── traj_data.json
|   │           └── video.mp4
```

View of traj_data.json
```

```

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

```
$ cd $BASELINES/end2end/violin

# train
$ CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --nproc_per_node=$NGPUS violin_baseline.py --num_workers $NWorkers --split_type "train" --batch_size 64 --text_feature_extractor <"bert/glove"> --visual_feature_extractor <"i3d/resnet"> --run_id $run_id --sample_rate 3
# if data split not preprocessed, specify "--preprocess" in the previous step
# for attention-based models, specify "--attention" in the previous step

# test (split_type = {"sub_goal_composition", "verb_noun_composition", "context_verb_noun_composition", "context_goal_composition", "abstraction"})
$ CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --nproc_per_node=$NGPUS test_baseline.py --num_workers $NWorkers --split_type <split_type> --batch_size 128 --sample_rate 3 --run_id $run_id --text_feature_extractor <"bert/glove"> --visual_feature_extractor <"i3d/resnet">
# if data split not preprocessed, specify "--preprocess" in the previous step
# for attention-based models, specify "--attention" in the previous step
```

Alternatively, modify and run from root
```
$ ./run_train.sh
# ./run_test.sh
```