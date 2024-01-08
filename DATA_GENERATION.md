## Data Generation

Here, we provide instructions to generate your custom dataset on AI2-THOR. We built our dataset generation code on top of [ALFRED dataset [3] repository.](https://github.com/askforalfred/alfred)

[comment]: <> (Get dependencies and compile the planner)

[comment]: <> (```)

[comment]: <> ($ sudo apt-get install ffmpeg flex bison)

[comment]: <> ($ cd $GENERATE_DATA/gen/ff_planner)

[comment]: <> ($ make)

[comment]: <> (```)

Clone repo.
```shell
git clone https://github.com/facebookresearch/EgoTV.git
export GENERATE_DATA=$(pwd)/EgoTV/alfred
cd $GENERATE_DATA
```
Install AI2-THOR requirements.
```shell
conda create -n <venv> python==3.10.0  # substitute with your own venv
source activate <venv>
bash install_requirements.sh
```
Generate dataset.
```shell
cd $GENERATE_DATA/gen
python scripts/generate_trajectories.py --save_path <your save path> --split_type <split_type>

# append the following to generate with multiprocessing for faster generation
# --num_threads <num_threads> --in_parallel 
```
The data is generated in: *save_path*  
Here, split_type can be one of the following [*"train", "novel_tasks", "novel_steps",
                                 "novel_scenes", "abstraction"*]



### Generate Layouts
If you want to generate new layouts (aside from the generated layouts in [alfred/gen/layouts](https://github.com/facebookresearch/EgoTV/tree/main/alfred/gen/layouts)),

```shell
cd $GENERATE_DATA/gen
python layouts/precompute_layout_locations.py 
```

Alternatively, the pre-built layouts can be downloaded from [here](https://www.dropbox.com/s/11cvvvcm4v7c5xg/layouts.zip?dl=0) and saved to the path alfred/gen/layouts/


### Download pddl files for tasks: 
The pddl task files can be downloaded from [here](https://www.dropbox.com/s/yd50ruzqasq6idm/domains.zip?dl=0) and saved to the path alfred/gen/planner/domains/


### Define new goals and generate data corresponding to those goals

* Define the goal conditions in [alfred/gen/goal_library.py](https://github.com/facebookresearch/EgoTV/blob/main/alfred/gen/goal_library.py)
* Add the list of goals in [alfred/gen/constants.py](https://github.com/facebookresearch/EgoTV/blob/main/alfred/gen/constants.py)
* Add the goal_variables in [alfred/gen/scripts/generate_trajectories.py](https://github.com/facebookresearch/EgoTV/blob/main/alfred/gen/scripts/generate_trajectories.py)
* Run the following commands:
```shell
cd $GENERATE_DATA/gen
python scripts/generate_trajectories.py --save_path <your save path>
```

To simply run the fastforward planner on the generated pddl problem
```shell
cd $GENERATE_DATA/gen
ff_planner/ff -o planner/domains/PutTaskExtended_domain.pddl -s 3 -f logs_gen/planner/generated_problems/problem_<num>.pddl
```