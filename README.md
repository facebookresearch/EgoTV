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
$ pip install --upgrade pip
$ pip install -r requirements.txt
```


## Data Generation

Get dependencies and compile the planner
```
$ sudo apt-get install ffmpeg flex bison
$ cd $GENERATE_DATA/gen/ff_planner
$ make
```

Now, generate dataset
```
$ cd $GENERATE_DATA/gen
$ python scripts/generate_trajectories.py
```

The data is generated in: alfred/gen/dataset/

## Generate Layouts
If you want to generate new layouts (aside from the generated layouts in alfred/gen/layouts/),

```
$ cd $GENERATE_DATA/gen
$ python layouts/precompute_layout_locations.py 
```

## Define new goals and generate data corresponding to those goals

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

