# Vision Langauge Grounding

## To set-up the dataset

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
$ cd $ALFRED_ROOT/gen/ff_planner
$ make
```

```
$ cd $GENERATE_DATA/gen
$ python scripts/generate_trajectories.py
```

## Generate Layouts
If you want to generate new layouts (aside from the generated layouts in alfred/gen/layouts/),

```
$ cd $GENERATE_DATA/gen
$ python layouts/precompute_layout_locations.py 
```

## Generate data corresponding to new goals

* Define the goal conditions in alfred/gen/goal_library.py
* Add the list of goals in alfred/gen/constants.py
* Add the goal_variables in alfred/gen/scripts/generate_trajectories.py
* Run the following commands:
```
$ cd $GENERATE_DATA/gen
$ python scripts/generate_trajectories.py
```

