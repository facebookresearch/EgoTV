# Vision Langauge Grounding

Clone the repository
```
$ git clone <>
$ export GENERATE_DATA=$(pwd)/VisionLangaugeGrounding/alfred
$ cd $GENERATE_DATA
```

Install all requirements
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
