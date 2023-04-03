### traj_data keys
* **dataset_params**
  > video_frame_rate: 5 (dataset was collected with this parameter)
* **images**
  > * frame-by-frame object and action information
  > * high_idx: represents the high-level action / sub-goal being executed in that frame
  > * low_idx: represents the low-level action being executed in that frame
  > * bbox: bounding box information of all objects in the frame 
* **objects_metadata**
  > * contains metadata of each object for initial state and subsequent states after high-level action
  > * = len(traj['plan']['high_pddl']) + 1  
  > *  contains AI2-THOR internal state metadata
  > * [https://ai2thor.allenai.org/ithor/documentation/environment-state/](https://ai2thor.allenai.org/ithor/documentation/environment-state/)
* **pddl_params**
  > * object_target: target object
  > * mrecep_target: movable receptable target
* **pddl_state**
  >* contains all pddl file names generated at the beginning of the episode and after each sub-goal is executed
  >* folder (for each sample): pddl_states/
  >* = len(traj['plan']['high_pddl']) + 1  
* **plan**
  >* high_pddl: step-by-step high-level actions / sub-goals executed to complete the task
  >  * discrete_action: action + args from the dataset generation
  >  * high_idx: index of high-level action (used to mark which low-level actions fall under this sub-goal)
  >  * planner_action: planner action + args
  >    * high_idx: denotes the high-level action / sub-goal which the low-level action falls under 
  >    * state_metadata: records state metadata after each low-level action
  >* low_actions: step-by-step low-level actions executed to complete the task
* **scene**
  > * floor_plan: 'FloorPlan27'
  > * scene_num: 27
* **state_metadata**
  > * contains metadata of each object for initial state and subsequent states after high-level action  
  > * = len(traj['plan']['high_pddl']) + 1  
  > * unlike objects_metadata which contains AI2-THOR internal state metadata, state_metadata contains state info for our proposed dataset
  > * object temp changes to *room-temp* after a while in objects metadata, however, in state metadata, it stays *hot* to satisfy the end goal for data generation
* **task_id**
  > trial id: 'trial_T20220917_235349_019133'
* **task_type**
  > task type for instance: 'heat_then_clean_then_slice'
* **template**
    >* high_descs: step-by-step natural language description of each high-level action / sub-goal # = len(traj['plan']['high_pddl'])
    >* neg: negative hypothesis for the video
    >* pos: positive hypothesis for the video


<details>
  <summary>View of traj_data.json slice</summary>

```json
{
"dataset_params": {
        "video_frame_rate": 5
    },
    "images": [
      {
        "bbox": {
          "Baseboard.022|2.38|0|-1.02": [
            0.0,
            348.0,
            73.0,
            438.0
          ],
          "Bowl|-00.03|+00.76|+01.36": [
            31.0,
            284.0,
            68.0,
            309.0
          ]
        },
      "before": "True",
      "high_idx": 0,
      "image_name": "000000000.png",
      "low_idx": 0
      }
    ],
    "objects_metadata": [
      [
        {"assetId": "",
        "axisAlignedBoundingBox": {
                    "center": {
                        "x": 2.1940131187438965,
                        "y": 1.8911657333374023,
                        "z": 2.4240050315856934
                    },"cornerPoints": [
                        [
                            2.357738494873047,
                            2.297971725463867,
                            2.6044836044311523
                        ],
                        [
                            2.357738494873047,
                            2.297971725463867,
                            2.2435264587402344
                        ],
                        [
                            2.357738494873047,
                            1.4843597412109375,
                            2.6044836044311523
                        ],
                        [
                            2.357738494873047,
                            1.4843597412109375,
                            2.2435264587402344
                        ],
                        [
                            2.030287742614746,
                            2.297971725463867,
                            2.6044836044311523
                        ],
                        [
                            2.030287742614746,
                            2.297971725463867,
                            2.2435264587402344
                        ],
                        [
                            2.030287742614746,
                            1.4843597412109375,
                            2.6044836044311523
                        ],
                        [
                            2.030287742614746,
                            1.4843597412109375,
                            2.2435264587402344
                        ]
                    ],
                    "size": {
                        "x": 0.3274507522583008,
                        "y": 0.8136119842529297,
                        "z": 0.36095714569091797
                    }
        },
        "breakable": false,
                "canBeUsedUp": false,
                "canFillWithLiquid": false,
                "controlledObjects": null,
                "cookable": false,
                "dirtyable": false,
                "distance": 2.9920334815979004,
                "fillLiquid": null,
                "isBroken": false,
                "isColdSource": false,
                "isCooked": false,
                "isDirty": false,
                "isFilledWithLiquid": false,
                "isHeatSource": false,
                "isInteractable": false,
                "isMoving": false,
                "isOpen": false,
                "isPickedUp": false,
                "isSliced": false,
                "isToggled": false,
                "isUsedUp": false,
                "mass": 0.0,
                "moveable": false,
                "name": "Cabinet_0676cbe2",
                "objectId": "Cabinet|+02.04|+02.11|+02.62",
                "objectOrientedBoundingBox": null,
                "objectType": "Cabinet",
                "openable": true,
                "openness": 0.0,
                "parentReceptacles": null,
                "pickupable": false,
                "position": {
                    "x": 2.04144287109375,
                    "y": 2.1134581565856934,
                    "z": 2.618363618850708
                },
                "receptacle": true,
                "receptacleObjectIds": [],
                "rotation": {
                    "x": -0.0,
                    "y": 0.0,
                    "z": 0.0
                },
                "salientMaterials": null,
                "sliceable": false,
                "temperature": "RoomTemp",
                "toggleable": false,
                "visible": false
            },  
      ]
    ],
    "pddl_params": {
        "mrecep_target": "",
        "object_sliced": false,
        "object_target": "Apple",
        "parent_target": "",
        "toggle_target": ""
    },
    "plan": {
      "high_pddl": [{
                "discrete_action": {
                    "action": "GotoLocation",
                    "args": [
                        "sink"
                    ]
                },
                "high_idx": 0,
                "planner_action": {
                    "action": "GotoLocation",
                    "location": "loc|5|8|0|45"
                }
            }],
      "low_actions": [
            {
                "api_action": {
                    "action": "LookDown"
                },
                "discrete_action": {
                    "action": "LookDown_15",
                    "args": {}
                },
                "high_idx": 0,
                "state_metadata": [
                    {
                        "assetId": "",
                        "axisAlignedBoundingBox": {
                            "center": {
                                "x": 2.1940131187438965,
                                "y": 1.8911657333374023,
                                "z": 2.4240050315856934
                            },
                        "cornerPoints": [
                                [
                                    2.357738494873047,
                                    2.297971725463867,
                                    2.6044836044311523
                                ]],
                            "size": {
                                "x": 0.3274507522583008,
                                "y": 0.8136119842529297,
                                "z": 0.36095714569091797
                            }},
                        "cleanable": false,
                        "coolable": false,
                        "heatable": false,
                        "holdsAny": false,
                        "inReceptacle": null,
                        "isClean": false,
                        "isCool": false,
                        "isHot": false,
                        "isOn": false,
                        "isOpen": false,
                        "isSliced": false,
                        "isToggled": false,
                        "moveable": false,
                        "name": "Cabinet_0676cbe2",
                        "objectId": "Cabinet|+02.04|+02.11|+02.62",
                        "objectOrientedBoundingBox": null,
                        "objectType": "Cabinet",
                        "openable": true,
                        "pickupable": false,
                        "position": {
                            "x": 2.04144287109375,
                            "y": 2.1134581565856934,
                            "z": 2.618363618850708
                        },
                        "receptacleAtLocation": [
                            6,
                            8,
                            0,
                            -30
                        ],
                        "receptacleType": "Cabinet",
                        "rotation": {
                            "x": -0.0,
                            "y": 0.0,
                            "z": 0.0
                        },
                        "salientMaterials": null,
                        "sliceable": false,
                        "toggleable": false,
                        "visible": false
                    }
                ]
            }
      ]
    },
    "scene": {
        "dirty_and_empty": false,
        "floor_plan": "FloorPlan27",
        "init_action": {
            "action": "TeleportFull",
            "horizon": 30,
            "rotation": 0,
            "standing": true,
            "x": 1.25,
            "y": 0.9010001420974731,
            "z": 0.0
        },
        "object_poses": [
            {
                "objectName": "Egg_fa86d803",
                "position": {
                    "x": 0.6937957406044006,
                    "y": 0.8302623629570007,
                    "z": 2.606858730316162
                },
                "rotation": {
                    "x": 3.1054576538736e-05,
                    "y": 6.309879840848964e-10,
                    "z": 0.0014264412457123399
                }
            }],
    "state_metadata": [[
                    {
                        "assetId": "",
                        "axisAlignedBoundingBox": {
                            "center": {
                                "x": 2.1940131187438965,
                                "y": 1.8911657333374023,
                                "z": 2.4240050315856934
                            },
                            "cornerPoints": [
                                [
                                    2.357738494873047,
                                    2.297971725463867,
                                    2.6044836044311523
                                ],
                                [
                                    2.357738494873047,
                                    2.297971725463867,
                                    2.2435264587402344
                                ],
                                [
                                    2.357738494873047,
                                    1.4843597412109375,
                                    2.6044836044311523
                                ],
                                [
                                    2.357738494873047,
                                    1.4843597412109375,
                                    2.2435264587402344
                                ],
                                [
                                    2.030287742614746,
                                    2.297971725463867,
                                    2.6044836044311523
                                ],
                                [
                                    2.030287742614746,
                                    2.297971725463867,
                                    2.2435264587402344
                                ],
                                [
                                    2.030287742614746,
                                    1.4843597412109375,
                                    2.6044836044311523
                                ],
                                [
                                    2.030287742614746,
                                    1.4843597412109375,
                                    2.2435264587402344
                                ]
                            ],
                            "size": {
                                "x": 0.3274507522583008,
                                "y": 0.8136119842529297,
                                "z": 0.36095714569091797
                            }
                        },
                        "cleanable": false,
                        "coolable": false,
                        "heatable": false,
                        "holdsAny": false,
                        "inReceptacle": null,
                        "isClean": false,
                        "isCool": false,
                        "isHot": false,
                        "isOn": false,
                        "isOpen": false,
                        "isSliced": false,
                        "isToggled": false,
                        "moveable": false,
                        "name": "Cabinet_0676cbe2",
                        "objectId": "Cabinet|+02.04|+02.11|+02.62",
                        "objectOrientedBoundingBox": null,
                        "objectType": "Cabinet",
                        "openable": true,
                        "pickupable": false,
                        "position": {
                            "x": 2.04144287109375,
                            "y": 2.1134581565856934,
                            "z": 2.618363618850708
                        },
                        "receptacleAtLocation": [
                            6,
                            8,
                            0,
                            -30
                        ],
                        "receptacleType": "Cabinet",
                        "rotation": {
                            "x": -0.0,
                            "y": 0.0,
                            "z": 0.0
                        },
                        "salientMaterials": null,
                        "sliceable": false,
                        "toggleable": false,
                        "visible": false
                    }],
    "task_id": "trial_T20220917_235349_019133",
    "task_type": "heat_then_clean_then_slice",
    "template": {
        "neg": "apple is heated, then sliced, then cleaned in a SinkBasin",
        "pos": "apple is picked, then heated, then cleaned in a SinkBasin, then sliced"
    }
}
```
</details>