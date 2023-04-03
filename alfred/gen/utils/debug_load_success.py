# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
from dataset_management_util import analyze_trajectories, analyze_videos

save_path = os.path.join(os.getcwd(), '../dataset/context_goal_composition')
# succ_traj = pd.DataFrame(columns=["goal", "pickup", "movable", "receptacle", "scene"])
# succ_traj, full_traj = load_successes_from_disk(save_path, 'slice_and_clean_and_place', succ_traj, True, 1)
# postprocess(save_path, 'slice_and_clean_and_place', True, max_count=1)
# succ_traj, full_traj = load_successes_from_disk(save_path, 'slice_and_clean_and_place', succ_traj, True, 1)
high_action_count, low_action_count, vid_lens = \
    analyze_trajectories('../dataset/context_goal_composition')

save_path = '/fb-agios-acai-efs/dataset/train'
# succ_traj, _ = load_successes_from_disk(save_path, succ_traj, False, 1)
# for goal in set(succ_traj.goal):
#     count = len(succ_traj[succ_traj.goal == goal])
#     if count > 100:
#         delete = count - 100
#         postprocess(save_path, goal, True, max_count=0, to_delete=delete)
