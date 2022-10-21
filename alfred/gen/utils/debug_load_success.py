import os
import pandas as pd
from dataset_management_util import load_successes_from_disk, postprocess, remove_img_dir

save_path = os.path.join(os.getcwd(), '../dataset/context_goal_composition')
# succ_traj = pd.DataFrame(columns=["goal", "pickup", "movable", "receptacle", "scene"])
# succ_traj, full_traj = load_successes_from_disk(save_path, 'slice_and_clean_and_place', succ_traj, True, 1)
# postprocess(save_path, 'slice_and_clean_and_place', True, max_count=1)
# succ_traj, full_traj = load_successes_from_disk(save_path, 'slice_and_clean_and_place', succ_traj, True, 1)
remove_img_dir(save_path)
print('done')