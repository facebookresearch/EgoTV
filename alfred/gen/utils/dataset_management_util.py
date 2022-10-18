import os
import shutil
import numpy as np
import matplotlib.pyplot as plt


def postprocess(succ_dir, goal_T, prune_trials, max_count=None, to_delete=None):
    tuple_counts = {}
    queue_for_delete = []
    for root, goals, _ in os.walk(succ_dir):
        for goal in goals:
            if goal != goal_T:
                continue
            # traj_per_goal_dir = os.path.join(root, g)
            for traj_root, trials, _ in os.walk(os.path.join(root, goal)):
                for trial in trials:
                    if trial.count('-') == 3:
                        pickup, movable, receptacle, scene_num = trial.split('-')
                        for file_path, _dirs, _ in os.walk(os.path.join(traj_root, trial)):
                            if len(_dirs) == 0:
                                queue_for_delete.append(file_path)
                            for _d in _dirs:
                                for trial_path, _, _files in os.walk(os.path.join(traj_root, trial, _d)):
                                    if 'video.mp4' in _files:
                                        k = (goal, pickup, movable, receptacle, scene_num)
                                        if k not in tuple_counts:
                                            tuple_counts[k] = 0
                                        tuple_counts[k] += 1
                                        if tuple_counts[k] > max_count:
                                            queue_for_delete.append(trial_path)
                                            tuple_counts[k] -= 1
                                        # deleted_all = False
                                    else:
                                        queue_for_delete.append(trial_path)
                                    break  # only examine top level
                            break  # only examine top level

        break  # only examine top level

    if prune_trials:
        deleted = 0
        for _d in queue_for_delete:
            if deleted >= to_delete:
                break
            print("Removing extra trial '%s'" % _d)
            shutil.rmtree(_d)
            deleted += 1

def load_successes_from_disk(succ_dir, succ_traj, prune_trials, target_count,
                             cap_count=None, min_count=None):
    tuple_counts = {}
    queue_for_delete = []
    for root, goals, _ in os.walk(succ_dir):
        for goal in goals:
            if goal == 'fails':
                continue
            # traj_per_goal_dir = os.path.join(root, g)
            for traj_root, trials, _ in os.walk(os.path.join(root, goal)):
                for trial in trials:
                    if trial.count('-') == 3:
                        pickup, movable, receptacle, scene_num = trial.split('-')
                        # Add an entry for every successful trial folder in the directory.
                        # deleted_all = True

                        for file_path, _dirs, _ in os.walk(os.path.join(traj_root, trial)):
                            # if len(_dirs) == 0:
                            #     print("Removing unfinished trial '%s'" % os.path.join(traj_root, trial, _d))
                            #     shutil.rmtree(os.path.join(traj_root, trial, _d))
                            if len(_dirs) == 0:
                                queue_for_delete.append(file_path)
                            for _d in _dirs:
                                for trial_path, _, _files in os.walk(os.path.join(traj_root, trial, _d)):
                                    if 'video.mp4' in _files:
                                        k = (goal, pickup, movable, receptacle, scene_num)
                                        if k not in tuple_counts:
                                            tuple_counts[k] = 0
                                        tuple_counts[k] += 1
                                        # deleted_all = False
                                    else:
                                        queue_for_delete.append(trial_path)
                                    break  # only examine top level
                            break  # only examine top level

        break  # only examine top level

    if prune_trials:
        for _d in queue_for_delete:
            print("Removing unfinished trial '%s'" % _d)
            shutil.rmtree(_d)
    # Populate dataframe based on tuple constraints.
    for k in tuple_counts:
       if min_count is None or tuple_counts[k] >= min_count:
            to_add = tuple_counts[k] if cap_count is None else cap_count
            for _ in range(to_add):
                succ_traj = succ_traj.append({
                    "goal": k[0],
                    "pickup": k[1],
                    "movable": k[2],
                    "receptacle": k[3],
                    "scene": k[4]}, ignore_index=True)
    tuples_at_target_count = set([t for t in tuple_counts if tuple_counts[t] >= target_count])

    return succ_traj, tuples_at_target_count


def load_fails_from_disk(succ_dir, to_write=None):
    fail_traj = set()
    fail_dir = os.path.join(succ_dir, "fails")
    if not os.path.isdir(fail_dir):
        os.makedirs(fail_dir)
    if to_write is not None:
        for goal, pickup, movable, receptacle, scene_num in to_write:
            with open(os.path.join(fail_dir, '-'.join([goal, pickup, movable, receptacle, scene_num])), 'w') as f:
                f.write("0")
    for root, dirs, files in os.walk(fail_dir):
        for fn in files:
            if fn.count('-') == 4:
                goal, pickup, movable, receptacle, scene_num = fn.split('-')
                fail_traj.add((goal, pickup, movable, receptacle, scene_num))
        break  # only examine top level
    return fail_traj


def plot_dataset_stats(succ_traj):
    # TODO: measure diversity of (["goal", "pickup", "movable", "receptacle", "scene"]) tuples
    plt.figure(figsize=(30, 30))
    succ_traj['pickup'].replace(to_replace='TomatoSliced', value='Tomato', inplace=True)
    succ_traj['pickup'].replace(to_replace='PotatoSliced', value='Potato', inplace=True)
    succ_traj['pickup'].replace(to_replace='LettuceSliced', value='Lettuce', inplace=True)
    succ_traj['pickup'].replace(to_replace='AppleSliced', value='Apple', inplace=True)
    succ_traj['pickup'].replace(to_replace='BreadSliced', value='Bread', inplace=True)
    succ_traj[['goal', 'pickup']].groupby('goal').pickup.value_counts().unstack().plot.barh(stacked=True,
                                                                                            figsize=(10, 12),
                                                                                            rot=45,
                                                                                            colormap='tab20')
    plt.title("goals vs. objects")
    plt.tight_layout()
    plt.show()
    try:
        movable_traj = succ_traj[succ_traj.movable != 'None']
        movable_traj[['goal', 'movable']].groupby('goal').movable.value_counts().unstack().plot.barh(stacked=True,
                                                                                                     figsize=(10, 8),
                                                                                                     rot=45,
                                                                                                     colormap='tab20')
        plt.title("goals vs. movable receptacles")
        plt.tight_layout()
        plt.show()
    except:
        print("no movable objects used")
    recep_traj = succ_traj[succ_traj.receptacle != 'None']
    recep_traj[['goal', 'receptacle']].groupby('goal').receptacle.value_counts().unstack().plot.barh(stacked=True,
                                                                                                     figsize=(10, 10),
                                                                                                     rot=45,
                                                                                                     colormap='tab20')
    plt.title("goals vs. all receptacles")
    plt.tight_layout()
    plt.show()
    succ_traj[['goal', 'scene']].groupby('goal').scene.value_counts().unstack().plot.barh(stacked=True,
                                                                                          figsize=(10, 12),
                                                                                          rot=45,
                                                                                          colormap='tab20')
    plt.title("goals vs. scenes")
    plt.tight_layout()
    plt.show()
    succ_traj['goal'].value_counts().plot(kind='barh', figsize=(10, 12), rot=40)
    plt.title("goal counts")
    plt.tight_layout()
    plt.show()
    succ_traj['pickup'].value_counts().plot(kind='bar', figsize=(10, 12), rot=40)
    max_val = succ_traj['pickup'].value_counts().max()
    plt.yticks(np.arange(0, max_val + 1, int(max_val / 10)))
    plt.title("all object counts")
    plt.tight_layout()
    plt.show()
    recep_traj['receptacle'].value_counts().plot(kind='bar', figsize=(10, 12), rot=40)
    max_val = recep_traj['receptacle'].value_counts().max()
    plt.yticks(np.arange(0, max_val + 1, int(max_val / 10)))
    plt.title("all receptacle counts")
    plt.tight_layout()
    plt.show()
