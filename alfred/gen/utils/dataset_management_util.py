import os
import shutil
import matplotlib.pyplot as plt


def load_successes_from_disk(succ_dir, succ_traj, prune_trials, target_count,
                             cap_count=None, min_count=None):
    tuple_counts = {}
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
                        queue_for_delete = []
                        deleted_all = True
                        for _, _dirs, _ in os.walk(os.path.join(traj_root, trial)):
                            # if len(_dirs) == 0:
                            #     print("Removing unfinished trial '%s'" % os.path.join(traj_root, trial, _d))
                            #     shutil.rmtree(os.path.join(traj_root, trial, _d))
                            for _d in _dirs:
                                for _, _, _files in os.walk(os.path.join(traj_root, trial, _d)):
                                    if 'video.mp4' in _files:
                                        k = (goal, pickup, movable, receptacle, scene_num)
                                        if k not in tuple_counts:
                                            tuple_counts[k] = 0
                                        tuple_counts[k] += 1
                                        deleted_all = False
                                    else:
                                        queue_for_delete.append(_d)
                                    break  # only examine top level
                            break  # only examine top level
                        if prune_trials:
                            # if deleted_all:
                            #     print("Removing trial-less parent dir '%s'" % os.path.join(traj_root, trial))
                            #     shutil.rmtree(os.path.join(traj_root, trial))
                            # else:
                            for _d in queue_for_delete:
                                print("Removing unfinished trial '%s'" % os.path.join(traj_root, trial, _d))
                                shutil.rmtree(os.path.join(traj_root, trial, _d))
        break  # only examine top level

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
    succ_traj[['goal', 'pickup']].groupby('goal').pickup.value_counts().unstack().plot.barh(stacked=True, figsize=(8, 6), rot=45)
    plt.title("goals vs. objects")
    plt.show()
    try:
        movable_traj = succ_traj[succ_traj.movable != 'None']
        movable_traj[['goal', 'movable']].groupby('goal').movable.value_counts().unstack().plot.barh(stacked=True, figsize=(8, 6), rot=45)
        plt.title("goals vs. movable receptacles")
        plt.show()
    except TypeError:
        print("no movable objects used")
    recep_traj = succ_traj[succ_traj.receptacle != 'None']
    recep_traj[['goal', 'receptacle']].groupby('goal').receptacle.value_counts().unstack().plot.barh(stacked=True, figsize=(8, 6), rot=45)
    plt.title("goals vs. all receptacles")
    plt.show()
    succ_traj[['goal', 'scene']].groupby('goal').scene.value_counts().unstack().plot.barh(stacked=True, figsize=(8, 6), rot=45)
    plt.title("goals vs. scenes")
    plt.show()
    succ_traj['goal'].value_counts().plot(kind='bar', figsize=(8, 6), rot=45)
    plt.title("goal counts")
    plt.show()
    succ_traj['pickup'].value_counts().plot(kind='bar', figsize=(8, 6), rot=45)
    plt.title("all object counts")
    plt.show()
    recep_traj['receptacle'].value_counts().plot(kind='bar', figsize=(8, 6), rot=45)
    plt.title("all receptacle counts")
    plt.show()