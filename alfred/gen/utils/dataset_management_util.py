import os
import json
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
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


def remove_img_dir(succ_dir):
    # cleaning dataset: removing image directories
    queue_for_delete = []
    for root, goals, _ in os.walk(succ_dir):
        for goal in goals:
            for traj_root, trials, _ in os.walk(os.path.join(root, goal)):
                for trial in trials:
                    if trial.count('-') == 3:
                        for file_path, _dirs, _ in os.walk(os.path.join(traj_root, trial)):
                            for _d in _dirs:
                                for trial_path, list_files, _files in os.walk(os.path.join(traj_root, trial, _d)):
                                    if 'raw_images' in list_files:
                                        img_dir_path = os.path.join(trial_path, 'raw_images')
                                        queue_for_delete.append(img_dir_path)
                                    break  # only examine top level
                            break  # only examine top level
        break  # only examine top level

    for _d in queue_for_delete:
        print("Removing img_dir '%s'" % _d)
        shutil.rmtree(_d)


def analyze_trajectories(succ_dir):
    high_actions, low_actions = [], []
    vid_lens = []
    complexity, ordering = [], []
    for root, goals, _ in os.walk(succ_dir):
        for goal in goals:
            if goal == 'fails':
                continue
            for traj_root, trials, _ in os.walk(os.path.join(root, goal)):
                for trial in trials:
                    if trial.count('-') == 3:
                        for file_path, _dirs, _ in os.walk(os.path.join(traj_root, trial)):
                            for _d in _dirs:
                                for trial_path, _, _files in os.walk(os.path.join(traj_root, trial, _d)):
                                    traj_filename_ind = _files.index('traj_data.json')
                                    traj = json.load(open(trial_path + '/' + _files[traj_filename_ind], 'r'))
                                    task_type = traj['task_type']
                                    complexity.append(len(task_type.replace('then', 'and').split('_and_')))
                                    ordering.append(task_type.count('then'))
                                    high_actions.extend([x['discrete_action']['action']
                                                         for x in traj['plan']['high_pddl']])
                                    low_actions.extend([x['api_action']['action'] for x in traj['plan']['low_actions']])
                                    vid_lens.append(round(len(traj['images']) / (60*5), 2))  # fps = 5
                                    break  # only examine top level
                            break  # only examine top level
        break  # only examine top level
    return Counter(high_actions), Counter(low_actions), vid_lens, complexity, ordering


def object_counter(succ_dir):
    unique_objs = set()
    for root, goals, _ in os.walk(succ_dir):
        for goal in goals:
            if goal == 'fails':
                continue
            for traj_root, trials, _ in os.walk(os.path.join(root, goal)):
                for trial in trials:
                    if trial.count('-') == 3:
                        for file_path, _dirs, _ in os.walk(os.path.join(traj_root, trial)):
                            for _d in _dirs:
                                for trial_path, _, _files in os.walk(os.path.join(traj_root, trial, _d)):
                                    traj_filename_ind = _files.index('traj_data.json')
                                    traj = json.load(open(trial_path + '/' + _files[traj_filename_ind], 'r'))
                                    for x in traj['images']:
                                        for y in x['bbox']:
                                           unique_objs.add(y.split('|')[0].split('.')[0].lower())
                                    break  # only examine top level
                            break  # only examine top level
        break  # only examine top level
    return unique_objs


def load_successes_from_disk(succ_dir, succ_traj, prune_trials, target_count,
                             cap_count=None, min_count=None):
    tuple_counts = {}
    queue_for_delete = []
    high_actions, low_actions = [], []
    sample_high_actions, sample_low_actions = [], []
    vid_lens = []
    complexity, ordering = [], []
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
                                    traj_filename_ind = _files.index('traj_data.json')
                                    traj = json.load(open(trial_path + '/' + _files[traj_filename_ind], 'r'))
                                    high_actions.extend([x['discrete_action']['action']
                                                         for x in traj['plan']['high_pddl']])
                                    sample_high_actions.append(len(traj['plan']['high_pddl']))
                                    sample_low_actions.append(len(traj['plan']['low_actions']))
                                    low_actions.extend([x['api_action']['action'] for x in traj['plan']['low_actions']])
                                    vid_lens.append(round(len(traj['images']) / (60*5), 2))  # fps = 5
                                    task_type = traj['task_type']
                                    complexity.append(len(task_type.replace('then', 'and').split('_and_')))
                                    ordering.append(task_type.count('then'))
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

    return succ_traj, tuples_at_target_count, Counter(high_actions), \
           Counter(low_actions), vid_lens, complexity, ordering, sample_high_actions, sample_low_actions


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


def analyze_videos(vid_lens):
    # plot mean, min and max of each split
    # plot distribution of videos over length
    # compare all splits over mean, min, max
    # count total hours of dataset
    vid_lens = pd.DataFrame({'video-lengths (minutes)': vid_lens})
    fig2 = plt.figure(figsize=(6, 6))
    plt.rcParams['font.size'] = 12
    sns.set_style('white')
    sns.set_context("paper", font_scale=2)
    sns.displot(data=vid_lens, x='video-lengths (minutes)', kind="hist", bins=100, aspect=1.5)
    plt.title("video length distribution", fontsize=15)
    plt.tight_layout()
    plt.show()
    plt.close(fig2)


def plot_dataset_stats(succ_traj, action_counts, vid_lens):
    # TODO: measure diversity of (["goal", "pickup", "movable", "receptacle", "scene"]) tuples
    fig = plt.figure(figsize=(30, 30))
    plt.rcParams['font.size'] = 12
    plt.rc('legend', fontsize=12)
    succ_traj['pickup'].replace(to_replace='TomatoSliced', value='Tomato', inplace=True)
    succ_traj['pickup'].replace(to_replace='PotatoSliced', value='Potato', inplace=True)
    succ_traj['pickup'].replace(to_replace='LettuceSliced', value='Lettuce', inplace=True)
    succ_traj['pickup'].replace(to_replace='AppleSliced', value='Apple', inplace=True)
    succ_traj['pickup'].replace(to_replace='BreadSliced', value='Bread', inplace=True)

    succ_traj[['goal', 'pickup']].groupby('goal').pickup.value_counts().unstack().plot.barh(stacked=True,
                                                                                            figsize=(10, 15),
                                                                                            fontsize=12,
                                                                                            rot=45,
                                                                                            colormap='tab20')
    plt.title("goals vs. objects", fontsize=15)
    plt.tight_layout()
    plt.show()

    try:
        movable_traj = succ_traj[succ_traj.movable != 'None']
        movable_traj[['goal', 'movable']].groupby('goal').movable.value_counts().unstack().plot.barh(stacked=True,
                                                                                                     figsize=(10, 8),
                                                                                                     fontsize=12,
                                                                                                     rot=45,
                                                                                                     colormap='tab20')
        plt.title("goals vs. movable receptacles", fontsize=15)
        plt.tight_layout()
        plt.show()
    except:
        print("no movable objects used")

    recep_traj = succ_traj[succ_traj.receptacle != 'None']
    recep_traj[['goal', 'receptacle']].groupby('goal').receptacle.value_counts().unstack().plot.barh(stacked=True,
                                                                                                     figsize=(10, 10),
                                                                                                     fontsize=12,
                                                                                                     rot=45,
                                                                                                     colormap='tab20')
    plt.title("goals vs. all receptacles")
    plt.tight_layout()
    plt.show()

    succ_traj[['goal', 'scene']].groupby('goal').scene.value_counts().unstack().plot.barh(stacked=True,
                                                                                          figsize=(10, 15),
                                                                                          fontsize=12,
                                                                                          rot=45,
                                                                                          colormap='tab20')
    plt.title("goals vs. scenes", fontsize=15)
    plt.tight_layout()
    plt.show()

    succ_traj['goal'].value_counts().plot(kind='barh', figsize=(10, 15), fontsize=12, rot=40)
    plt.title("goal counts", fontsize=12)
    plt.tight_layout()
    plt.show()

    succ_traj['pickup'].value_counts().plot(kind='bar', figsize=(16, 6), fontsize=12, rot=40)
    max_val = succ_traj['pickup'].value_counts().max()
    plt.yticks(np.arange(0, max_val + 1, int(max_val / 10)), fontsize=12)
    plt.title("all object counts", fontsize=15)
    plt.tight_layout()
    plt.show()

    recep_traj['receptacle'].value_counts().plot(kind='bar', figsize=(8, 8), fontsize=12, rot=40)
    max_val = recep_traj['receptacle'].value_counts().max()
    plt.yticks(np.arange(0, max_val + 1, int(max_val / 10)), fontsize=12)
    plt.title("all receptacle counts", fontsize=15)
    plt.tight_layout()
    plt.show()

    high_action_count, low_action_count = action_counts
    plt.figure(figsize=(8, 6))
    plt.bar(high_action_count.keys(), high_action_count.values())
    plt.xticks(rotation=30, ha='right', fontsize=12)
    # max_val = np.array(high_action_count.values()).max()
    # plt.yticks(np.arange(0, max_val + 1, int(max_val / 10)))
    plt.title("high-level action counts", fontsize=15)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.bar(low_action_count.keys(), low_action_count.values())
    plt.xticks(rotation=30, ha='right', fontsize=12)
    # max_val = np.array(low_action_count.values()).max()
    # plt.yticks(np.arange(0, max_val + 1, int(max_val / 10)))
    plt.title("low-level action counts", fontsize=15)
    plt.tight_layout()
    plt.show()

    analyze_videos(vid_lens)
    plt.close(fig)
    plt.rc('legend', fontsize=12)
