# calculate mean frames per sub-goal
import os
import sys
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from dataset_utils import *
import json
from operator import itemgetter
import statistics
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data import random_split


def iterate(dataloader):
    mean_counts = []
    min_counts = []
    for data_batch, _ in tqdm(dataloader):
        for filepath in data_batch:
            traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))
            count = Counter([x['high_idx'] for ind, x in enumerate(traj['images']) if ind % sr == 0])
            mean_count = statistics.mean(count.values())
            _, min_count = min(count.items(), key=itemgetter(1))
            # print(min_count)
            mean_counts.append(mean_count)
            min_counts.append(min_count)
    return np.array(mean_counts).mean(), np.array(min_counts).mean()


if __name__ == '__main__':
    sr = 1  # sample_rate
    path = os.path.join(os.environ['DATA_ROOT'], 'train')
    dataset = CustomDataset(data_path=path)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_set = random_split(dataset, [train_size, val_size])
    val_loader =  DataLoader(val_set, batch_size=256, shuffle=False, pin_memory=True)
    mean_c, min_c = iterate(val_loader)
    print('Mean Count: {} | Min Count: {}'.format(mean_c, min_c))
    # sr = 2, mean_count = 22, min_count = 7.18
    # sr = 1, mean_count =
