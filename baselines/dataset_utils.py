import os
import re
import json

import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics


class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.dataset_json_path = os.path.join(data_path, 'filenames.json')
        self.file_tuples = json.load(open(self.dataset_json_path, 'r'))

    def __len__(self):
        return len(self.file_tuples)

    def __getitem__(self, item):
        return self.file_tuples[item]  # each_tuple: (filename, label)


def prepare_dataloader(data, batch_size, num_workers):
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def preprocess_dataset(path, split_type):
    print("==================== Processing {} data ===================".format(split_type))
    all_files = []
    for root, goals, _ in tqdm(os.walk(path)):
        for goal in goals:
            if goal != 'fails':
                for traj_root, trials, _ in os.walk(os.path.join(root, goal)):
                    for trial in trials:
                        for file_path, _dirs, _ in os.walk(os.path.join(traj_root, trial)):
                            for _d in _dirs:
                                for trial_path, _, trial_files in os.walk(os.path.join(file_path, _d)):
                                    assert 'video.mp4' in trial_files, ("No video file in the directory: " +
                                                                        str(trial_path))
                                    all_files.append((trial_path, '1'))
                                    all_files.append((trial_path, '0'))
                                    break  # only examine top level
                            break  # only examine top level
                    break  # only examine top level
        break  # only examine top level
    print("Processed {} samples in the {} split".format(len(all_files), split_type))
    json.dump(all_files, open(os.path.join(path, 'filenames.json'), 'w'))


def transform_image(video_frame, type='rgb'):
    if type == 'rgb':
        transform =  transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif type == 'flow':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transform(video_frame)


def dictfilt(x, y):
    return dict([(i, x[i]) for i in x if i in set(y)])

def tokenize_and_pad(sent_batch, tokenizer, feature_extractor_type):
    if feature_extractor_type == 'bert':
        return tokenizer(sent_batch, padding=True, return_tensors='pt', return_length=True)
    else:
        return [tokenizer(sent) for sent in sent_batch]


def round_val(values, decimal_places):
    round_values = []
    for val in values:
        round_values.append(round(val, decimal_places))
    return round_values


def calc_metrics(true_labels, pred_labels):
    confusion_matrix = metrics.confusion_matrix(true_labels, pred_labels)
    precision = metrics.precision_score(true_labels, pred_labels)
    recall = metrics.recall_score(true_labels, pred_labels)
    f1_score = metrics.f1_score(true_labels, pred_labels)
    train_acc = metrics.accuracy_score(true_labels, pred_labels)
    return confusion_matrix, round_val([precision, recall, f1_score, train_acc], 3)

def clean_str(string):
    """ Tokenization/string cleaning for strings.
       Taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
       """
    string = re.sub(r"[^A-Za-z0-9(),!?:.\'`]", " ", string)  # <> are added after the cleaning process
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)  # split as two words
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.\.\.", " . ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
