import os
import re
import json
import cv2
import random
import torch
from sklearn import metrics
from collections import Counter
from operator import itemgetter
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


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
    save_file = os.path.join(path, 'filenames.json')
    if os.path.isfile(save_file):
        return
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
    json.dump(all_files, open(save_file, 'w'))


def transform_image(video_frame, type='rgb'):
    if type == 'rgb':
        transform = transforms.Compose([
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
    elif type == 'mvit':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ])
    return transform(video_frame)


def sample_vid(filename, sample_rate=1):
    video_frames = []
    video = cv2.VideoCapture(os.path.join(filename, 'video.mp4'))
    # print(video.get(cv2.CAP_PROP_FPS))
    success = video.grab()
    fno = 0
    while success:
        if fno % sample_rate == 0:
            _, img = video.retrieve()
            video_frames.append(transform_image(img))
        success = video.grab()
        fno += 1
    return video_frames


def plot_bb(img, x1, y1, x2, y2):
    plt.imshow(img)
    ax = plt.gca()
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


def sample_vid_with_roi(filename, sample_rate, bboxes):
    # TODO: generalize to bbox of multiple objects
    video_frames, roi = [], []
    video = cv2.VideoCapture(os.path.join(filename, 'video.mp4'))
    # print(video.get(cv2.CAP_PROP_FPS))
    success = video.grab()
    fno = 0
    while success:
        if fno % sample_rate == 0:
            _, img = video.retrieve()
            video_frames.append(transform_image(img))
            # TODO: see if the alignment is accurate
            bbox = bboxes[fno]
            # if bbox is not None:
            roi_box = torch.zeros(3, 3, 224, 224)
            try:
                for box_ind, box in enumerate(bbox):
                    # TODO: generalize to multiple bboxes and not just [0]
                    x1, y1, x2, y2 = list(map(lambda x: int(x), box))[:]
                    roi_box[box_ind] = transform_image(img[y1:y2, x1:x2, :])
                    # Want to test if the slicing is correct? go ahead and uncomment the next line
                    # plot_bb(img, x1, y1, x2, y2)
            except:
                pass
            roi.append(roi_box)
        success = video.grab()
        fno += 1
        if fno >= len(bboxes):
            # print('break executed, fno={}'.format(fno))
            break
    assert len(video_frames) == len(roi)
    return video_frames, roi


def extract_segment_labels(trajectory, sample_rate, frames_per_segment, action_args, positive=False, supervised=True):
    """
    used for supervised NeSy model
    returns:
    actions_labs: (dominant) action labels for each segment
    segment_args: dominant action arguments for each segment
    roi_bb: RoI bounding box coordinates corresponding to action arguments
    """
    action_labs = []
    roi_bb = []
    count_labs = []
    segment_args = []

    # extract label, bbox per frame based on sample_rate
    for ind, x in enumerate(trajectory['images']):
        if ind % sample_rate == 0:
            count_labs.append(x['high_idx'])
        bb_dict = x['bbox']
        bb = []
        for action_arg in action_args:
            for obj_id, bbox in bb_dict.items():
                if action_arg.lower() in obj_id.lower():
                    bb.append(bbox)
        if len(bb) != 0:
            # TODO: selecting a fixed number of objects may lead to discarding useful info
            roi_bb.append(bb[:3])
        else:
            roi_bb.append(None)
    if not supervised:
        return roi_bb
    # assert len(roi_bb) == len(count_labs)
    # getting dominant action label in each video segment
    # (each segment has #frames = frames_per_segment)
    for split_ind in range(0, len(count_labs), frames_per_segment):
        if split_ind >= len(count_labs) - frames_per_segment:
            max_lab = max(Counter(count_labs[split_ind:len(count_labs)]).items(), key=itemgetter(1))[0]
        else:
            max_lab = max(Counter(count_labs[split_ind:split_ind + frames_per_segment]).items(), key=itemgetter(1))[0]
        if not positive:
            action = random.sample(['HeatObject', 'SliceObject', 'CoolObject',
                                    'CleanObject', 'PutObject', 'PickupObject', 'GotoLocation'], 1)[0]
        else:
            discrete_action = trajectory['plan']['high_pddl'][max_lab]['discrete_action']
            # TODO: consider ['args'][1] for receptacle for putObject
            action = discrete_action['action']
        segment_args.append(action + ' ' + ' '.join(action_args))
        action_labs.append(action)

    return action_labs, roi_bb, segment_args


def retrieve_query_args(graph_batch):
    """
    retrieves query arguments which is then
    used to get bounding boxes for each frame
    """
    args_batch = []
    for graph in graph_batch:
        nodes = [node for node in graph.nodes]
        args_graph = set()
        for node in nodes:
            node = re.sub('Step \d+ ', '', node)
            node = re.sub(r"[()]", " ", node).strip().split(" ")
            query_type, pred_arg = node[0], ','.join(node[1:])
            if query_type == 'StateQuery':
                args_graph.update([pred_arg.split(',')[0]])
                # args_graph.update(pred_arg.split(',')[:2])
            elif query_type == 'RelationQuery':
                args_graph.update(pred_arg.split(',')[:2])
        args_graph = list(args_graph)
        for ind, arg in enumerate(args_graph):
            args_graph[ind] = arg.replace('sliced', '')
        args_batch.append(list(args_graph))
    return args_batch


def action_mapping(action):
    action_map = {'HeatObject': 0, 'SliceObject': 1,
                  'CoolObject': 2, 'CleanObject': 3,
                  'PutObject': 4, 'PickupObject': 5,
                  'GotoLocation': 6, 'NoOp': 6}
    return action_map[action]


def dictfilt(x, y):
    return dict([(i, x[i]) for i in x if i in set(y)])


def train_log_process(filepath):
    lines = open(filepath, 'r').readlines()[1:]
    # Epoch: 1 | Train Acc: 0.5652528405189514 | Val Acc: 0.6091205477714539 | Train F1: 0.5888705253601074 | Val F1: 0.4636015295982361
    val_acc, val_f1 = 0, 0
    for line in lines:
        split_line = line.split(' | ')
        val_acc = max(val_acc, float(split_line[2].split('Val Acc:')[1]))
        val_f1 = max(val_f1, float(split_line[4].split('Val F1:')[1]))
    return val_acc, val_f1

def task_axis_stats(task_type):
    """
    return ordering and complexity of each task
    heat_simple: complexity = 1, ordering = 0
    heat_and_place: complexity = 2, ordering = 0
    heat_and_slice_and_place: complexity = 3, ordering = 0
    heat_then_clean_and_place : complexity = 3, ordering = 1
    heat_then_clean_then_place: complexity = 3, ordering = 2
    """
    complexity = len(task_type.replace('then', 'and').split('_and_'))
    ordering = task_type.count('then')
    return complexity, ordering


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


def translate(step):
    node, segment_ind = step
    node = re.sub('Step \d+ ', '', node)
    node = re.sub(r"[()]", " ", node).strip().split(" ")
    query, pred_args = node[0], ','.join(node[1:])
    split_text = [pred_args.split(',')[0], pred_args.split(',')[-1]]
    arg_translate = {'heat': 'HeatObject', 'cool': 'CoolObject', 'slice': 'SliceObject', 'clean': 'CleanObject',
                     'place': 'PutObject', 'pick': 'PickupObject'}
    return arg_translate[split_text[-1]], query, segment_ind, split_text[-1]

def check_alignment(pred_alignment:List[List[Tuple]], segment_labels:List[List], ent_labels):
    """
    checks if the predicted (dynamic programming-based) alignment is correct
    for positively entailed hypotheses
    """
    state_pred_dict = {'heat': [], 'cool': [], 'clean': []}
    state_pred_labs, state_true_labs = [torch.tensor(1.)], [torch.tensor(1)]
    relation_pred_labs, relation_true_labs = [torch.tensor(1.)], [torch.tensor(1)]
    for ind, (pred, true) in enumerate(zip(pred_alignment, segment_labels)):
        try:
            if ent_labels[ind].item() == 1:
                for step in pred:
                    pred_action, query, segment_ind, sub_goal = translate(step)
                    if query == 'StateQuery':
                        state_pred_labs.append(torch.tensor(1.))
                        if true[segment_ind] == pred_action:
                            state_true_labs.append(torch.tensor(1))
                            state_pred_dict[sub_goal].append(1)
                        else:
                            state_true_labs.append(torch.tensor(0))
                            state_pred_dict[sub_goal].append(0)
                    else:  # query == 'RelationQuery'
                        relation_pred_labs.append(torch.tensor(1))
                        # try:
                        if true[segment_ind] == pred_action:
                            relation_true_labs.append(torch.tensor(1))
                        else:
                            relation_true_labs.append(torch.tensor(0))
        except KeyError:
            continue
    return torch.stack(state_pred_labs), torch.stack(state_true_labs), \
           torch.stack(relation_pred_labs), torch.stack(relation_true_labs), state_pred_dict


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
