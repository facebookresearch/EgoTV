# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# measure accuracy and F1 scores over different test splits, run_ids and along different axes of complexity and ordering
# also calculates mean test results over different runs
import os
import sys

sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
sys.path.append(os.path.join(os.environ['BASELINES'], 'all_train'))
sys.path.append(os.environ['CKPTS'])
from dataset_utils import *
from feature_extraction import *
from end2end.base import ModelBase
from distributed_utils import *
from end2end.arguments import Arguments
import json
import pickle as pkl
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, F1Score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler


def test_model(test_loader):
    with torch.no_grad():
        for video_feats, text_feats, labels, axis_stats in tqdm(iterate(test_loader), desc='Test'):
            output = model(video_feats, text_feats)
            labels = labels.type(torch.int).cuda()
            # test_metrics.update(preds=output, target=labels)
            for ind, axis_stat in enumerate(axis_stats):
                axis_metrics[axis_stat[0]][axis_stat[1]].update(preds=output[ind].view(-1),
                                                                target=labels[ind].view(-1))
            test_metrics[split_type].update(preds=output, target=labels)
        dist.barrier()


def iterate(dataloader):
    for data_batch, label_batch in tqdm(dataloader):
        yield process_batch(data_batch, label_batch)


def process_batch(data_batch, label_batch):
    hypotheses, labels = [], []
    video_feat_batch, vid_lengths = [], []
    axis_stats_batch = []
    for filepath, label in zip(data_batch, label_batch):
        labels.append(float(label))
        '''loading trajectory data'''
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))
        complexity, ordering = task_axis_stats(task_type=traj['task_type'])
        axis_stats_batch.append((complexity, ordering))

        '''sampling frames from the video'''
        # video_frames = [cv2.imread(frame) for frame in glob.glob(os.path.join(filepath, 'raw_images') + "/*.png")]
        video_frames = sample_vid(filepath, args.sample_rate)
        video_frames = torch.stack(video_frames).cuda()  # [t, c, h, w]

        '''process video features using resnet/I3D/mvit'''
        video_feats = extract_video_features(video_frames,
                                             model=visual_model,
                                             feature_extractor=args.visual_feature_extractor,
                                             feat_size=vid_feat_size,
                                             finetune=args.finetune,
                                             test=True)  # [max_seq_len, batch_size, embed_dim]
        video_feat_batch.append(video_feats)
        vid_lengths.append(len(video_feats))

        '''process natural language hypothesis using bert/glove'''
        if label == '0':
            clean_string = ' '.join([clean_str(word).lower() for word in traj['template']['neg'].split(' ')])
        else:
            clean_string = ' '.join([clean_str(word).lower() for word in traj['template']['pos'].split(' ')])
        hypotheses.append(clean_string)

    # tuple: (features, vid_lens) |  sizes: ([batch_size, max_seq_len, embed_dim], [batch_size])
    video_feat_batch = (pad_sequence(video_feat_batch).permute(1, 0, 2).contiguous(),
                        torch.tensor(vid_lengths))
    # tuple: (features, text_len) |  sizes: ([batch_size, max_seq_len, embed_dim], [batch_size])
    text_feat_batch = extract_text_features(hypotheses,
                                            model=text_model,
                                            feature_extractor=args.text_feature_extractor,
                                            tokenizer=tokenizer)
    # print(axis_stats_batch)
    return video_feat_batch, text_feat_batch, torch.tensor(labels).cuda(), axis_stats_batch


if __name__ == '__main__':
    dist_url = "env://"  # default
    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank)
    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()

    args = Arguments()
    print("============ attention is {} ============".format(str(args.attention)))
    # text feature extractor for the hypothesis
    text_model, tokenizer, embed_size = initiate_text_module(args.text_feature_extractor)
    if args.text_feature_extractor == 'bert':
        text_model.cuda()
        text_model = DDP(text_model, device_ids=[local_rank])
        text_model.eval()

    # visual feature extractor for the video
    visual_model, vid_feat_size = initiate_visual_module(args.visual_feature_extractor,
                                                         args.pretrained_mvit)
    visual_model.cuda()
    visual_model = nn.SyncBatchNorm.convert_sync_batchnorm(visual_model)
    visual_model = DDP(visual_model, device_ids=[local_rank])
    visual_model.eval()

    # violin base model
    hsize1 = 150
    hsize2 = 300
    metrics = MetricCollection([Accuracy(threshold=0.5, dist_sync_on_step=True),
                                F1Score(threshold=0.5, dist_sync_on_step=True)]).cuda()
    all_test_splits = ['sub_goal_composition', 'verb_noun_composition',
                       'context_verb_noun_composition', 'context_goal_composition', 'abstraction']

    # axis metrics measure accuracy and F1 along the axes of complexity and ordering (averaged over different run_ids)
    axis_metrics = {complexity: {ordering: metrics.clone(prefix='test_')
                                 for ordering in np.arange(0, 3, 1)} for complexity in np.arange(1, 4, 1)}

    axis_results = {complexity: {ordering: tuple()
                                 for ordering in np.arange(0, 3, 1)} for complexity in np.arange(1, 4, 1)}
    # test metrics measure accuracy, F1 for each test split (averaged over different run_ids)
    test_metrics = {k: metrics.clone(prefix='test_') for k in all_test_splits}

    test_logger = '{}_{}{}_log_test.txt'.format(args.visual_feature_extractor,
                                                    args.text_feature_extractor,
                                                    '_attention' if args.attention else '')
    logger_path = os.path.join(os.getcwd(), test_logger)
    test_log_file = open(logger_path, "w")
    axes_log_file = '{}_{}{}_axes.pkl'.format(args.visual_feature_extractor,
                                              args.text_feature_extractor,
                                              '_attention' if args.attention else '')
    axes_log_path = os.path.join(os.getcwd(), axes_log_file)
    model_val_acc, model_val_f1 = [], []

    for run_id in ['1', '2', '3', '4']:
        ckpt_root = '{}_{}_{}'.format(args.visual_feature_extractor, args.text_feature_extractor,
                                      'attention' if args.attention else 'no_attention')
        ckpt_file = 'violin_{}_{}{}_best_{}.pth'.format(args.visual_feature_extractor,
                                                        args.text_feature_extractor,
                                                        '_attention' if args.attention else '',
                                                        run_id)
        model_ckpt_path = os.path.join(os.environ['CKPTS'], ckpt_root, ckpt_file)
        train_log_file = 'violin_{}_{}{}_log_{}.txt'.format(args.visual_feature_extractor,
                                                            args.text_feature_extractor,
                                                            '_attention' if args.attention else '',
                                                            run_id)
        train_log_path = os.path.join(os.environ['CKPTS'], ckpt_root, train_log_file)
        best_val_acc, best_val_f1 = train_log_process(train_log_path)
        model_val_acc.append(best_val_acc)
        model_val_f1.append(best_val_f1)

        model = ModelBase(hsize1=hsize1, hsize2=hsize2, embed_size=embed_size, vid_feat_size=vid_feat_size,
                           attention=args.attention)
        model.load_state_dict(torch.load(model_ckpt_path))
        model.cuda()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])
        model.eval()

        for split_type in all_test_splits:
            if is_main_process():
                print('======== run_id: {} | split type: {} =============='.format(int(run_id), split_type))
            path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', split_type)
            if args.preprocess:
                preprocess_dataset(path, args.split_type)
            test_set = CustomDataset(data_path=path)
            test_sampler = DistributedSampler(dataset=test_set, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler,
                                     num_workers=args.num_workers, shuffle=False)
            test_model(test_loader)

    # dist.barrier()
    # if is_main_process():
    for com_key, com_val in axis_metrics.items():
        for ord_key, ord_val in com_val.items():
            try:
                acc, f1 = ord_val['Accuracy'].compute(), ord_val['F1Score'].compute()
                axis_results[com_key][ord_key] = tuple((acc.item(), f1.item()))
            except:
                axis_results[com_key][ord_key] = tuple((0, 0))
    print('Axis Results: {}'.format(axis_results))
    pkl.dump(axis_results, open(axes_log_path, 'wb'))

    mean_acc, mean_f1 = str(np.array(model_val_acc).mean()), str(np.array(model_val_f1).mean())
    print('Split Type: validation | Acc: {} | F1: {}'.format(mean_acc, mean_f1))
    test_log_file.write('Split: ' + 'validation' + ' | Test Acc: ' + mean_acc + ' | Test F1: ' + mean_f1 + "\n")
    test_log_file.flush()
    for split_type in all_test_splits:
        test_acc, test_f1 = test_metrics[split_type]['Accuracy'].compute(), \
                            test_metrics[split_type]['F1Score'].compute()
        print('Split Type: {} | Acc: {} | F1: {}'.format(split_type, str(test_acc.item()), str(test_f1.item())))
        test_log_file.write('Split: ' + split_type + ' | Test Acc: ' + str(test_acc.item()) + ' | Test F1: ' + str(test_f1.item()) + "\n")
        test_log_file.flush()
    test_log_file.close()
    print('Done!')
