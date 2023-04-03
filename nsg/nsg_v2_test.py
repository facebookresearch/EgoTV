# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import sys

sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from nesy_arguments import Arguments
from proScript.utils import GraphEditDistance
from dataset_utils import *
from distributed_utils import *
from feature_extraction import *
from VIOLIN.rnn import RNNEncoder
from nesy_model_v2 import NeSyBase
import json
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, F1Score, ConfusionMatrix
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.distributed as dist
from torch.utils.data import random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def test_model(test_loader):
    with torch.no_grad():
        for video_feats, graphs, labels, segment_labs, task_types in tqdm(iterate(test_loader), desc='Test'):
            preds, labels, pred_alignment, _ = model(video_feats, graphs, labels, task_types, train=False)
            labels = labels.type(torch.int)
            action_pred_labs, action_true_labs = \
                check_alignment_v2(pred_alignment, segment_labs, labels)
            test_metrics.update(preds=preds, target=labels)
            if len(action_pred_labs) != 0 and len(action_true_labs) != 0:
                action_query_metrics.update(preds=action_pred_labs.cuda(), target=action_true_labs.cuda())

        dist.barrier()
        test_acc, test_f1 = list(test_metrics.compute().values())
        action_cf = torch.cat(list(action_query_metrics.compute().values()))
        dist.barrier()
        if is_main_process():
            print('Test Acc: {} | Test F1: {}'.format(test_acc, test_f1))
            log_file.write('Test Acc: ' + str(test_acc.item()) + ' | Test F1: ' + str(test_f1.item()) + "\n")
            print(action_cf)
            np.savetxt(cf_path, action_cf.cpu().numpy(), fmt='%d')
            log_file.flush()


def iterate(dataloader):
    for data_batch, ent_label_batch in tqdm(dataloader):
        # try:
        yield process_batch(data_batch, ent_label_batch, frames_per_segment=args.fp_seg)
        # except TypeError:
        #     print('Skipping batch')
        #     continue


def process_batch(data_batch, label_batch, frames_per_segment):
    hypotheses = []
    video_features_batch = []  # transforms + visual model features
    labels = []
    task_types = []
    segment_labs_batch = []

    for filepath, label in zip(data_batch, label_batch):
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))
        task_types.append(traj['task_type'])
        hypotheses.append(traj['template']['neg']) if label == '0' \
            else hypotheses.append(traj['template']['pos'])
        labels.append(float(label))

    # generate graphs for hypotheses
    with torch.no_grad():
        inputs_tokenized = tokenizer(hypotheses, return_tensors="pt",
                                     padding="longest",
                                     max_length=max_source_length,
                                     truncation=True)
        graphs_batch = t5_model.module.generate(inputs_tokenized["input_ids"].cuda(),
                                                attention_mask=inputs_tokenized["attention_mask"].cuda(),
                                                max_length=max_target_length,
                                                do_sample=False)  # greedy generation
        graphs_batch = tokenizer.batch_decode(graphs_batch, skip_special_tokens=True)
        graphs_batch = ([ged.pydot_to_nx(graph_str) for graph_str in graphs_batch], hypotheses)

    all_arguments = retrieve_query_args(graphs_batch[0])  # retrieve query arguments from the graph batch

    for sample_ind, (filepath, label) in enumerate(zip(data_batch, label_batch)):
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))
        segment_labs, roi_bb, _ = extract_segment_labels(traj, args.sample_rate, frames_per_segment,
                                                         all_arguments[sample_ind])
        segment_labs_batch.append(segment_labs)
        video_frames, _ = sample_vid_with_roi(filepath, args.sample_rate, roi_bb)
        video_frames = torch.stack(video_frames).cuda() # [t, c, h, w]
        # here b=1 since we are processing one video at a time
        video_frames = extract_video_features(video_frames, model=visual_model,
                                              feature_extractor='clip',
                                              feat_size=vid_feat_size,
                                              finetune=args.finetune).reshape(1, -1, vid_feat_size)
        # roi_frames = extract_video_features(roi_frames.reshape(-1, 3, 224, 224), model=visual_model,
        #                                     feature_extractor='clip',
        #                                     feat_size=vid_feat_size,
        #                                     finetune=args.finetune).reshape(1, -1, 3 * vid_feat_size)
        b, t, _ = video_frames.shape
        num_segments = math.ceil(t / frames_per_segment)
        to_pad = num_segments * frames_per_segment - t
        # zero-padding to match the number of frames per segment
        video_frames = torch.cat((video_frames, torch.zeros(b, to_pad, vid_feat_size).cuda()), dim=1)
        # roi_frames = torch.cat((roi_frames, torch.zeros(b, to_pad, 3 * vid_feat_size).cuda()), dim=1)
        # [num_segments, frames_per_segment, 512]
        video_frames = video_frames.reshape(b * num_segments, frames_per_segment, vid_feat_size)
        # roi_frames = roi_frames.reshape(b * num_segments, frames_per_segment, 3 * vid_feat_size)

        # [num_segments, frames_per_segment, k*512]
        # video_frames = torch.cat((video_frames, roi_frames), dim=-1)
        video_features_batch.append(video_frames)

    return video_features_batch, graphs_batch, torch.tensor(labels).cuda(), segment_labs_batch, task_types


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
    ged = GraphEditDistance()
    args = Arguments()
    # ged = GraphEditDistance()
    if args.split_type == 'train':
        path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
    else:
        path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.split_type)

    ckpt_file = 'nesy_v2_best_{}.pth'.format(str(args.run_id))
    model_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    logger_filename = 'nesy_v2_log_test_{}.txt'.format(str(args.run_id))
    logger_path = os.path.join(os.getcwd(), logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')
    cf_filename = 'confusionMat_v2_{}_{}.txt'.format(args.split_type, str(args.run_id))
    cf_path = os.path.join(os.getcwd(), cf_filename)

    if args.preprocess:
        preprocess_dataset(path, args.split_type)

    if args.split_type == 'train':
        dataset = CustomDataset(data_path=path)
        train_size = int(args.data_split * len(dataset))
        val_size = len(dataset) - train_size
        _, test_set = random_split(dataset, [train_size, val_size])
    else:
        test_set = CustomDataset(data_path=path)
    test_sampler = DistributedSampler(dataset=test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler,
                             num_workers=args.num_workers, shuffle=False)

    # text module to generate graph
    max_source_length = 80
    max_target_length = 300
    proscript_model_ckpt_path = os.path.join(os.environ['BASELINES'],
                                             'proScript/proscript_best_nl_1.json')
    t5_model = T5ForConditionalGeneration.from_pretrained(proscript_model_ckpt_path)
    t5_model.cuda()
    t5_model = DDP(t5_model, device_ids=[local_rank])
    t5_model.eval()
    # transformers use layer norm (and not batch norm) which is local -- no need to sync across all instances
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    visual_model, vid_feat_size, _ = initiate_visual_module(feature_extractor='clip')
    visual_model.cuda()
    # visual_model = nn.SyncBatchNorm.convert_sync_batchnorm(visual_model)
    visual_model = DDP(visual_model, device_ids=[local_rank])
    visual_model.eval()

    _, _, text_feat_size = initiate_text_module(feature_extractor='clip')
    text_model = visual_model  # for clip model
    text_model.eval()

    hsize = 150  # of the aggregator
    model = NeSyBase(vid_embed_size=vid_feat_size, hsize=hsize, rnn_enc=RNNEncoder, text_model=text_model)
    model.load_state_dict(torch.load(model_ckpt_path))
    model.cuda()
    # will have unused params for certain samples (StateQuery / RelationQuery)
    # , find_unused_parameters=True
    model = DDP(model, device_ids=[local_rank])
    model.eval()
    metrics = MetricCollection([Accuracy(threshold=0.5, dist_sync_on_step=True, task='binary'),
                                F1Score(threshold=0.5, dist_sync_on_step=True, task='binary')]).cuda()
    test_metrics = metrics.clone(prefix='test_')

    metrics_multiclass = MetricCollection([ConfusionMatrix(dist_sync_on_step=True,
                                                           task='multiclass', num_classes=6)]).cuda()
    action_query_metrics = metrics_multiclass.clone(prefix='action_query_')
    test_model(test_loader)
    test_metrics.reset()
    action_query_metrics.reset()
    cleanup()
    log_file.close()
    print('Done!')
