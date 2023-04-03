# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29500 oracleoracle_test.py --num_workers 4 --split_type 'sub_goal_composition' --batch_size 64 --sample_rate 2 --run_id 1 --fp_seg 8
import os
import sys

sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from nesy_arguments import Arguments
from dataset_utils import *
from distributed_utils import *
from feature_extraction import *
from VIOLIN.rnn import RNNEncoder
from oracle_model import NeSyBase
import json
import math
import torch
from tqdm import tqdm
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, F1Score
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def test_model(test_loader):
    with torch.no_grad():
        for video_feats, text_feats, segment_labels, ent_labels in tqdm(iterate(test_loader), desc='Test'):
            ent_preds = model(video_feats, text_feats, segment_labels)
            ent_labels = ent_labels.type(torch.int)
            test_metrics.update(preds=ent_preds, target=ent_labels)
    dist.barrier()
    test_acc, test_f1 = list(test_metrics.compute().values())
    dist.barrier()
    if is_main_process():
        print('Test Acc: {} | Test F1: {}'.format(test_acc, test_f1))
        log_file.write('Test Acc: ' + str(test_acc.item()) + ' | Test F1: ' + str(test_f1.item()) + "\n")
        log_file.flush()


def iterate(dataloader):
    for data_batch, ent_label_batch in tqdm(dataloader):
        yield process_batch(data_batch, ent_label_batch, frames_per_segment=args.fp_seg)


def process_batch(data_batch, ent_label_batch, frames_per_segment):
    # TODO: frames_per_segment as a hyperparameter
    video_features_batch = []  # transforms + visual model features
    segment_labels_batch = []
    ent_labels_batch = []
    text_features_batch = []
    for filepath, ent_label in zip(data_batch, ent_label_batch):
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))
        segment_labels, roi_bb, segment_args = extract_segment_labels(traj, args.sample_rate, frames_per_segment,
                                                                      [traj['pddl_params']['object_target']],
                                                                      positive=True if ent_label == '1' else False)
        # segment_args = extract_text_features(segment_args, text_model,
        #                                      args.text_feature_extractor, tokenizer)[0]  # [num_segments, 1, 300d]

        video_frames, roi_frames = sample_vid_with_roi(filepath, args.sample_rate, roi_bb)
        video_frames, roi_frames = torch.stack(video_frames).cuda(), torch.stack(roi_frames).cuda()  # [t, c, h, w]
        # here b=1 since we are processing one video at a time
        video_frames = extract_video_features(video_frames, model=visual_model,
                                              feature_extractor='clip',
                                              feat_size=vid_feat_size,
                                              finetune=args.finetune).reshape(1, -1, vid_feat_size)
        roi_frames = extract_video_features(roi_frames, model=visual_model,
                                            feature_extractor='clip',
                                            feat_size=vid_feat_size,
                                            finetune=args.finetune).reshape(1, -1, vid_feat_size)
        b, t, _ = video_frames.shape

        num_segments = math.ceil(t / frames_per_segment)
        # since segment_labels are extracted from traj.json, and num_segments from the video :
        # len(segment_labels) <= num_segments
        try:
            assert len(segment_labels) == num_segments or len(segment_labels) == num_segments - 1
        except AssertionError:
            continue
        if len(segment_labels) == num_segments - 1:
            segment_labels.append(6)  # NoOp
        to_pad = num_segments * frames_per_segment - t
        video_frames = torch.cat((video_frames, torch.zeros(b, to_pad, vid_feat_size).cuda()), dim=1)
        roi_frames = torch.cat((roi_frames, torch.zeros(b, to_pad, vid_feat_size).cuda()), dim=1)
        # [num_segments, frames_per_segment, 512]
        video_frames = video_frames.reshape(b * num_segments, frames_per_segment, vid_feat_size)
        roi_frames = roi_frames.reshape(b * num_segments, frames_per_segment, vid_feat_size)

        # TODO: is there any other way to do this? (like concatenation)
        # [num_segments, frames_per_segment, k*512]
        video_frames = torch.cat((video_frames, roi_frames), dim=-1)
        video_features_batch.append(video_frames)
        # vid_lengths.append(num_segments)
        segment_labels_batch.append(torch.tensor(segment_labels).cuda())
        ent_labels_batch.append(float(ent_label))

        segment_text_feats = extract_text_features(segment_args, text_model, 'clip', tokenizer)
        text_features_batch.append(segment_text_feats)

    return video_features_batch, text_features_batch, segment_labels_batch, torch.tensor(ent_labels_batch).cuda()


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
    # ged = GraphEditDistance()
    path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.split_type)
    ckpt_file = 'without_dp_best_{}.pth'.format(str(args.run_id))
    model_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    logger_filename = 'without_dp_log_test_{}.txt'.format(str(args.run_id))
    logger_path = os.path.join(os.getcwd(), logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')

    if args.preprocess:
        preprocess_dataset(path, args.split_type)
    test_set = CustomDataset(data_path=path)
    test_sampler = DistributedSampler(dataset=test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler,
                             num_workers=args.num_workers, shuffle=False)

    visual_model, vid_feat_size, _ = initiate_visual_module(feature_extractor='clip')
    visual_model.cuda()
    # visual_model = nn.SyncBatchNorm.convert_sync_batchnorm(visual_model)
    visual_model = DDP(visual_model, device_ids=[local_rank])
    visual_model.eval()

    _, tokenizer, text_feat_size = initiate_text_module(feature_extractor='clip')
    text_model = visual_model  # for clip model
    text_model.eval()

    hsize = 150  # of the aggregator
    hsize2 = 50
    model = NeSyBase(vid_embed_size=vid_feat_size, hsize=hsize, hsize2=50, rnn_enc=RNNEncoder)
    model.load_state_dict(torch.load(model_ckpt_path))
    model.cuda()
    # will have unused params for certain samples (StateQuery / RelationQuery)
    # , find_unused_parameters=True
    model = DDP(model, device_ids=[local_rank])
    model.eval()
    metrics = MetricCollection([Accuracy(threshold=0.5, dist_sync_on_step=True, task='binary'),
                                F1Score(threshold=0.5, dist_sync_on_step=True, task='binary')]).cuda()
    test_metrics = metrics.clone(prefix='test_')
    test_model(test_loader)
    test_metrics.reset()
    cleanup()
    log_file.close()
    print('Done!')
