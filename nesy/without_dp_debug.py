import os
import sys

os.environ['DATA_ROOT'] = '/mnt/c/Users/rishihazra/PycharmProjects/VisionLangaugeGrounding/alfred/gen/dataset'
os.environ['BASELINES'] = '/mnt/c/Users/rishihazra/PycharmProjects/VisionLangaugeGrounding/baselines'
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from proScript.utils import GraphEditDistance
from nesy_arguments import Arguments
from dataset_utils import *
from feature_extraction import *
from end2end.violin.rnn import RNNEncoder
from without_dp_model import NeSyBase
import json
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
from torchmetrics import MetricCollection, Accuracy, F1Score


def train_epoch(model, train_loader, val_loader, epoch, previous_best_acc):
    model.train()
    train_loss = []
    for video_feats, text_feats, segment_labels, ent_labels in tqdm(iterate(train_loader), desc='Train'):
        # ent_preds, state_preds, state_labs, relation_preds, relation_labs = \
        #     model(video_feats, segment_labels)
        ent_preds = model(video_feats, text_feats, segment_labels)
        loss = bce_loss(ent_preds, ent_labels)
        # state_preds = torch.exp(torch.cat(state_preds, dim=0))
        # state_labs = torch.cat(state_labs).long()
        # loss += ce_loss(state_preds, state_labs)
        # adding the state classifier loss
        # try:
        #     state_preds = torch.exp(torch.cat(state_preds, dim=0))
        #     state_labs = torch.cat(state_labs).long()
        #     loss += ce_loss(state_preds, state_labs)
        # except RuntimeError:
        #     pass
        # # adding the relation classifier loss
        # try:
        #     relation_preds = torch.exp(torch.cat(relation_preds, dim=0))
        #     relation_labs = torch.cat(relation_labs).long()
        #     loss += ce_loss(relation_preds, relation_labs)
        # except RuntimeError:
        #     pass
        optimizer.zero_grad()
        loss.backward()
        # model.state_dict(keep_vars=True)
        # {k: v.grad for k, v in zip(model.state_dict(), model.parameters())}
        optimizer.step()
        # print('Loss: {}'.format(loss.item()))
        train_loss.append(loss.item())
        ent_labels = ent_labels.type(torch.int)
        train_metrics.update(preds=ent_preds, target=ent_labels)

    acc, f1 = train_metrics['Accuracy'].compute(), train_metrics['F1Score'].compute()
    print('Train Loss: {}'.format(np.array(train_loss).mean()))
    # dist.barrier()
    val_acc, val_f1 = validate(model, val_loader=val_loader)
    # dist.barrier()
    # if is_main_process():
    print('Epoch: {} | Train Acc: {} | Val Acc: {} | Train F1: {} | Val F1: {}'.format(
        epoch, acc, val_acc, f1, val_f1))
    # log_file.write('Epoch: ' + str(epoch) + ' | Train Acc: ' + str(acc.item()) +
    #                ' | Val Acc: ' + str(val_acc.item()) + ' | Train F1: ' + str(f1.item()) +
    #                ' | Val F1: ' + str(val_f1.item()) + "\n")
    if val_acc > torch.tensor(previous_best_acc):
        previous_best_acc = val_acc.item()
    # print('============== Saving best model(s) ================')
    # torch.save(model.module.state_dict(), model_ckpt_path)
    # torch.save(visual_model.module.state_dict(), visual_model_ckpt_path)
    log_file.flush()
    return previous_best_acc


def validate(model, val_loader):
    model.eval()
    visual_model.eval()
    with torch.no_grad():
        for video_feats, text_feats, segment_labels, ent_labels in tqdm(iterate(val_loader), desc='Validation'):
            # ent_preds, _, _, _, _ = \
            #     model(video_feats, segment_labels)
            ent_preds = model(video_feats, text_feats, segment_labels)
            ent_labels = ent_labels.type(torch.int)
            val_metrics.update(preds=ent_preds, target=ent_labels)
    return val_metrics['Accuracy'].compute(), val_metrics['F1Score'].compute()


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
        ent_labels_batch.append(float(ent_label))
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))
        segment_labels, roi_bb, segment_args = extract_segment_labels(traj, args.sample_rate, frames_per_segment,
                                                        [traj['pddl_params']['object_target']],
                                                                      positive=True if ent_label == '1' else False)
        segment_labels_batch.append(torch.tensor(segment_labels))

        segment_text_feats = extract_text_features(segment_args, text_model, 'clip', tokenizer)
        text_features_batch.append(segment_text_feats)

        video_frames, roi_frames = sample_vid_with_roi(filepath, args.sample_rate, roi_bb)
        video_frames, roi_frames = torch.stack(video_frames), torch.stack(roi_frames)  # [t, c, h, w]
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
        except AssertionError:  # a couple of videos may be corrupted (recording was not done properly)
            continue
        if len(segment_labels) == num_segments - 1:
            segment_labels.append(6)  # NoOp
        to_pad = num_segments * frames_per_segment - t
        video_frames = torch.cat((video_frames, torch.zeros(b, to_pad, vid_feat_size)), dim=1)
        roi_frames = torch.cat((roi_frames, torch.zeros(b, to_pad, vid_feat_size)), dim=1)
        # [num_segments, frames_per_segment, 512]
        video_frames = video_frames.reshape(b * num_segments, frames_per_segment, vid_feat_size)
        roi_frames = roi_frames.reshape(b * num_segments, frames_per_segment, vid_feat_size)

        # TODO: is there any other way to do this? (like concatenation)
        # [num_segments, frames_per_segment, k*512]
        video_frames = torch.cat((video_frames, roi_frames), dim=-1)
        video_features_batch.append(video_frames)
        # vid_lengths.append(num_segments)

    # pad_sequence(video_frames_batch) -> [batch_size, max_seq_len, embed_dim]

    # video_features_batch = (pad_sequence(video_features_batch).permute(1, 0, 2).contiguous())
    # video_features_batch = (video_features_batch, torch.tensor(vid_lengths))
    return video_features_batch, text_features_batch, segment_labels_batch, torch.tensor(ent_labels_batch)


if __name__ == '__main__':
    args = Arguments()
    ged = GraphEditDistance()
    if args.split_type == 'train':
        path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
    else:
        # path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.split_type)
        path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
    ckpt_file = 'nesy_supervised_best_{}.pth'.format(str(args.run_id))
    model_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    logger_filename = 'nesy_supervised_log_{}.txt'.format(str(args.run_id))
    logger_path = os.path.join(os.getcwd(), logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')

    if args.preprocess:
        preprocess_dataset(path, args.split_type)
    dataset = CustomDataset(data_path=path)
    train_size = int(args.data_split * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    # train_sampler, val_sampler = DistributedSampler(dataset=train_set, shuffle=True), \
    #                              DistributedSampler(dataset=val_set, shuffle=True)
    train_loader, val_loader = DataLoader(train_set, batch_size=args.batch_size,
                                          num_workers=args.num_workers, pin_memory=True), \
                               DataLoader(val_set, batch_size=args.batch_size,
                                          num_workers=args.num_workers, shuffle=False, pin_memory=True)

    visual_model, vid_feat_size = initiate_visual_module(feature_extractor='clip')
    if not args.finetune:
        visual_model.eval()

    _, tokenizer, text_feat_size = initiate_text_module(feature_extractor='clip')
    text_model = visual_model  # for clip model

    hsize = 150  # of the aggregator
    hsize2 = 50
    model = NeSyBase(vid_embed_size=vid_feat_size, hsize=hsize, hsize2=hsize2, rnn_enc=RNNEncoder)
    all_params = list(model.parameters())
    if args.finetune:
        all_params += list(visual_model.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
    metrics = MetricCollection([Accuracy(threshold=0.5, dist_sync_on_step=True),
                                F1Score(threshold=0.5, dist_sync_on_step=True)])
    train_metrics = metrics.clone(prefix='train_')
    val_metrics = metrics.clone(prefix='val_')

    best_acc = 0.
    for epoch in range(1, args.epochs + 1):
        # enable different shuffling with set_epoch (uses it to set seed = epoch)
        # train_loader.sampler.set_epoch(epoch)
        # val_loader.sampler.set_epoch(epoch)
        best_acc = train_epoch(model, train_loader, val_loader, epoch=epoch, previous_best_acc=best_acc)
        train_metrics.reset()
        val_metrics.reset()
    # cleanup()
    log_file.close()
    print('Done!')
