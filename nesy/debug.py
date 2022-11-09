import os
import sys

os.environ['DATA_ROOT'] = '/mnt/c/Users/rishihazra/PycharmProjects/VisionLangaugeGrounding/alfred/gen/dataset'
os.environ['BASELINES'] = '/mnt/c/Users/rishihazra/PycharmProjects/VisionLangaugeGrounding/baselines'
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from mvit_tx.mvit import mvit_v2_s
from proScript.proscript_utils import GraphEditDistance
from nesy_arguments import Arguments
from dataset_utils import *
from nesy_model_debug import NeSyBase
import json
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import MetricCollection, Accuracy, F1Score
from transformers import T5Tokenizer, T5ForConditionalGeneration


def train_epoch(model, train_loader, val_loader, epoch, previous_best_acc):
    model.train()
    if args.finetune:
        visual_model.train()
    train_loss = []
    for video_feats, graphs, labels in tqdm(iterate(train_loader), desc='Train'):
        output = model(video_feats, graphs)
        print(torch.exp(output))
        loss = bce_loss(torch.exp(output), labels)
        optimizer.zero_grad()
        loss.backward()
        # model.state_dict(keep_vars=True)
        # {k: v.grad for k, v in zip(model.state_dict(), model.parameters())}
        optimizer.step()
        # print('Loss: {}'.format(loss.item()))
        train_loss.append(loss.item())
        labels = labels.type(torch.int)
        train_metrics.update(preds=output, target=labels)

    acc, f1 = train_metrics['Accuracy'].compute(), train_metrics['F1Score'].compute()
    print('Train Loss: {}'.format(np.array(train_loss).mean()))
    # dist.barrier()
    val_acc, val_f1 = validate(model, val_loader=val_loader)
    # dist.barrier()
    # if is_main_process():
    print('Epoch: {} | Train Acc: {} | Val Acc: {} | Train F1: {} | Val F1: {}'.format(
        epoch, acc, val_acc, f1, val_f1))
    log_file.write('Epoch: ' + str(epoch) + ' | Train Acc: ' + str(acc.item()) +
                   ' | Val Acc: ' + str(val_acc.item()) + ' | Train F1: ' + str(f1.item()) +
                   ' | Val F1: ' + str(val_f1.item()) + "\n")
    if val_acc > torch.tensor(previous_best_acc):
        previous_best_acc = val_acc.item()
        # print('============== Saving best model(s) ================')
        # torch.save(model.module.state_dict(), model_ckpt_path)
        # if args.finetune:
        #     torch.save(visual_model.module.state_dict(), visual_model_ckpt_path)
    log_file.flush()
    return previous_best_acc


def validate(model, val_loader):
    model.eval()
    visual_model.eval()
    with torch.no_grad():
        for video_feats, graphs, labels in tqdm(iterate(val_loader), desc='Validation'):
            output = model(video_feats, graphs)
            labels = labels.type(torch.int)
            val_metrics.update(preds=output, target=labels)
    return val_metrics['Accuracy'].compute(), val_metrics['F1Score'].compute()


def iterate(dataloader):
    for data_batch, label_batch in tqdm(dataloader):
        yield process_batch(data_batch, label_batch)


def process_batch(data_batch, label_batch):
    hypotheses = []
    video_features_batch = []  # transforms + visual model features
    labels = []
    vid_lengths = []
    for filepath, label in zip(data_batch, label_batch):
        labels.append(float(label))
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))
        segment_labels = extract_segment_labels(traj, args.sample_rate)

        video_frames = sample_vid(filepath, args.sample_rate)
        video_frames = torch.stack(video_frames)  # [t, c, h, w]
        # here b=1 since we are processing one video at a time
        video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [b, c, t, h, w]
        b, c, t, h, w = video_frames.shape
        num_segments = math.ceil(t / 16)
        assert len(segment_labels) == num_segments or len(segment_labels) == num_segments - 1
        to_pad = num_segments * 16 - t
        video_frames = torch.cat((video_frames, torch.zeros(b, c, to_pad, h, w)), dim=2)
        video_frames = video_frames.reshape(b * num_segments, c, 16, h, w)
        if not args.finetune:
            with torch.no_grad():
                video_segments = visual_model(video_frames).reshape(b * num_segments,
                                                                    vid_feat_size)  # [num_segments, 768]
        else:
            video_segments = visual_model(video_frames).reshape(b * num_segments, vid_feat_size)  # [num_segments, 768]
        video_features_batch.append(video_segments)
        vid_lengths.append(num_segments)

        # process natural language hypothesis using bert
        hypotheses.append(traj['template']['neg']) if label == '0' \
            else hypotheses.append(traj['template']['pos'])

    # generate graphs for hypotheses
    with torch.no_grad():
        # TODO: remove sliced from applesclided etc. in hypotheses
        # print(hypotheses)
        inputs_tokenized = tokenizer(hypotheses, return_tensors="pt", padding=True)
        graphs_batch = t5_model.generate(inputs_tokenized["input_ids"],
                                         attention_mask=inputs_tokenized["attention_mask"],
                                         max_length=max_target_length,
                                         do_sample=False)  # greedy generation
        graphs_batch = tokenizer.batch_decode(graphs_batch, skip_special_tokens=True)
        graphs_batch = [ged.pydot_to_nx(graph_str) for graph_str in graphs_batch]
    # pad_sequence(video_frames_batch) -> [batch_size, max_seq_len, embed_dim]
    video_features_batch = (pad_sequence(video_features_batch).permute(1, 0, 2).contiguous(), torch.tensor(vid_lengths))
    # video_features_batch = (video_features_batch, torch.tensor(vid_lengths))
    return video_features_batch, graphs_batch, torch.tensor(labels)


if __name__ == '__main__':
    args = Arguments()
    ged = GraphEditDistance()
    if args.split_type == 'train':
        path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
    else:
        # path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.split_type)
        path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
    ckpt_file = 'nesy_best_{}.pth'.format(str(args.run_id))
    model_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    logger_filename = 'nesy_log_{}.txt'.format(str(args.run_id))
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

    # text module to generate graph
    max_source_length = 50
    max_target_length = 150
    proscript_model_ckpt_path = os.path.join(os.environ['BASELINES'],
                                             'proScript/proscript_best_1.json')
    t5_model = T5ForConditionalGeneration.from_pretrained(proscript_model_ckpt_path)
    # t5_model.cuda()
    # t5_model = DDP(t5_model, device_ids=[local_rank])
    t5_model.eval()
    # transformers use layer norm (and not batch norm) which is local -- no need to sync across all instances
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    vid_feat_size = 768
    weights = 'KINETICS400_V1' if args.pretrained_mvit else None
    visual_model = mvit_v2_s(weights=weights)
    # visual_model.cuda()
    # visual_model = nn.SyncBatchNorm.convert_sync_batchnorm(visual_model)
    # visual_model = DDP(visual_model, device_ids=[local_rank])
    if not args.finetune:
        visual_model.eval()
    else:
        visual_model_ckpt_path = os.path.join(os.getcwd(), "{}.pth".format('mvit'))

    model = NeSyBase(text_feature_extractor=args.text_feature_extractor)
    all_params = list(model.parameters())
    if args.finetune:
        all_params += list(visual_model.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCELoss()
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
