import os
import sys
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from proScript.utils import GraphEditDistance
from nesy_arguments import Arguments
from dataset_utils import *
from distributed_utils import *
from feature_extraction import *
from end2end.rnn import RNNEncoder
from nesy_model import NeSyBase
import json
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, F1Score
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler


def train_epoch(model, train_loader, val_loader, epoch, previous_best_acc):
    model.train()
    if args.finetune:
        visual_model.train()
    train_loss = []
    for video_feats, graphs, labels, task_types in tqdm(iterate(train_loader), desc='Train'):
        preds, labels = model(video_feats, graphs, labels, task_types)
        loss = bce_loss(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        labels = labels.type(torch.int)
        train_metrics.update(preds=preds, target=labels)

    acc, f1 = train_metrics['Accuracy'].compute(), train_metrics['F1Score'].compute()
    print('Train Loss: {}'.format(np.array(train_loss).mean()))
    dist.barrier()
    val_acc, val_f1, state_acc, state_f1, rel_acc, rel_f1 = validate(model, val_loader=val_loader)
    dist.barrier()
    if is_main_process():
        print('Epoch: {} | Train Acc: {} | Val Acc: {} | Train F1: {} | Val F1: {} | State Acc: {} | State F1: {} | '
                  'Relation Acc: {} | Relation F1: {}'.format(epoch, acc, val_acc, f1, val_f1, state_acc,
                                                              state_f1, rel_acc, rel_f1))
        log_file.write('Epoch: ' + str(epoch) + ' | Train Acc: ' + str(acc.item()) +
                       ' | Val Acc: ' + str(val_acc.item()) + ' | Train F1: ' + str(f1.item()) +
                       ' | Val F1: ' + str(val_f1.item()) +
                       ' | State Acc: ' + str(state_acc.item()) + ' | State F1: ' + str(state_f1.item()) +
                       ' | Relation Acc: ' + str(rel_acc.item()) + ' | Relation F1: ' + str(rel_f1.item()) + "\n")
        if val_acc > torch.tensor(previous_best_acc):
            previous_best_acc = val_acc.item()
            print('============== Saving best model(s) ================')
            torch.save(model.module.state_dict(), model_ckpt_path)
            if args.finetune:
                torch.save(visual_model.module.state_dict(), visual_model_ckpt_path)
        log_file.flush()
    return previous_best_acc


def validate(model, val_loader):
    model.eval()
    visual_model.eval()
    text_model.eval()
    with torch.no_grad():
        for video_feats, graphs, labels, segment_labs, task_types in tqdm(iterate(val_loader, validation=True), desc='Validation'):
            preds, labels, pred_alignment, tasks = model(video_feats, graphs, labels, task_types, train=False)
            labels = labels.type(torch.int)
            state_pred_labs, state_true_labs, relation_pred_labs, relation_true_labs, _ = \
                check_alignment(pred_alignment, segment_labs, labels)
            val_metrics.update(preds=preds, target=labels)
            state_query_metrics.update(preds=state_pred_labs, target=state_true_labs)
            relation_query_metrics.update(preds=relation_pred_labs, target=relation_true_labs)
    return val_metrics['Accuracy'].compute(), val_metrics['F1Score'].compute(), \
           state_query_metrics['Accuracy'].compute(), state_query_metrics['F1Score'].compute(), \
           relation_query_metrics['Accuracy'].compute(), relation_query_metrics['F1Score'].compute()


def iterate(dataloader, validation=False):
    for data_batch, label_batch in tqdm(dataloader):
        yield process_batch(data_batch, label_batch, frames_per_segment=args.fp_seg, validation=validation)

def process_batch(data_batch, label_batch, frames_per_segment, validation=False):
    hypotheses = []
    video_features_batch = []  # transforms + visual model features
    labels = []
    task_types = []
    segment_labs_batch = []

    for filepath, label in zip(data_batch, label_batch):
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))
        hypotheses.append(traj['template']['neg']) if label == '0' \
            else hypotheses.append(traj['template']['pos'])
        labels.append(float(label))
        task_types.append(traj['task_type'])

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
        if validation:
            segment_labs, roi_bb, _ = extract_segment_labels(traj, args.sample_rate, frames_per_segment,
                                                             all_arguments[sample_ind],
                                                             positive=True if label == '1' else False)
            segment_labs_batch.append(segment_labs)
        else:
            roi_bb = extract_segment_labels(traj, args.sample_rate, frames_per_segment,
                                            all_arguments[sample_ind],
                                            positive=True if label == '1' else False,
                                            supervised=False)
        video_frames, roi_frames = sample_vid_with_roi(filepath, args.sample_rate, roi_bb)
        video_frames, roi_frames = torch.stack(video_frames).cuda(), torch.stack(roi_frames).cuda()  # [t, c, h, w]
        # here b=1 since we are processing one video at a time
        video_frames = extract_video_features(video_frames, model=visual_model,
                                              feature_extractor='clip',
                                              feat_size=vid_feat_size,
                                              finetune=args.finetune).reshape(1, -1, vid_feat_size)
        # here the factor of '3' comes from
        # there being max 3 bounding box in each frame
        roi_frames = extract_video_features(roi_frames.reshape(-1, 3, 224, 224), model=visual_model,
                                            feature_extractor='clip',
                                            feat_size=vid_feat_size,
                                            finetune=args.finetune).reshape(1, -1, 3*vid_feat_size)
        b, t, _ = video_frames.shape
        num_segments = math.ceil(t / frames_per_segment)
        to_pad = num_segments * frames_per_segment - t
        # zero-padding to match the number of frames per segment
        video_frames = torch.cat((video_frames, torch.zeros(b, to_pad, vid_feat_size).cuda()), dim=1)
        roi_frames = torch.cat((roi_frames, torch.zeros(b, to_pad, 3*vid_feat_size).cuda()), dim=1)
        # [num_segments, frames_per_segment, 512]
        video_frames = video_frames.reshape(b * num_segments, frames_per_segment, vid_feat_size)
        roi_frames = roi_frames.reshape(b * num_segments, frames_per_segment, 3 * vid_feat_size)

        # [num_segments, frames_per_segment, k*512]
        video_frames = torch.cat((video_frames, roi_frames), dim=-1)
        video_features_batch.append(video_frames)

    if validation:
        return video_features_batch, graphs_batch, torch.tensor(labels).cuda(), segment_labs_batch, task_types
    return video_features_batch, graphs_batch, torch.tensor(labels).cuda(), task_types


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
    ged = GraphEditDistance()
    if args.split_type == 'train':
        path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
    else:
        path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.split_type)
        # path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
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
    train_sampler, val_sampler = DistributedSampler(dataset=train_set, shuffle=True), \
                                 DistributedSampler(dataset=val_set, shuffle=True)
    train_loader, val_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                                          num_workers=args.num_workers, pin_memory=True), \
                               DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler,
                                          num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # text module to generate graph
    max_source_length = 80
    max_target_length = 300
    proscript_model_ckpt_path = os.path.join(os.environ['BASELINES'],
                                             'proScript/proscript_best_dsl_3.json')
    t5_model = T5ForConditionalGeneration.from_pretrained(proscript_model_ckpt_path)
    t5_model.cuda()
    t5_model = DDP(t5_model, device_ids=[local_rank])
    t5_model.eval()
    # transformers use layer norm (and not batch norm) which is local -- no need to sync across all instances
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    visual_model, vid_feat_size = initiate_visual_module(feature_extractor='clip')
    visual_model.cuda()
    visual_model = nn.SyncBatchNorm.convert_sync_batchnorm(visual_model)
    visual_model = DDP(visual_model, device_ids=[local_rank])
    if not args.finetune:
        visual_model.eval()
    else:
        visual_model_ckpt_path = os.path.join(os.getcwd(), "{}.pth".format('clip'))

    _, _, text_feat_size = initiate_text_module(feature_extractor='clip')
    text_model = visual_model  # for clip model

    hsize = 150
    model = NeSyBase(vid_embed_size=vid_feat_size, hsize=hsize, rnn_enc=RNNEncoder, text_model=text_model)
    if args.resume:  # to resume from a previously stored checkpoint
        model.load_state_dict(torch.load(model_ckpt_path))
    model.cuda()
    # will have unused params for certain samples (StateQuery / RelationQuery)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    all_params = list(model.parameters())
    if args.finetune:
        all_params += list(visual_model.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCELoss()
    metrics = MetricCollection([Accuracy(threshold=0.5, dist_sync_on_step=True),
                                F1Score(threshold=0.5, dist_sync_on_step=True)]).cuda()
    train_metrics = metrics.clone(prefix='train_')
    val_metrics = metrics.clone(prefix='val_')
    state_query_metrics = metrics.clone(prefix='state_query_')
    relation_query_metrics = metrics.clone(prefix='relation_query_')

    best_acc = 0.
    for epoch in range(1, args.epochs+1):
        # enable different shuffling with set_epoch (uses it to set seed = epoch)
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        best_acc = train_epoch(model, train_loader, val_loader, epoch=epoch, previous_best_acc=best_acc)
        train_metrics.reset()
        val_metrics.reset()
        state_query_metrics.reset()
        relation_query_metrics.reset()
    cleanup()
    log_file.close()
    print('Done!')