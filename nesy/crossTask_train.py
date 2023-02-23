import os
import sys
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from nesy_arguments import Arguments
from dataset_utils import *
from distributed_utils import *
from feature_extraction import *
from VIOLIN.rnn import RNNEncoder
from crossTask_model import NeSyBase
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, F1Score
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler


def train_epoch(model, train_loader, val_loader, epoch, previous_best_acc):
    model.train()
    if args.finetune:
        visual_model.train()
    train_loss = []
    for video_feats, all_sorts_batch, labels in tqdm(iterate(train_loader), desc='Train'):
        preds, labels = model(video_feats, all_sorts_batch, labels)
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
    val_acc, val_f1 = validate(model, val_loader=val_loader)
    dist.barrier()
    if is_main_process():
        print('Epoch: {} | Train Acc: {} | Val Acc: {} | Train F1: {} | Val F1: {}'.format(epoch, acc, val_acc, f1, val_f1,))
        log_file.write('Epoch: ' + str(epoch) + ' | Train Acc: ' + str(acc.item()) +
                       ' | Val Acc: ' + str(val_acc.item()) + ' | Train F1: ' + str(f1.item()) +
                       ' | Val F1: ' + str(val_f1.item()) + "\n")
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
        for video_feats, all_sorts_batch, labels in tqdm(iterate(val_loader, validation=True), desc='Validation'):
            preds, labels, pred_alignment = model(video_feats, all_sorts_batch, labels, train=False)
            labels = labels.type(torch.int)
            val_metrics.update(preds=preds, target=labels)
    return val_metrics['Accuracy'].compute(), val_metrics['F1Score'].compute()


def iterate(dataloader, validation=False):
    for data_batch, label_batch in tqdm(dataloader):
        yield process_batch(data_batch, label_batch, frames_per_segment=args.fp_seg, validation=validation)


def process_batch(data_batch, label_batch, frames_per_segment, validation=False):
    all_sorts_batch = []
    video_features_batch = []  # transforms + visual model features
    labels = []

    for filepath, label in zip(data_batch, label_batch):
        #TODO: change this line sample frames from the video
        video_frames = your_function()

        video_frames = torch.stack(video_frames).cuda()  # [t, c, h, w]
        # here b=1 since we are processing one video at a time
        video_frames = extract_video_features(video_frames, model=visual_model,
                                              feature_extractor='clip',
                                              feat_size=vid_feat_size,
                                              finetune=args.finetune).reshape(1, -1, vid_feat_size)

        # padding the videos to have equal number of frames per segment
        b, t, _ = video_frames.shape
        num_segments = math.ceil(t / frames_per_segment)
        to_pad = num_segments * frames_per_segment - t
        video_frames = torch.cat((video_frames, torch.zeros(b, to_pad, vid_feat_size).cuda()), dim=1)
        # [num_segments, frames_per_segment, 512]
        video_frames = video_frames.reshape(b * num_segments, frames_per_segment, vid_feat_size)
        video_features_batch.append(video_frames)

        # process natural language hypothesis using bert
        #TODO: change this line to extract sorted hypothesis
        #TODO: the format is List[List] where each inner list is a possible sorted sequence
        #TODO: [[action 0, action 1, ...], [action 0, action 2, action 1, ...]]
        #TODO: i believe you now have just one sorted sequence: [[action 0, action 1, ....]]
        all_sorts = your_function()
        assert len(all_sorts[0]) <= num_segments, "number of segments should be more than the number of nodes, " \
                                                  "use a lower value for --fp_seg"
        all_sorts_batch.append(all_sorts)

        labels.append(float(label))

    return video_features_batch, all_sorts_batch, torch.tensor(labels).cuda()


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

    # TODO: change this
    if args.split_type == 'train':
        path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
    else:
        path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.split_type)

    ckpt_file = 'cross_task_nesy_best_{}.pth'.format(str(args.run_id))
    model_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    logger_filename = 'cross_task_nesy_log_{}.txt'.format(str(args.run_id))
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

    best_acc = 0.
    for epoch in range(1, args.epochs+1):
        # enable different shuffling with set_epoch (uses it to set seed = epoch)
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        best_acc = train_epoch(model, train_loader, val_loader, epoch=epoch, previous_best_acc=best_acc)
        train_metrics.reset()
        val_metrics.reset()
    cleanup()
    log_file.close()
    print('Done!')