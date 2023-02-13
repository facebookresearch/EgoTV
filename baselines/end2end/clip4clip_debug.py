import os
import sys
os.environ['DATA_ROOT'] = '/home/rishihazra/PycharmProjects/VisionLangaugeGrounding/alfred/gen/dataset'
os.environ['BASELINES'] = '/home/rishihazra/PycharmProjects/VisionLangaugeGrounding/baselines'
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from dataset_utils import *
import json
import numpy as np
from tqdm import tqdm
import clip
from CLIP4Clip.clip4clip_model_debug import CLIP4Clip
from arguments import Arguments
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import torch.optim as optim
from torchmetrics import MetricCollection, Accuracy, F1Score
from torch.utils.data import random_split


def train_epoch(clip4clip_model, train_loader, val_loader, epoch, previous_best_acc):
    clip4clip_model.train()
    if args.finetune:
        clip_model.train()
    train_loss = []
    for video_feats, text_feats, labels in tqdm(iterate(train_loader), desc='Train'):
        output = clip4clip_model(video_feats, text_feats)
        loss = bce_loss(output, labels)
        optimizer.zero_grad()
        loss.backward()
        # model.state_dict(keep_vars=True)
        # {k: v.grad for k, v in zip(model.state_dict(), model.parameters())}
        optimizer.step()
        # print('Loss: {}'.format(loss.item()))
        train_loss.append(loss.item())
        labels = labels.type(torch.int)
        train_metrics.update(preds=output, target=labels)

    acc, f1 = list(train_metrics.compute().values())
    print('Train Loss: {}'.format(np.array(train_loss).mean()))
    # dist.barrier()
    val_acc, val_f1 = validate(clip4clip_model, val_loader=val_loader)
    dist.barrier()
    # if is_main_process():
    print('Epoch: {} | Train Acc: {} | Val Acc: {} | Train F1: {} | Val F1: {}'.format(
        epoch, acc, val_acc, f1, val_f1))
    log_file.write('Epoch: ' + str(epoch) + ' | Train Acc: ' + str(acc.item()) +
                   ' | Val Acc: ' + str(val_acc.item()) + ' | Train F1: ' + str(f1.item()) +
                   ' | Val F1: ' + str(val_f1.item()) + "\n")
    # if val_acc > torch.tensor(previous_best_acc).cuda():
    #     previous_best_acc = val_acc.item()
    #     print('============== Saving best model(s) ================')
    #     torch.save(model.module.state_dict(), model_ckpt_path)
    #     if args.finetune:
    #         torch.save(clip_model.module.state_dict(), clip_model_ckpt_path)
    log_file.flush()
    return previous_best_acc


def validate(clip4clip_model, val_loader):
    clip4clip_model.eval()
    clip_model.eval()
    with torch.no_grad():
        for video_feats, text_feats, labels in tqdm(iterate(val_loader), desc='Validation'):
            output = clip4clip_model(video_feats, text_feats)
            labels = labels.type(torch.int)
            val_metrics.update(preds=output, target=labels)
    return list(val_metrics.compute().values())


def iterate(dataloader):
    for data_batch, label_batch in tqdm(dataloader):
        yield process_batch(data_batch, label_batch)


def process_batch(data_batch, label_batch):
    hypotheses, labels = [], []
    video_feat_batch, vid_lengths = [], []
    for filepath, label in zip(data_batch, label_batch):
        labels.append(float(label))
        # ============ loading trajectory data ============ #
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))

        # ============ sampling frames from the video ============ #
        video_frames = sample_vid(filepath, args.sample_rate)
        video_frames = torch.stack(video_frames)  # [t, c, h, w]
        # ============ process image features using clip ============ #
        video_feats = clip_model.encode_image(video_frames)  # [max_seq_len, embed_dim]
        video_feat_batch.append(video_feats)
        vid_lengths.append(len(video_feats))

        # ============ process natural language hypothesis using bert/glove ============ #
        if label == '0':
            clean_string = ' '.join([clean_str(word).lower() for word in traj['template']['neg'].split(' ')])
        else:
            clean_string = ' '.join([clean_str(word).lower() for word in traj['template']['pos'].split(' ')])
        hypotheses.append(clean_string)

    # tuple: (features, vid_lens) |  sizes: ([batch_size, max_seq_len, embed_dim], [batch_size])
    video_feat_batch = (pad_sequence(video_feat_batch).permute(1, 0, 2).contiguous(),
                            torch.tensor(vid_lengths))
    # tuple: features |  size: [batch_size, embed_dim]
    text_feat_batch = clip_model.encode_text(clip.tokenize(hypotheses, truncate=True))
    return video_feat_batch, text_feat_batch, torch.tensor(labels)


if __name__ == '__main__':
    args = Arguments()
    path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
    ckpt_file = 'clip4clip_best_{}.pth'.format(str(args.run_id))
    model_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    logger_filename = 'clip4clip_log_{}.txt'.format(str(args.run_id))
    logger_path = os.path.join(os.getcwd(), logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')

    if args.preprocess:
        preprocess_dataset(path, args.split_type)
    dataset = CustomDataset(data_path=path)
    train_size = int(args.data_split * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader, val_loader = DataLoader(train_set, batch_size=args.batch_size,
                                          num_workers=args.num_workers, pin_memory=True), \
                               DataLoader(val_set, batch_size=args.batch_size,
                                          num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # CLIP4Clip model (https://github.com/ArrowLuo/CLIP4Clip)
    clip_model, _ =  clip.load("ViT-B/32")
    if not args.finetune:
        clip_model.eval()
    else:
        clip_model_ckpt_path = os.path.join(os.getcwd(), "{}.pth".format('clip'))
        clip_model.load_state_dict(torch.load(clip_model_ckpt_path))


    # clip4clip baseline
    args.sim_type = 'seqLSTM'

    clip4clip_model = CLIP4Clip(embed_size=512, sim_type=args.sim_type)
    if args.resume:  # to resume from a previously stored checkpoint
        clip4clip_model.load_state_dict(torch.load(model_ckpt_path))

    if args.sim_type == 'meanPool':
        all_params = list()
    else:
        all_params = list(clip4clip_model.parameters())
    if args.finetune:
        all_params += list(clip_model.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCELoss()
    metrics = MetricCollection([Accuracy(dist_sync_on_step=True, task='binary'),
                                F1Score(dist_sync_on_step=True, task='binary')])
    train_metrics = metrics.clone(prefix='train_')
    val_metrics = metrics.clone(prefix='val_')

    best_acc = 0.
    for epoch in range(1, args.epochs+1):
        # enable different shuffling with set_epoch (uses it to set seed = epoch)
        # train_loader.sampler.set_epoch(epoch)
        # val_loader.sampler.set_epoch(epoch)
        best_acc = train_epoch(clip4clip_model, train_loader, val_loader, epoch=epoch, previous_best_acc=best_acc)
        train_metrics.reset()
        val_metrics.reset()
    # cleanup()
    log_file.close()
    print('Done!')
