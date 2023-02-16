import os
import sys
os.environ['DATA_ROOT'] = '/home/rishihazra/PycharmProjects/VisionLangaugeGrounding/alfred/gen/dataset'
os.environ['BASELINES'] = '/home/rishihazra/PycharmProjects/VisionLangaugeGrounding/baselines'
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])

from dataset_utils import *
from distributed_utils import *
from CoCa.coca_model import CocaVidModel
import open_clip
import json
import numpy as np
from tqdm import tqdm
from arguments import Arguments
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import torch.optim as optim
from torchmetrics import MetricCollection, Accuracy, F1Score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler


def train_epoch(model, train_loader, val_loader, epoch, previous_best_acc):
    model.train()
    if args.finetune:
        coca_backbone.train()
    train_loss = []
    for video_feats, text_feats, labels in tqdm(iterate(train_loader), desc='Train'):
        output = model(video_feats, text_feats)
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
    val_acc, val_f1 = validate(model, val_loader=val_loader)
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


def validate(model, val_loader):
    model.eval()
    coca_backbone.eval()
    with torch.no_grad():
        for video_feats, text_feats, labels in tqdm(iterate(val_loader), desc='Validation'):
            output = model(video_feats, text_feats)
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

        # ===========, type='rgb'= sampling frames from the video ============ #
        video_frames = sample_vid(filepath, args.sample_rate)
        video_frames = torch.stack(video_frames) # [t, c, h, w]
        # ============ process image features using clip ============ #
        with torch.no_grad():
            video_feats = coca_backbone.encode_image(video_frames).float() # [max_seq_len, embed_dim]
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
    with torch.no_grad():
        text_feat_batch = coca_backbone.encode_text(tokenizer(hypotheses))
    return video_feat_batch, text_feat_batch, torch.tensor(labels)


if __name__ == '__main__':
    args = Arguments()
    path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
    ckpt_file = 'coca_best_{}.pth'.format(str(args.run_id))
    model_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    logger_filename = 'coca_log_{}.txt'.format(str(args.run_id))
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

    # CoCa (finetuned model from OpenCLIP)
    coca_backbone, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-B-32",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )
    if not args.finetune:
        coca_backbone.eval()
    else:
        coca_model_ckpt_path = os.path.join(os.getcwd(), "{}.pth".format('coca'))
        coca_backbone.load_state_dict(torch.load(coca_model_ckpt_path))
    tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

    # coca video baseline
    model = CocaVidModel(embed_size=512)
    if args.resume:  # to resume from a previously stored checkpoint
        model.load_state_dict(torch.load(model_ckpt_path))

    all_params = list(model.parameters())
    if args.finetune:
        all_params += list(coca_backbone.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCELoss()
    metrics = MetricCollection([Accuracy(dist_sync_on_step=True, task='binary'),
                                F1Score(dist_sync_on_step=True, task='binary')])
    train_metrics = metrics.clone(prefix='train_')
    val_metrics = metrics.clone(prefix='val_')

    best_acc = 0.
    for epoch in range(1, args.epochs+1):
        best_acc = train_epoch(model, train_loader, val_loader, epoch=epoch, previous_best_acc=best_acc)
        train_metrics.reset()
        val_metrics.reset()
    cleanup()
    log_file.close()
    print('Done!')
