'''
VideoCLIP setup

# create a new conda env since the packages used are different from EgoTV packages
conda create -n videoclip python=3.8.8
source activate videoclip

# clone the repo and run the following installations
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install -e .  # also optionally follow fairseq README for apex installation for fp16 training.
export MKL_THREADING_LAYER=GNU  # fairseq may need this for numpy
cd examples/MMPT  # MMPT can be in any folder, not necessarily under fairseq/examples.
pip install -e .
pip install transformers==3.4

# download the checkpoint using
wget -P runs/retri/videoclip/ "https://dl.fbaipublicfiles.com/MMPT/retri/videoclip/checkpoint_best.pt"
'''
import os
import sys
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
sys.path.append(os.environ['S3D'])

from dataset_utils import *
from distributed_utils import *
from VideoCLIP.videoclip_model import VideoClipModel
from fairseq.examples.MMPT.mmpt.models import MMPTModel
import json
import math
import numpy as np
from tqdm import tqdm
from arguments import Arguments
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torchmetrics import MetricCollection, Accuracy, F1Score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler


def train_epoch(model, train_loader, val_loader, epoch, previous_best_acc):
    model.train()
    videoclip_backbone.eval()
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
        labels = labels.type(torch.int).cuda()
        train_metrics.update(preds=output, target=labels)

    acc, f1 = list(train_metrics.compute().values())
    print('Train Loss: {}'.format(np.array(train_loss).mean()))
    dist.barrier()
    val_acc, val_f1 = validate(model, val_loader=val_loader)
    dist.barrier()
    if is_main_process():
        print('Epoch: {} | Train Acc: {} | Val Acc: {} | Train F1: {} | Val F1: {}'.format(
            epoch, acc, val_acc, f1, val_f1))
        log_file.write('Epoch: ' + str(epoch) + ' | Train Acc: ' + str(acc.item()) +
                       ' | Val Acc: ' + str(val_acc.item()) + ' | Train F1: ' + str(f1.item()) +
                       ' | Val F1: ' + str(val_f1.item()) + "\n")
        if val_acc > torch.tensor(previous_best_acc).cuda():
            previous_best_acc = val_acc.item()
            print('============== Saving best model(s) ================')
            torch.save(model.module.state_dict(), model_ckpt_path)
        log_file.flush()
    return previous_best_acc


def validate(model, val_loader):
    model.eval()
    videoclip_backbone.eval()
    with torch.no_grad():
        for video_feats, text_feats, labels in tqdm(iterate(val_loader), desc='Validation'):
            output = model(video_feats, text_feats)
            labels = labels.type(torch.int).cuda()
            val_metrics.update(preds=output, target=labels)
    return list(val_metrics.compute().values())


def iterate(dataloader):
    for data_batch, label_batch in tqdm(dataloader):
        yield process_batch(data_batch, label_batch)


def process_batch(data_batch, label_batch):
    hypotheses, labels = [], []
    video_feat_batch, text_feat_batch = [], []
    for filepath, label in zip(data_batch, label_batch):
        labels.append(float(label))
        # ============ loading trajectory data ============ #
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))

        # ============ sampling frames from the video ============ #
        video_frames = sample_vid(filepath, args.sample_rate)
        video_frames = torch.stack(video_frames).cuda() # [t, c, h, w]
        t, c, h, w = video_frames.shape
        # b, c, t, h, w = video_frames.shape
        num_segments = math.ceil(t / 30)  # (VideoCLIP is trained on 30 fps of S3D)
        to_pad = num_segments * 30 - t
        video_frames = torch.cat((video_frames, torch.zeros(to_pad, c, h, w).cuda()), dim=0)
        video_frames = video_frames.permute(0, 2, 3, 1)
        video_frames = video_frames.reshape(num_segments, 30, h, w, c)

        # ============ process natural language hypothesis ============ #
        if label == '0':
            clean_string = ' '.join([clean_str(word).lower() for word in traj['template']['neg'].split(' ')])
        else:
            clean_string = ' '.join([clean_str(word).lower() for word in traj['template']['pos'].split(' ')])
        caps, cmasks = aligner._build_text_seq(
            tokenizer(clean_string, add_special_tokens=False)["input_ids"])
        caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

        # ========= getting (video_pooled, text_pooled) output ========= #
        with torch.no_grad():
            output = videoclip_backbone(video_frames.unsqueeze(0), caps.cuda(), cmasks.cuda(), return_score=False)
        video_feat_batch.append(output[0])
        text_feat_batch.append(output[1])

    # video_feat_batch: [batch_size, 768]
    # text_feat_batch: [batch_size, 768]
    return torch.stack(video_feat_batch), torch.stack(text_feat_batch), torch.tensor(labels).cuda()


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
    if args.split_type == 'train':
        path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
    else:
        path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.split_type)
    ckpt_file = 'videoclip_best_{}.pth'.format(str(args.run_id))
    model_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    logger_filename = 'videoclip_log_{}.txt'.format(str(args.run_id))
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

    # VideoCLIP backbone
    videoclip_backbone, tokenizer, aligner = \
        MMPTModel.from_pretrained("projects/retri/videoclip/how2.yaml")
    videoclip_backbone.cuda()
    videoclip_backbone = DDP(videoclip_backbone, device_ids=[local_rank])
    videoclip_backbone.eval()

    # trainable layer on top of videoclip backbone
    model = VideoClipModel(embed_size=768)
    if args.resume:  # to resume from a previously stored checkpoint
        model.load_state_dict(torch.load(model_ckpt_path))
    model.cuda()
    model = DDP(model, device_ids=[local_rank])

    all_params = list(model.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCELoss()
    metrics = MetricCollection([Accuracy(dist_sync_on_step=True, task='binary'),
                                F1Score(dist_sync_on_step=True, task='binary')]).cuda()
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
