# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import sys
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])

from dataset_utils import *
from distributed_utils import *
from CoCa.coca_model import CocaVidModel
import open_clip
import json
from tqdm import tqdm
from arguments import Arguments
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from torchmetrics import MetricCollection, Accuracy, F1Score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def test_model(model, test_loader):
    with torch.no_grad():
        for video_feats, text_feats, labels in tqdm(iterate(test_loader), desc='Test'):
            output = model(video_feats, text_feats)
            labels = labels.type(torch.int).cuda()
            test_metrics.update(preds=output, target=labels)
        dist.barrier()
        test_acc, test_f1 = list(test_metrics.compute().values())
        dist.barrier()
        if is_main_process():
            print('Test Acc: {} | Test F1: {}'.format(test_acc, test_f1))
            log_file.write('Test Acc: ' + str(test_acc.item()) + ' | Test F1: ' + str(test_f1.item()) + "\n")
            log_file.flush()


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
        video_frames = [transform(img) for img in sample_vid(filepath, args.sample_rate, type='coca')]
        video_frames = torch.stack(video_frames).cuda()  # [t, c, h, w]
        # ============ process image features using clip ============ #
        with torch.no_grad():
            video_feats = coca_backbone.module.encode_image(video_frames).float() # [max_seq_len, batch_size, embed_dim]
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
        text_feat_batch = coca_backbone.module.encode_text(tokenizer(hypotheses).cuda())
    return video_feat_batch, text_feat_batch, torch.tensor(labels).cuda()


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
    path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.split_type)
    ckpt_file = 'coca_best_{}.pth'.format(str(args.run_id))
    model_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    logger_filename = 'coca_log_test_{}.txt'.format(args.sim_type, str(args.run_id))
    logger_path = os.path.join(os.getcwd(), logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')

    if args.preprocess:
        preprocess_dataset(path, args.split_type)
    test_set = CustomDataset(data_path=path)
    test_sampler = DistributedSampler(dataset=test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler,
                             num_workers=args.num_workers, shuffle=False)

    # CoCa (finetuned model from OpenCLIP)
    coca_backbone, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-B-32",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )
    coca_backbone.cuda()
    # clip_model = nn.SyncBatchNorm.convert_sync_batchnorm(clip_model)
    coca_backbone = DDP(coca_backbone, device_ids=[local_rank])
    if not args.finetune:
        coca_backbone.eval()
    else:
        coca_model_ckpt_path = os.path.join(os.getcwd(), "{}.pth".format('coca'))
        coca_backbone.load_state_dict(torch.load(coca_model_ckpt_path))
    tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

    # coca video baseline
    model = CocaVidModel(embed_size=512)
    model.load_state_dict(torch.load(model_ckpt_path))
    model.cuda()
    model = DDP(model, device_ids=[local_rank])

    metrics = MetricCollection([Accuracy(dist_sync_on_step=True, task='binary'),
                                F1Score(dist_sync_on_step=True, task='binary')]).cuda()
    test_metrics = metrics.clone(prefix='test_')

    best_acc = 0.
    test_loader.sampler.set_epoch(0)
    test_model(model, test_loader)
    test_metrics.reset()
    cleanup()
    log_file.close()
    print('Done!')
