import os
import sys
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])

from dataset_utils import *
from distributed_utils import *
from CLIP4Clip.clip4clip_model import CLIP4Clip
import json
from tqdm import tqdm
import clip
from arguments import Arguments
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from torchmetrics import MetricCollection, Accuracy, F1Score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def test_model(clip4clip_model, test_loader):
    with torch.no_grad():
        for video_feats, text_feats, labels in tqdm(iterate(test_loader), desc='Test'):
            output = clip4clip_model(video_feats, text_feats)
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
        video_frames = sample_vid(filepath, args.sample_rate)
        video_frames = torch.stack(video_frames).cuda() # [t, c, h, w]
        # ============ process image features using clip ============ #
        video_feats = clip_model.module.encode_image(video_frames).float() # [max_seq_len, batch_size, embed_dim]
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
    text_feat_batch = clip_model.module.encode_text(clip.tokenize(hypotheses, truncate=True).cuda())
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
    ckpt_file = 'clip4clip_{}_best_{}.pth'.format(args.split_type, str(args.run_id))
    model_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    logger_filename = 'clip4clip_{}_log_test_{}.txt'.format(args.split_type, str(args.run_id))
    logger_path = os.path.join(os.getcwd(), logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')

    if args.preprocess:
        preprocess_dataset(path, args.split_type)
    test_set = CustomDataset(data_path=path)
    test_sampler = DistributedSampler(dataset=test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler,
                             num_workers=args.num_workers, shuffle=False)

    # CLIP4Clip model (https://github.com/ArrowLuo/CLIP4Clip)
    clip_model, _ =  clip.load("ViT-B/32")
    clip_model.cuda()
    # clip_model = nn.SyncBatchNorm.convert_sync_batchnorm(clip_model)
    clip_model = DDP(clip_model, device_ids=[local_rank])
    if not args.finetune:
        clip_model.eval()
    else:
        clip_model_ckpt_path = os.path.join(os.getcwd(), "{}.pth".format('clip'))
        clip_model.load_state_dict(torch.load(clip_model_ckpt_path))

    # clip4clip baseline
    clip4clip_model = CLIP4Clip(embed_size=512, sim_type=args.sim_type)
    if args.sim_type in ['seqLSTM', 'tightTransfer']:
        clip4clip_model.load_state_dict(torch.load(model_ckpt_path))
    clip4clip_model.cuda()
    if args.sim_type in ['seqLSTM', 'tightTransfer']:
        clip4clip_model = DDP(clip4clip_model, device_ids=[local_rank])

    metrics = MetricCollection([Accuracy(dist_sync_on_step=True, task='binary'),
                                F1Score(dist_sync_on_step=True, task='binary')]).cuda()
    test_metrics = metrics.clone(prefix='test_')

    best_acc = 0.
    test_loader.sampler.set_epoch(0)
    test_model(clip4clip_model, test_loader)
    test_metrics.reset()
    cleanup()
    log_file.close()
    print('Done!')
