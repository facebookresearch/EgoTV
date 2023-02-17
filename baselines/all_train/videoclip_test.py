import os
import sys
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])

from dataset_utils import *
from distributed_utils import *
from VideoCLIP.videoclip_model import VideoClipModel
from fairseq.MMPT.examples.mmpt.models import MMPTModel
import json
import math
from tqdm import tqdm
from arguments import Arguments
import torch
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
    video_feat_batch, text_feat_batch = [], []
    for filepath, label in zip(data_batch, label_batch):
        labels.append(float(label))
        # ============ loading trajectory data ============ #
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))

        # ============ sampling frames from the video ============ #
        video_frames = sample_vid(filepath, args.sample_rate)
        video_frames = torch.stack(video_frames).cuda() # [t, c, h, w]
        b, c, t, h, w = video_frames.shape
        num_segments = math.ceil(t / 30)  # (VideoCLIP is trained on 30 fps of S3D)
        to_pad = num_segments * 30 - t
        video_frames = torch.cat((video_frames, torch.zeros(b, c, to_pad, h, w).cuda()), dim=2)
        video_frames = video_frames.permute(b, t, h, w, c)
        video_frames = video_frames.reshape(b, num_segments, 30, h, w, c)

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
            output = videoclip_backbone(video_frames, caps, cmasks, return_score=False)
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
    path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.split_type)
    ckpt_file = 'videoclip_best_{}.pth'.format(str(args.run_id))
    model_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    logger_filename = 'videoclip_log_test_{}.txt'.format(args.sim_type, str(args.run_id))
    logger_path = os.path.join(os.getcwd(), logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')

    if args.preprocess:
        preprocess_dataset(path, args.split_type)
    test_set = CustomDataset(data_path=path)
    test_sampler = DistributedSampler(dataset=test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler,
                             num_workers=args.num_workers, shuffle=False)

    # VideoCLIP backbone
    videoclip_backbone, tokenizer, aligner = \
        MMPTModel.from_pretrained("projects/retri/videoclip/how2.yaml")
    videoclip_backbone.cuda()
    # clip_model = nn.SyncBatchNorm.convert_sync_batchnorm(clip_model)
    videoclip_backbone = DDP(videoclip_backbone, device_ids=[local_rank])
    videoclip_backbone.eval()

    # trainable layer on top of videoclip backbone
    model = VideoClipModel(embed_size=768)
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
