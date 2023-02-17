import os
import sys
os.environ['DATA_ROOT'] = '/mnt/c/Users/rishihazra/PycharmProjects/VisionLangaugeGrounding/alfred/gen/dataset'
os.environ['BASELINES'] = '/mnt/c/Users/rishihazra/PycharmProjects/VisionLangaugeGrounding/baselines'
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from dataset_utils import *
from baselines.VIOLIN.violin_model import ViolinBase
from i3d.pytorch_i3d import InceptionI3d
from mvit_tx.mvit import mvit_v2_s
# from maskRCNN.mrcnn import load_pretrained_model
from arguments import Arguments
import math
import json
import numpy as np
from tqdm import tqdm
import torch
# import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, F1Score
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data.distributed import DistributedSampler
from torchvision.models import resnet18 as resnet
from transformers import DistilBertModel, DistilBertTokenizer
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe


def train_epoch(model, train_loader, val_loader, epoch, previous_best_acc):
    model.train()
    if args.finetune:
        visual_model.train()
    train_loss = []
    for video_features, hypotheses, labels in tqdm(iterate(train_loader), desc='Train'):
        output = model(video_features, hypotheses)
        loss = bce_loss(output, labels)
        optimizer.zero_grad()
        loss.backward()
        # model.state_dict(keep_vars=True)
        # {k: v.grad for k, v in zip(model.state_dict(), model.parameters())}
        optimizer.step()
        train_loss.append(loss.item())
        labels = labels.type(torch.int)
        train_metrics.update(preds=output, target=labels)

    acc, f1 = list(train_metrics.compute().values())
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
        print('============== Saving best model ================')
        torch.save(model.state_dict(), model_ckpt_path)
    log_file.flush()
    return previous_best_acc


def validate(model, val_loader):
    model.eval()
    visual_model.eval()
    with torch.no_grad():
        for video_features, hypotheses, labels in tqdm(iterate(val_loader), desc='Validation'):
            output = model(video_features, hypotheses)
            labels = labels.type(torch.int)
            val_metrics.update(preds=output, target=labels)
    return list(val_metrics.compute().values())


def iterate(dataloader):
    for data_batch, label_batch in dataloader:
        yield process_batch(data_batch, label_batch)


def process_batch(data_batch, label_batch):
    hypotheses = []
    video_features_batch = []  # transforms + visual model features
    labels = []
    vid_lengths = []
    for filepath, label in zip(data_batch, label_batch):
        labels.append(float(label))
        # loading trajectory data
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))

        # sampling frames from video
        # video_frames = [cv2.imread(frame) for frame in glob.glob(os.path.join(filepath, 'raw_images') + "/*.png")]
        video_frames = sample_vid(filepath, args.sample_rate)
        video_frames = torch.stack(video_frames) # [t, c, h, w]
        # process video features using resnet/I3D
        if args.visual_feature_extractor == 'resnet':
            # vid_lengths.append(len(video_frames))
            if not args.finetune:
                with torch.no_grad():
                    video_segments = visual_model(video_frames).view(-1, vid_feat_size)
            else:
                video_segments = visual_model(video_frames).view(-1, vid_feat_size)
            video_features_batch.append(video_segments)  # [t, 512]
            vid_lengths.append(len(video_segments))
        elif args.visual_feature_extractor == 'I3D':
            # obtaining action-segment information
            #assert len(traj['images']) == len(video_frames) - 10
            # vid_seg_changepoints = []
            # prev_high_idx = 0
            # for high_idx, img_dict in enumerate(traj['images']):
            #     if img_dict['high_idx'] == prev_high_idx + 1:
            #         vid_seg_changepoints.append(high_idx)
            #         prev_high_idx += 1
            #         # TODO: handle corner cases
            #         if (len(vid_seg_changepoints) > 1 and vid_seg_changepoints[-1] == vid_seg_changepoints[-2] + 1) or \
            #                 (len(vid_seg_changepoints) == 1 and vid_seg_changepoints[0] == 1):
            #             vid_seg_changepoints[-1] += 1
            # # adding the index of the last frame of video
            # if len(video_frames) > vid_seg_changepoints[-1] + 1:
            #     vid_seg_changepoints.append(len(video_frames))
            video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [b, c, t, h, w]
            # video_segments = []
            # s_ind, e_ind = 0, 0  # start_index, end_index
            # # decomposing the video into segments and extracting segment features using I3D
            # for vid_seg_changepoint in vid_seg_changepoints:
            #     s_ind = e_ind
            #     e_ind = vid_seg_changepoint
            if not args.finetune:
                with torch.no_grad():
                    video_segments = visual_model.extract_features(
                        video_frames).view(-1, vid_feat_size)
            else:
                video_segments = visual_model.extract_features(
                    video_frames).view(-1, vid_feat_size)
            # aggregate video features (averaging or RNN)
            # if not args.attention:
            #     if args.i3d_aggregate == 'rnn':  # using RNN to aggregate
            #         # pad_sequence(video_segments) -> [batch_size, max_seq_len over sub-segments in a single video, 1024]
            #         _, video_segments = segment_aggregator(pad_sequence(video_segments, batch_first=True),
            #                                             torch.tensor([video_segment.shape[0]
            #                                                           for video_segment in video_segments]))
            #     else:  # averaging all vectors for a segment
            #         video_segments = \
            #             torch.stack([video_segment.mean(dim=0) for video_segment in video_segments])
            video_features_batch.append(video_segments)
            vid_lengths.append(len(video_segments))
        elif args.visual_feature_extractor == 'mvit':
            # here b=1 since we are processing one video at a time
            video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [b, c, t, h, w]
            b, c, t, h , w = video_frames.shape
            num_segments = math.ceil(t / 16)
            to_pad = num_segments*16 - t
            video_frames = torch.cat((video_frames, torch.zeros(b, c, to_pad, h, w)), dim=2)
            video_frames = video_frames.reshape(b * num_segments, c, 16, h, w)
            if not args.finetune:
                with torch.no_grad():
                    video_segments = visual_model(video_frames).reshape(b*num_segments, embed_size)  # [num_segments, 768]
            else:
                video_segments = visual_model(video_frames).reshape(b*num_segments, embed_size)  # [num_segments, 768]
            video_features_batch.append(video_segments)
            vid_lengths.append(len(video_segments))

        # process natural language hypothesis using bert
        if label == '0':
            clean_string = ' '.join([clean_str(word).lower() for word in traj['template']['neg'].split(' ')])
        else:
            clean_string = ' '.join([clean_str(word).lower() for word in traj['template']['pos'].split(' ')])
        hypotheses.append(clean_string)

    with torch.no_grad():
        # last_hidden_state: [batch_size, max_seq_len, 768/300]
        tokenizer_out = tokenize_and_pad(hypotheses, tokenizer, args.text_feature_extractor)
        if args.text_feature_extractor == 'bert':
            hypotheses = (text_model(**dictfilt(tokenizer_out.data, ("input_ids", "attention_mask"))).last_hidden_state,
                          tokenizer_out.data['length'])
        elif args.text_feature_extractor == 'glove':
            hypotheses = (pad_sequence([global_vectors.get_vecs_by_tokens(x)
                                       for x in tokenizer_out]).permute(1, 0, 2),
                          torch.tensor([len(x) for x in tokenizer_out]))
    # pad_sequence(video_frames_batch) -> [batch_size, max_seq_len, embed_dim]
    video_features_batch = (pad_sequence(video_features_batch).permute(1, 0, 2).contiguous(), torch.tensor(vid_lengths))
    return video_features_batch, hypotheses, torch.tensor(labels)


if __name__ == '__main__':
    args = Arguments()
    if args.split_type == 'train':
        path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
    else:
        path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
    model_ckpt_path = os.path.join(os.getcwd(), "violin_best_valid.pth")
    hsize1 = 150
    hsize2 = 300

    if args.attention:
        logger_filename = 'violin_' + args.visual_feature_extractor + '_' + \
                          args.text_feature_extractor + '_attention_log.txt'
    else:
        logger_filename = 'violin_' + args.visual_feature_extractor + '_' + \
                          args.text_feature_extractor + '_log.txt'

    logger_path = os.path.join(os.environ['BASELINES'], logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')
    # assert args.batch_size % args.num_workers == 0, "batch_size should be divisible by num_workers"

    if args.preprocess:
        preprocess_dataset(path, args.split_type)

    dataset = CustomDataset(data_path=path)
    train_size = int(args.data_split * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    # train_sampler, val_sampler = DistributedSampler(dataset=train_set, shuffle=True), \
    #                              DistributedSampler(dataset=val_set, shuffle=True)
    train_loader, val_loader = DataLoader(train_set, batch_size=2,
                                          num_workers=args.num_workers, pin_memory=True), \
                               DataLoader(val_set, batch_size=2,
                                          num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # text feature extractor for the hypothesis
    if args.text_feature_extractor == 'bert':
        embed_size = 768
        # TODO: also try TinyBert
        text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # bert_model = DDP(bert_model, device_ids=[local_rank])
        # TODO: get vocabulary file for bert tokenizer: BertTokenizer(vocab_file=?)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        text_model.eval()
    elif args.text_feature_extractor == 'glove':
        embed_size = 300
        tokenizer = get_tokenizer("basic_english")
        global_vectors = GloVe(name='840B', dim=embed_size)
    else:
        raise NotImplementedError

    # visual feature extractor for the video
    if args.visual_feature_extractor == 'resnet':
        # resnet model
        vid_feat_size = 512  # 512 for resnet 18, 34; 2048 for resnet50, 101
        visual_model = resnet(pretrained=True)
        visual_model = nn.Sequential(*list(visual_model.children())[:-1])
    elif args.visual_feature_extractor == 'I3D':
        # I3D model
        vid_feat_size = 1024
        kinetics_pretrained = 'I3D/rgb_imagenet.pt'
        visual_model = InceptionI3d(400, in_channels=3)
        visual_model.load_state_dict(torch.load(os.path.join(os.environ['BASELINES'], kinetics_pretrained)))
        visual_model.replace_logits(157)
    elif args.visual_feature_extractor == 'mvit':
        # MViT ("https://arxiv.org/pdf/2104.11227.pdf")
        # TODO: [shallow version, with pretrained, with aggregation]
        vid_feat_size = 768
        weights = 'KINETICS400_V1' if args.pretrained_mvit else None
        visual_model = mvit_v2_s(weights=weights)
    else:
        raise NotImplementedError

    if not args.finetune:
        visual_model.eval()

    # violin base model
    model = ViolinBase(hsize1=hsize1, hsize2=hsize2, embed_size=embed_size,
                       vid_feat_size=vid_feat_size, attention=args.attention)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    all_params = list(model.parameters())
    if args.finetune:
        all_params += list(visual_model.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCELoss()
    metrics = MetricCollection([Accuracy(threshold=0.5, task='binary'),
                                F1Score(threshold=0.5, task='binary')])
    train_metrics = metrics.clone(prefix='train_')
    val_metrics = metrics.clone(prefix='val_')

    best_acc = 0.
    for epoch in range(1, args.epochs+1):
        # train_loader.sampler.set_epoch(epoch)
        # val_loader.sampler.set_epoch(epoch)
        best_acc = train_epoch(model, train_loader, val_loader, epoch=epoch, previous_best_acc=best_acc)
        train_metrics.reset()
        val_metrics.reset()
    # cleanup()
    log_file.close()
    print('Done!')
