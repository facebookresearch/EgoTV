# pip install transformers
# pip install scikit-learn
# pip install torchmetrics
# pip install torchtext==0.10.0
import os
import sys
os.environ['DATA_ROOT'] = '/mnt/c/Users/rishihazra/PycharmProjects/VisionLangaugeGrounding/alfred/gen/dataset'
os.environ['BASELINES'] = '/mnt/c/Users/rishihazra/PycharmProjects/VisionLangaugeGrounding/baselines'
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from dataset_utils import *
from violin_base import ViolinBase
from rnn import RNNEncoder
from i3d.pytorch_i3d import InceptionI3d
from maskRCNN.mrcnn import load_pretrained_model
from arguments import Arguments
import cv2
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
        if args.visual_feature_extractor == 'resnet':
            resnet_model.train()
        elif args.visual_feature_extractor == 'i3d':
            i3d_model.train()
            if args.i3d_aggregate == 'rnn':
                segment_aggregator.train()
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
        print('============== Saving best model ================')
        torch.save(model.state_dict(), model_ckpt_path)
    log_file.flush()
    return previous_best_acc


def validate(model, val_loader):
    model.eval()
    if args.visual_feature_extractor == 'resnet':
        resnet_model.eval()
    elif args.visual_feature_extractor == 'i3d':
        i3d_model.eval()
        if args.i3d_aggregate == 'rnn':
            segment_aggregator.eval()
    with torch.no_grad():
        for video_features, hypotheses, labels in tqdm(iterate(val_loader), desc='Validation'):
            output = model(video_features, hypotheses)
            labels = labels.type(torch.int)
            val_metrics.update(preds=output, target=labels)
    return val_metrics['Accuracy'].compute(), val_metrics['F1Score'].compute()


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
        video_frames = []
        video = cv2.VideoCapture(os.path.join(filepath, 'video.mp4'))
        success = video.grab()
        fno = 0
        while success:
            if fno % args.sample_rate == 0:
                _, img = video.retrieve()
                video_frames.append(transform_image(img))
            success = video.grab()
            fno += 1
        try:
            video_frames = torch.stack(video_frames) # [t, c, h, w]
        except:
            print(filepath)

        # process video features using resnet/i3d
        if args.visual_feature_extractor == 'resnet':
            vid_lengths.append(len(video_frames))
            if not args.finetune:
                with torch.no_grad():
                    #mrcnn_model([video_frames[0]])
                    video_features_batch.append(resnet_model(video_frames).view(-1, vid_feat_size))  # [t, 512]
            else:
                video_features_batch.append(resnet_model(video_frames).view(-1, vid_feat_size))
        elif args.visual_feature_extractor == 'i3d':
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
            video_frames = video_frames.unsqueeze(0).permute(0,2,1,3,4).contiguous()  # [b, c, t, h, w]
            # video_segments = []
            # s_ind, e_ind = 0, 0  # start_index, end_index
            # # decomposing the video into segments and extracting segment features using I3D
            # for vid_seg_changepoint in vid_seg_changepoints:
            #     s_ind = e_ind
            #     e_ind = vid_seg_changepoint
            if not args.finetune:
                with torch.no_grad():
                    video_segments = i3d_model.extract_features(
                        video_frames).view(-1, vid_feat_size)
            else:
                video_segments = i3d_model.extract_features(
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
            hypotheses = (bert_model(**dictfilt(tokenizer_out.data, ("input_ids", "attention_mask"))).last_hidden_state,
                          tokenizer_out.data['length'])
        elif args.text_feature_extractor == 'glove':
            hypotheses = (pad_sequence([global_vectors.get_vecs_by_tokens(x)
                                       for x in tokenizer_out]).permute(1, 0, 2),
                          torch.tensor([len(x) for x in tokenizer_out]))
    # pad_sequence(video_frames_batch) -> [max_seq_len, batch_size, resnet_dim/i3d_dim]
    video_features_batch = (pad_sequence(video_features_batch), torch.tensor(vid_lengths))
    return video_features_batch, hypotheses, torch.tensor(labels)


features = []
def save_features(mod, inp, outp):
    features.append(outp)

if __name__ == '__main__':
    # dist_url = "env://"  # default
    # # only works with torch.distributed.launch // torch.run
    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ['WORLD_SIZE'])
    # local_rank = int(os.environ['LOCAL_RANK'])
    # dist.init_process_group(
    #     backend="nccl",
    #     init_method=dist_url,
    #     world_size=world_size,
    #     rank=rank)
    # # this will make all .cuda() calls work properly
    # torch.cuda.set_device(local_rank)
    # # synchronizes all the threads to reach this point before moving on
    # dist.barrier()
    # mrcnn_pth = os.path.join(os.environ['BASELINES'], 'maskRCNN/mrcnn_alfred_all_004.pth')
    # mrcnn_model = load_pretrained_model(mrcnn_pth, num_classes=106)
    # you can also hook layers inside the roi_heads
    # layer_to_hook = 'roi_heads'
    # mrcnn_model.roi_heads.box_head.fc7.register_forward_hook(save_features)
    # mrcnn_model.roi_heads.box_predictor.cls_score.register_forward_hook(save_features)
    # # mrcnn_model.roi_heads.box_predictor.bbox_pred.register_forward_hook(save_features)
    # mrcnn_model.eval()

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
    train_loader, val_loader = DataLoader(train_set, batch_size=args.batch_size,
                                          num_workers=args.num_workers, pin_memory=True), \
                               DataLoader(val_set, batch_size=args.batch_size,
                                          num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # text feature extractor for the hypothesis
    if args.text_feature_extractor == 'bert':
        embed_size = 768
        # bert base model
        # TODO: also try TinyBert
        bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # bert_model = nn.SyncBatchNorm.convert_sync_batchnorm(bert_model) # Convert BatchNorm to SyncBatchNorm
        # bert_model = DDP(bert_model, device_ids=[local_rank])
        # TODO: get vocabulary file for bert tokenizer: BertTokenizer(vocab_file=?)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        bert_model.eval()
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
        resnet_model = resnet(pretrained=True)
        resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
        if not args.finetune:
            resnet_model.eval()
        # resnet_model = nn.SyncBatchNorm.convert_sync_batchnorm(resnet_model)
        # resnet_model = DDP(resnet_model, device_ids=[local_rank])
    elif args.visual_feature_extractor == 'i3d':
        # i3d model
        vid_feat_size = 1024
        kinetics_pretrained = 'i3d/rgb_imagenet.pt'
        i3d_model = InceptionI3d(400, in_channels=3)
        i3d_model.load_state_dict(torch.load(os.path.join(os.environ['BASELINES'], kinetics_pretrained)))
        i3d_model.replace_logits(157)
        if not args.finetune:
            i3d_model.eval()
        # i3d_model.cuda()
        # i3d_model = nn.SyncBatchNorm.convert_sync_batchnorm(i3d_model)
        # i3d_model = DDP(i3d_model, device_ids=[local_rank])
        if args.i3d_aggregate == 'rnn':
            segment_aggregator = \
                RNNEncoder(vid_feat_size, vid_feat_size, bidirectional=False, dropout_p=0, n_layers=1, rnn_type='lstm')
            # segment_aggregator.cuda()
            # segment_aggregator = nn.SyncBatchNorm.convert_sync_batchnorm(segment_aggregator)
            # segment_aggregator = DDP(segment_aggregator, device_ids=[local_rank])
    else:
        raise NotImplementedError

    # violin base model
    model = ViolinBase(hsize1=hsize1, hsize2=hsize2, embed_size=embed_size,
                       vid_feat_size=vid_feat_size, attention=args.attention)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    all_params = list(model.parameters())
    if args.finetune:
        if args.visual_feature_extractor == 'i3d':
            all_params += list(i3d_model.parameters())
            if args.i3d_aggregate == 'rnn':
                all_params += list(segment_aggregator.parameters())
        elif args.visual_feature_extractor == 'resnet':
            all_params += list(resnet_model.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCELoss()
    metrics = MetricCollection([Accuracy(threshold=0.5),
                                F1Score(threshold=0.5)])
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