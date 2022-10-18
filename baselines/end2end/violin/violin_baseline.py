# pip install transformers
# pip install scikit-learn
# pip install torchmetrics
# pip install torchtext==0.12.0
# sudo apt install graphviz
# pip install graphviz
# pip install pydot
# pip install sentencepiece
# import nltk
# nltk.download('punkt')
#
import os
import sys
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from dataset_utils import *
from violin_base import ViolinBase
# from rnn import RNNEncoder
from i3d.pytorch_i3d import InceptionI3d
from arguments import Arguments
import cv2
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, F1Score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
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
    # if args.visual_feature_extractor == 'i3d' and args.i3d_aggregate == 'rnn':
    #     segment_aggregator.train()
    train_loss = []
    for video_features, hypotheses, labels in tqdm(iterate(train_loader), desc='Train'):
        output = model(video_features, hypotheses)
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

    acc, f1 = train_metrics['Accuracy'].compute(), train_metrics['F1Score'].compute()
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
            if args.finetune:
                if args.visual_feature_extractor == 'resnet':
                    torch.save(resnet_model.module.state_dict(), visual_model_ckpt_path)
                elif args.visual_feature_extractor == 'i3d':
                    torch.save(i3d_model.module.state_dict(), visual_model_ckpt_path)
            # if args.visual_feature_extractor == 'i3d' and args.i3d_aggregate == 'rnn':
            #     torch.save(segment_aggregator.module.state_dict(), aggregator_model_ckpt_path)
        log_file.flush()
    return previous_best_acc


def validate(model, val_loader):
    model.eval()
    if args.visual_feature_extractor == 'resnet':
        resnet_model.eval()
    elif args.visual_feature_extractor == 'i3d':
        i3d_model.eval()
        # if args.i3d_aggregate == 'rnn':
        #     segment_aggregator.eval()
    with torch.no_grad():
        for video_features, hypotheses, labels in tqdm(iterate(val_loader), desc='Validation'):
            output = model(video_features, hypotheses)
            labels = labels.type(torch.int).cuda()
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
            video_frames = torch.stack(video_frames).cuda()  # [t, c, h, w]
        except:
            print(filepath)

        # process video features using resnet/i3d
        if args.visual_feature_extractor == 'resnet':
            vid_lengths.append(len(video_frames))
            if not args.finetune:
                with torch.no_grad():
                    video_features_batch.append(resnet_model(video_frames).view(-1, vid_feat_size))  # [t, 512]
            else:
                video_features_batch.append(resnet_model(video_frames).view(-1, vid_feat_size))
        elif args.visual_feature_extractor == 'i3d':
            video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [b, c, t, h, w]
            if not args.finetune:
                with torch.no_grad():    # [t, 1024]
                    video_segments = i3d_model.module.extract_features(
                        video_frames).view(-1, vid_feat_size)
            else:
                video_segments = i3d_model.module.extract_features(
                    video_frames).view(-1, vid_feat_size)
            # # obtaining action-segment information
            #             # # assert len(traj['images']) == len(video_frames) - 10
            #             # vid_seg_changepoints = []
            #             # prev_high_idx = 0
            #             # for high_idx, img_dict in enumerate(traj['images']):
            #             #     if img_dict['high_idx'] == prev_high_idx + 1:
            #             #         vid_seg_changepoints.append(high_idx)
            #             #         prev_high_idx += 1
            #             #         # TODO: handle corner cases
            #             #         if (len(vid_seg_changepoints) > 1 and vid_seg_changepoints[-1] == vid_seg_changepoints[-2] + 1) or \
            #             #                 (len(vid_seg_changepoints) == 1 and vid_seg_changepoints[0] == 1):
            #             #             vid_seg_changepoints[-1] += 1
            #             # # adding the index of the last frame of video
            #             # if len(video_frames) > vid_seg_changepoints[-1] + 1:
            #             #     vid_seg_changepoints.append(len(video_frames))
            #             # video_frames = video_frames.unsqueeze(0).permute(0,2,1,3,4).contiguous()  # [b, c, t, h, w]
            #             # video_segments = []
            #             # s_ind, e_ind = 0, 0  # start_index, end_index
            #             # # decomposing the video into segments and extracting segment features using I3D
            #             # for vid_seg_changepoint in vid_seg_changepoints:
            #             #     s_ind = e_ind
            #             #     e_ind = vid_seg_changepoint
            #             #     if not args.finetune:
            #             #         with torch.no_grad():
            #             #             video_segments.append(i3d_model.module.extract_features(
            #             #                 video_frames[:, :, s_ind:e_ind, :, :].contiguous()).view(-1, vid_feat_size))
            #             #     else:
            #             #         video_segments.append(i3d_model.module.extract_features(
            #             #             video_frames[:, :, s_ind:e_ind, :, :].contiguous()).view(-1, vid_feat_size))
            #             # # aggregate video features (averaging or RNN)
            #             # if args.i3d_aggregate == 'rnn':  # using RNN to aggregate
            #             #     # pad_sequence(video_segments) -> [batch_size, max_seq_len over sub-segments in a single video, 1024]
            #             #     _, video_segments = segment_aggregator(pad_sequence(video_segments, batch_first=True),
            #             #                                            torch.tensor([video_segment.shape[0]
            #             #                                                          for video_segment in video_segments]).cuda())
            #             # else:  # averaging all vectors for a segment
            #             #     video_segments = \
            #             #         torch.stack([video_segment.mean(dim=0) for video_segment in video_segments])
            video_features_batch.append(video_segments)
            vid_lengths.append(len(video_segments))

        # process natural language hypothesis using bert
        if label == '0':
            clean_string = ' '.join([clean_str(word).lower() for word in traj['template']['neg'].split(' ')])
        else:
            clean_string = ' '.join([clean_str(word).lower() for word in traj['template']['pos'].split(' ')])
        hypotheses.append(clean_string)

    with torch.no_grad():
        # last_hidden_state: [batch_size, max_seq_len, 768]
        tokenizer_out = tokenize_and_pad(hypotheses, tokenizer, args.text_feature_extractor)
        if args.text_feature_extractor == 'bert':
            hypotheses = (bert_model(**dictfilt(tokenizer_out.data, ("input_ids", "attention_mask"))).last_hidden_state,
                          tokenizer_out.data['length'])
        elif args.text_feature_extractor == 'glove':
            hypotheses = (pad_sequence([global_vectors.get_vecs_by_tokens(x).cuda()
                                        for x in tokenizer_out]).permute(1, 0, 2),
                          torch.tensor([len(x) for x in tokenizer_out]).cuda())
    # pad_sequence(video_frames_batch) -> [max_seq_len, batch_size, resnet_dim]
    video_features_batch = (pad_sequence(video_features_batch), torch.tensor(vid_lengths).cuda())
    return video_features_batch, hypotheses, torch.tensor(labels).cuda()


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


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
    print("============ attention is {} ============".format(str(args.attention)))
    if args.split_type == 'train':
        path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
    else:
        path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.split_type)
    if args.attention:
        ckpt_file = 'violin_' + args.visual_feature_extractor + '_' + \
                          args.text_feature_extractor + '_attention_best_' + str(args.run_id) + '.pth'
    else:
        ckpt_file = 'violin_' + args.visual_feature_extractor + '_' + \
                          args.text_feature_extractor + '_best_' + str(args.run_id) + '.pth'
    model_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    hsize1 = 150
    hsize2 = 300

    if args.attention:
        logger_filename = 'violin_' + args.visual_feature_extractor + '_' + \
                          args.text_feature_extractor + '_attention_log_' + str(args.run_id) + '.txt'
    else:
        logger_filename = 'violin_' + args.visual_feature_extractor + '_' + \
                          args.text_feature_extractor + '_log_' + str(args.run_id) + '.txt'

    logger_path = os.path.join(os.getcwd(), logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')

    if args.preprocess:
        #TODO: automate [if not present json_file, preprocess must be done]
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

    # distil bert model
    # TODO: also try TinyBert
    if args.text_feature_extractor == 'bert':
        embed_size = 768
        bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").cuda()
        # transformers use layer norm (and not batch norm) which is local -- no need to sync across all instances
        bert_model = DDP(bert_model, device_ids=[local_rank])
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
        resnet_model = nn.Sequential(*list(resnet_model.children())[:-1]).cuda()
        resnet_model = nn.SyncBatchNorm.convert_sync_batchnorm(resnet_model)
        resnet_model = DDP(resnet_model, device_ids=[local_rank])
        if not args.finetune:
            resnet_model.eval()
        else:
            visual_model_ckpt_path = os.path.join(os.getcwd(), "resnet.pth")
    elif args.visual_feature_extractor == 'i3d':
        # i3d model
        vid_feat_size = 1024
        kinetics_pretrained = 'i3d/rgb_imagenet.pt'
        i3d_model = InceptionI3d(400, in_channels=3)
        i3d_model.load_state_dict(torch.load(os.path.join(os.environ['BASELINES'], kinetics_pretrained)))
        i3d_model.replace_logits(157)
        i3d_model.cuda()
        i3d_model = nn.SyncBatchNorm.convert_sync_batchnorm(i3d_model)
        i3d_model = DDP(i3d_model, device_ids=[local_rank])
        if not args.finetune:
            i3d_model.eval()
        else:
            visual_model_ckpt_path = os.path.join(os.getcwd(), "i3d.pth")
        # if args.i3d_aggregate == 'rnn':
        #     segment_aggregator = \
        #         RNNEncoder(vid_feat_size, vid_feat_size, bidirectional=False, dropout_p=0, n_layers=1,
        #                    rnn_type='lstm').cuda()
        #     segment_aggregator = DDP(segment_aggregator, device_ids=[local_rank])
        #     aggregator_model_ckpt_path = os.path.join(os.getcwd(), "segment_aggregator.pth")
    else:
        raise NotImplementedError

    # violin base model
    model = ViolinBase(hsize1=hsize1, hsize2=hsize2, embed_size=embed_size, vid_feat_size=vid_feat_size,
                       attention=args.attention).cuda()
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])

    all_params = list(model.parameters())
    if args.finetune:
        if args.visual_feature_extractor == 'i3d':
            all_params += list(i3d_model.parameters())
            # if args.i3d_aggregate == 'rnn':
            #     all_params += list(segment_aggregator.parameters())
        elif args.visual_feature_extractor == 'resnet':
            all_params += list(resnet_model.parameters())
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
