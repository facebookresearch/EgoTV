# pip install transformers
# pip install scikit-learn
# pip install torchmetrics
# pip install torchtext==0.12.0
import os
import sys
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from dataset_utils import *
from violin_base import ViolinBase
# from rnn import RNNEncoder
from i3d.pytorch_i3d import InceptionI3d
from s3d.s3d import S3D
import clip
from pathlib import Path
import glob

from arguments import Arguments
import cv2
import json
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, F1Score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import resnet18 as resnet
from transformers import DistilBertModel, DistilBertTokenizer
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe


def test_model(test_loader):
    with torch.no_grad():
        for video_features, hypotheses, labels in tqdm(iterate(test_loader), desc='Test'):
            output = model(video_features, hypotheses)
            labels = labels.type(torch.int).cuda()
            test_metrics.update(preds=output, target=labels)
    dist.barrier()
    test_acc, test_f1 = test_metrics['Accuracy'].compute(), test_metrics['F1Score'].compute()
    dist.barrier()
    if is_main_process():
        print('Test Acc: {} | Test F1: {}'.format(test_acc, test_f1))
        log_file.write('Test Acc: ' + str(test_acc.item()) + ' | Test F1: ' + str(test_f1.item()) + "\n")
        log_file.flush()


def iterate(dataloader):
    for data_batch, label_batch in tqdm(dataloader):
        yield process_batch(data_batch, label_batch)


def process_batch(data_batch, label_batch):
    hypotheses = []
    video_features_batch = []  # transforms + visual model features
    labels = []
    vid_lengths = []
    for filepath, label in zip(data_batch, label_batch):
        labels.append(float(label))
        # loading trajectory data
        filepath = filepath.replace('/fb-agios-acai-efs/dataset/','/nobackup/projects/public/howto100m/datasets/Thor/')
        
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
            with torch.no_grad():
                video_features_batch.append(video_model(video_frames).view(-1, vid_feat_size))  # [t, 512]
        elif args.visual_feature_extractor == 'i3d':
            video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [b, c, t, h, w]
            with torch.no_grad():  # [t, 1024]
                video_segments = video_model.module.extract_features(
                    video_frames).view(-1, vid_feat_size)
            # # obtaining action-segment information
            # # assert len(traj['images']) == len(video_frames) - 10
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
            # video_frames = video_frames.unsqueeze(0).permute(0,2,1,3,4).contiguous()  # [b, c, t, h, w]
            # video_segments = []
            # s_ind, e_ind = 0, 0  # start_index, end_index
            # # decomposing the video into segments and extracting segment features using I3D
            # for vid_seg_changepoint in vid_seg_changepoints:
            #     s_ind = e_ind
            #     e_ind = vid_seg_changepoint
            #     with torch.no_grad():
            #         video_segments.append(video_model.module.extract_features(
            #             video_frames[:, :, s_ind:e_ind, :, :].contiguous()).view(-1, vid_feat_size))
            # # aggregate video features (averaging or RNN)
            # if args.i3d_aggregate == 'rnn':  # using RNN to aggregate
            #     # pad_sequence(video_segments) -> [batch_size, max_seq_len over sub-segments in a single video, 1024]
            #     _, video_segments = segment_aggregator(pad_sequence(video_segments, batch_first=True),
            #                                            torch.tensor([video_segment.shape[0]
            #                                                          for video_segment in video_segments]).cuda())
            # else:  # averaging all vectors for a segment
            #     video_segments = \
            #         torch.stack([video_segment.mean(dim=0) for video_segment in video_segments])
            video_features_batch.append(video_segments)
            vid_lengths.append(len(video_segments))
        elif args.visual_feature_extractor == 's3d':
            video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [b, c, t, h, w]
            if not args.finetune:
                with torch.no_grad():    # [t, 1024]
                    video_segments = video_model.module.extract_features(
                        video_frames).view(-1, vid_feat_size)
            else:
                video_segments = video_model.module.extract_features(
                    video_frames).view(-1, vid_feat_size)

            video_features_batch.append(video_segments)
            vid_lengths.append(len(video_segments))

        elif args.visual_feature_extractor == 'clip':
            video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [b, c, t, h, w]
            image = video_frames.permute(0, 2, 1,3, 4).half() 
            image = torch.flatten(image, start_dim=0, end_dim=1)
            if not args.finetune:
                with torch.no_grad():    # [t, 1024]
                    video_segments =  video_model.visual.forward(image).view(-1, vid_feat_size).float()
                    #video_segments = video_model.module.extract_features(
                    #    video_frames).view(-1, vid_feat_size)
            else:
                video_segments =  video_model.visual.forward(image).view(-1, vid_feat_size).float()

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

        elif args.text_feature_extractor == 'clip':
            input_x = clip.tokenize(hypotheses, truncate=True).cuda()
            x = video_model.token_embedding(input_x).type(
            video_model.dtype)  # [batch_size, n_ctx, d_model]
            x = x + video_model.positional_embedding.type(video_model.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = video_model.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

            x = video_model.ln_final(x).type(video_model.dtype)
        
            x = x @ video_model.text_projection

            batch_size, _, dim = x.shape
            prev_n_tokens = 20#data['text'].shape[1]

            input_x = input_x[:, 1:]  # first token is a token of beginning of the sentence
            x = x[:, 1:]  # first token is a token of beginning of the sentence

            new_text = x[:, :prev_n_tokens] #20 is max?

            new_text_length = torch.zeros(batch_size).cuda()

            for i in range(len(input_x)):
                # take features from the eot embedding (eot_token is the highest number in each sequence)
                n_eot = input_x[i].argmax().item()
                new_text_length[i] = min(n_eot,prev_n_tokens)

            hypotheses = (new_text.float(),
                          new_text_length)
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
                          args.text_feature_extractor + '_attention_log_test_' + str(args.run_id) + '.txt'
    else:
        logger_filename = 'violin_' + args.visual_feature_extractor + '_' + \
                          args.text_feature_extractor + '_log_test_' + str(args.run_id) + '.txt'

    logger_path = os.path.join(os.environ['BASELINES'], logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')

    if args.preprocess:
        preprocess_dataset(path, args.split_type)

    test_set = CustomDataset(data_path=path)
    test_sampler = DistributedSampler(dataset=test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler,
                                          num_workers=args.num_workers, shuffle=False)

    # bert base model
    # TODO: also try TinyBert
    if args.text_feature_extractor == 'bert':
        embed_size = 768
        bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").cuda()
        # bert_model = nn.SyncBatchNorm.convert_sync_batchnorm(bert_model) # Convert BatchNorm to SyncBatchNorm
        bert_model = DDP(bert_model, device_ids=[local_rank])
        # TODO: get vocabulary file for bert tokenizer: BertTokenizer(vocab_file=?)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        bert_model.eval()
    elif args.text_feature_extractor == 'glove':
        embed_size = 300
        tokenizer = get_tokenizer("basic_english")
        global_vectors = GloVe(name='840B', dim=embed_size)
    elif  args.text_feature_extractor == 'clip':
        embed_size = 512
        device = "cuda" if torch.cuda.is_available() else "cpu"
        video_model, preprocess = clip.load("ViT-B/32", device=device)
        video_model.eval()
        tokenizer = get_tokenizer("basic_english")
        #global_vectors = GloVe(name='840B', dim=embed_size)
    else:
        raise NotImplementedError

    # visual feature extractor for the video
    if args.visual_feature_extractor == 'resnet':
        # resnet model
        vid_feat_size = 512  # 512 for resnet 18, 34; 2048 for resnet50, 101
        if not args.finetune:
            video_model = resnet(pretrained=True)
        else:
            visual_model_ckpt_path = os.path.join(os.getcwd(), "resnet.pth")
            video_model = resnet()
            video_model.load_state_dict(torch.load(visual_model_ckpt_path))
        

        video_model = nn.Sequential(*list(video_model.children())[:-1]).cuda()
        video_model = nn.SyncBatchNorm.convert_sync_batchnorm(video_model)
        video_model = DDP(video_model, device_ids=[local_rank])
        video_model.eval()
    elif args.visual_feature_extractor == 'i3d':
        # i3d model
        vid_feat_size = 1024
        video_model = InceptionI3d(400, in_channels=3)
        if not args.finetune:
            visual_model_ckpt_path = os.path.join(os.environ['BASELINES'], 'i3d/rgb_imagenet.pt')
        else:
            visual_model_ckpt_path = os.path.join(os.getcwd(), "i3d.pth")
        video_model.load_state_dict(torch.load(visual_model_ckpt_path))

        video_model.replace_logits(157)
        video_model.cuda()
        video_model = nn.SyncBatchNorm.convert_sync_batchnorm(video_model)
        video_model = DDP(video_model, device_ids=[local_rank])
        video_model.eval()
        # if args.i3d_aggregate == 'rnn':
        #     aggregator_model_ckpt_path = os.path.join(os.getcwd(), "segment_aggregator.pth")
        #     segment_aggregator = \
        #         RNNEncoder(vid_feat_size, vid_feat_size, bidirectional=False, dropout_p=0, n_layers=1,
        #                    rnn_type='lstm')
        #     segment_aggregator.load_state_dict(torch.load(aggregator_model_ckpt_path))
        #     segment_aggregator.cuda()
        #     segment_aggregator = DDP(segment_aggregator, device_ids=[local_rank])
        #     segment_aggregator.eval()
    elif args.visual_feature_extractor == 's3d':
        # s3d model
        vid_feat_size = 1024
        pretrain_path = args.pretrain_path#'s3d/S3D_kinetics400.pt'
        print('pretrain_path',pretrain_path)
        #ssh://satori/nobackup/users/brian27/ECCV22/mil_nce/S3D_HowTo100M/s3d_howto100m.pth
        if 'howto100' in pretrain_path:
            from s3dg import S3D
            video_model = S3D(
            512, space_to_depth=True)
        else:
            video_model = S3D(400)
        #video_model.replace_logits(157)
        video_model.cuda()
        video_model = nn.SyncBatchNorm.convert_sync_batchnorm(video_model)
        video_model = DDP(video_model, device_ids=[local_rank])

        if 'howto100' in pretrain_path:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            checkpoint_module = {'module.' + k:v for k,v in checkpoint.items()}
            video_model.load_state_dict(checkpoint_module,strict=False)
            
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm3d') != -1:
                    print('classname',classname)
                    m.eval()
            #if args.fix_bn:
            print('fix_bn using')
            video_model.apply(set_bn_eval)
        else:
            
            video_model.load_state_dict(torch.load(os.path.join(os.environ['BASELINES'], pretrain_path)))
        
        if not args.finetune:
            video_model.eval()
        else:
            visual_model_ckpt_path = os.path.join(os.getcwd(), "s3d.pth")
    elif args.visual_feature_extractor == 'clip':
        # clip model
        vid_feat_size = 512
        print('use CLIP')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        video_model, preprocess = clip.load("ViT-B/32", device=device)

        #kinetics_pretrained = 's3d/S3D_kinetics400.pt'
        #video_model = S3D(400)
        #video_model.replace_logits(157)
        video_model.cuda()
        #video_model = nn.SyncBatchNorm.convert_sync_batchnorm(video_model)
        #video_model = DDP(video_model, device_ids=[local_rank])
        #video_model.load_state_dict(torch.load(os.path.join(os.environ['BASELINES'], kinetics_pretrained)))
        
        if not args.finetune:
            video_model.eval()
        else:
            visual_model_ckpt_path = os.path.join(os.getcwd(), "clip.pth")
    else:
        raise NotImplementedError

    # violin base model
    model = ViolinBase(hsize1=hsize1, hsize2=hsize2, embed_size=embed_size, vid_feat_size=vid_feat_size,
                       attention=args.attention)
    model.load_state_dict(torch.load(model_ckpt_path))
    model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])
    model.eval()
    metrics = MetricCollection([Accuracy(threshold=0.5, dist_sync_on_step=True),
                                F1Score(threshold=0.5, dist_sync_on_step=True)]).cuda()
    test_metrics = metrics.clone(prefix='test_')

    test_loader.sampler.set_epoch(0)
    test_model(test_loader)
    test_metrics.reset()
    cleanup()
    log_file.close()
    print('Done!')
