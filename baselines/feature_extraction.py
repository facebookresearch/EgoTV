from dataset_utils import tokenize_and_pad, dictfilt
import os
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from i3d.pytorch_i3d import InceptionI3d
from mvit_tx.mvit import mvit_v2_s
from torchvision.models import resnet18 as resnet
from transformers import DistilBertModel, DistilBertTokenizer
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe


def initiate_visual_module(feature_extractor, pretrained_mvit=True):
    # visual feature extractor for the video
    if feature_extractor == 'resnet':
        # resnet model
        vid_feat_size = 512  # 512 for resnet 18, 34; 2048 for resnet50, 101
        visual_model = resnet(pretrained=True)
        visual_model = nn.Sequential(*list(visual_model.children())[:-1])
    elif feature_extractor == 'i3d':
        # i3d model
        vid_feat_size = 1024
        kinetics_pretrained = 'i3d/rgb_imagenet.pt'
        visual_model = InceptionI3d(400, in_channels=3)
        visual_model.load_state_dict(torch.load(os.path.join(os.environ['BASELINES'], kinetics_pretrained)))
        visual_model.replace_logits(157)
    elif feature_extractor == 'mvit':
        # MViT-S ("https://arxiv.org/pdf/2104.11227.pdf")
        vid_feat_size = 768
        weights = 'KINETICS400_V1' if pretrained_mvit else None
        visual_model = mvit_v2_s(weights=weights)
    else:
        raise NotImplementedError
    return visual_model, vid_feat_size

def initiate_text_module(feature_extractor):
    # text feature extractor for the hypothesis
    if feature_extractor == 'bert':
        # distil bert model
        embed_size = 768
        text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # transformers use layer norm (and not batch norm) which is local -- no need to sync across all instances
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        # text_model.eval()
    elif feature_extractor == 'glove':
        embed_size = 300
        tokenizer = get_tokenizer("basic_english")
        text_model = GloVe(name='840B', dim=embed_size)
    else:
        raise NotImplementedError
    return text_model, tokenizer, embed_size


def extract_video_features(video_frames, model, feature_extractor, feat_size, finetune=False, test=False):
    # video_features_batch = []  # transforms + visual model features
    # vid_lengths = []
    if feature_extractor == 'resnet':
        # vid_lengths.append(len(video_frames))
        if not finetune or test:
            with torch.no_grad():
                video_feats = model(video_frames).view(-1, feat_size)  # [t, 512]
        else:
            video_feats = model(video_frames).view(-1, feat_size)
    elif feature_extractor == 'i3d':
        video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [b, c, t, h, w]
        if not finetune or test:
            with torch.no_grad():  # [t, 1024]
                video_feats = model.module.extract_features(
                    video_frames).view(-1, feat_size)
        else:
            video_feats = model.module.extract_features(
                video_frames).view(-1, feat_size)

        # video_features_batch.append(video_segments)
        # vid_lengths.append(len(video_segments))
    elif feature_extractor == 'mvit':
        # here b=1 since we are processing one video at a time
        video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [b, c, t, h, w]
        b, c, t, h, w = video_frames.shape
        num_segments = math.ceil(t / 16)
        to_pad = num_segments * 16 - t
        video_frames = torch.cat((video_frames, torch.zeros(b, c, to_pad, h, w).cuda()), dim=2)
        video_frames = video_frames.reshape(b * num_segments, c, 16, h, w)
        if not finetune or test:
            with torch.no_grad():
                video_feats = model(video_frames).reshape(b * num_segments, feat_size)  # [num_segments, 768]
        else:
            video_feats = model(video_frames).reshape(b * num_segments, feat_size)  # [num_segments, 768]
        # video_features_batch.append(video_segments)
        # vid_lengths.append(len(video_segments))

    # pad_sequence(video_frames_batch) -> [batch_size, max_seq_len, embed_dim]
    # video_features_batch = (pad_sequence(video_features_batch).permute(1, 0, 2).contiguous(),
    #                         torch.tensor(vid_lengths).cuda())
    return video_feats


def extract_text_features(hypotheses, model, feature_extractor, tokenizer):
    with torch.no_grad():
        # last_hidden_state: [batch_size, max_seq_len, embed_dim]
        tokenizer_out = tokenize_and_pad(hypotheses, tokenizer, feature_extractor)
        if feature_extractor == 'bert':
            text_feats = (model(**dictfilt(tokenizer_out.data, ("input_ids", "attention_mask"))).last_hidden_state,
                          tokenizer_out.data['length'])
        elif feature_extractor == 'glove':
            text_feats = (pad_sequence([model.get_vecs_by_tokens(x).cuda()
                                        for x in tokenizer_out]).permute(1, 0, 2),
                          torch.tensor([len(x) for x in tokenizer_out]).cuda())
    return text_feats
