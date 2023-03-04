from dataset_utils import tokenize_and_pad, dictfilt
import os
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from I3D.pytorch_i3d import InceptionI3d
import clip
import open_clip
from S3D.s3d import S3D
from MViT.mvit import mvit_v2_s
from torchvision.models import resnet18 as resnet
from transformers import DistilBertModel, DistilBertTokenizer
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe


def initiate_visual_module(feature_extractor, pretrained_mvit=True):
    transform = None
    # visual feature extractor for the video
    if feature_extractor == 'resnet':
        # resnet model
        vid_feat_size = 512  # 512 for resnet 18, 34; 2048 for resnet50, 101
        visual_model = resnet(pretrained=True)
        visual_model = nn.Sequential(*list(visual_model.children())[:-1])
    elif feature_extractor == 'I3D':
        # I3D model
        vid_feat_size = 1024
        kinetics_pretrained = 'I3D/models/rgb_imagenet.pt'
        visual_model = InceptionI3d(400, in_channels=3)
        visual_model.load_state_dict(torch.load(os.path.join(os.environ['BASELINES'], kinetics_pretrained)))
        visual_model.replace_logits(157)
    elif feature_extractor == 'S3D':
        # S3D model
        vid_feat_size = 1024
        kinetics_pretrained = 'S3D/S3D_kinetics400.pt'
        visual_model = S3D(400)
        visual_model.load_state_dict(torch.load(os.path.join(os.environ['BASELINES'], kinetics_pretrained)))
    elif feature_extractor == 'mvit':
        # MViT-S ("https://arxiv.org/pdf/2104.11227.pdf")
        vid_feat_size = 768
        weights = 'KINETICS400_V1' if pretrained_mvit else None
        visual_model = mvit_v2_s(weights=weights)
    elif feature_extractor == 'clip':
        vid_feat_size = 512
        visual_model, _ = clip.load("ViT-B/32")
    elif feature_extractor == 'coca':
        vid_feat_size = 512
        visual_model, _, transform = open_clip.create_model_and_transforms(
            model_name="coca_ViT-B-32",
            pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
    else:
        raise NotImplementedError
    return visual_model, vid_feat_size, transform


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
    elif feature_extractor == 'clip':
        embed_size = 512
        text_model, _ = clip.load("ViT-B/32")
        tokenizer = get_tokenizer("basic_english")
    elif feature_extractor == 'coca':
        embed_size = 512
        text_model, _, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-B-32",
            pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
        tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
    else:
        raise NotImplementedError
    return text_model, tokenizer, embed_size


def extract_video_features(video_frames, model, feature_extractor, feat_size, finetune=False, test=False):
    if feature_extractor == 'resnet':
        # [B=num_frames, C=3, H=224, W=224]
        if not finetune or test:
            with torch.no_grad():
                video_feats = model(video_frames).view(-1, feat_size)  # [t, 512]
        else:
            video_feats = model(video_frames).view(-1, feat_size)
    elif feature_extractor in ['I3D', 'S3D']:
        video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [b, c, t, h, w]
        if not finetune or test:
            with torch.no_grad():  # [t, 1024]
                video_feats = model.module.extract_features(
                    video_frames).view(-1, feat_size)
        else:
            video_feats = model.module.extract_features(
                video_frames).view(-1, feat_size)
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
    elif feature_extractor == 'clip':
        video_frames = video_frames.unsqueeze(0)  # [b, t, c, h, w]
        video_frames = video_frames.half()  # half precision
        video_frames = torch.flatten(video_frames, start_dim=0, end_dim=1)
        if not finetune or test:
            with torch.no_grad():  # [t, 512]
                video_feats = model.module.visual(video_frames).view(-1, feat_size)
        else:
            video_feats = model.module.visual(video_frames).view(-1, feat_size)
    elif feature_extractor == 'coca':
        if not finetune or test:
            with torch.no_grad():
                video_feats = model.module.encode_image(video_frames)
        else:
            video_feats = model.module.encode_image(video_frames)  # [max_seq_len, batch_size, embed_dim=512]
    # elif feature_extractor == 'clip':
    #     video_frames = video_frames.unsqueeze(0)  # [b, t, c, h, w]
    #     video_frames = torch.flatten(video_frames, start_dim=0, end_dim=1)
    #     if not finetune or test:
    #         with torch.no_grad():  # [t, 1024]
    #             video_feats = model.visual(video_frames).view(-1, feat_size)
    #     else:
    #         video_feats = model.visual(video_frames).view(-1, feat_size)
    return video_feats.float()


def extract_text_features(hypotheses, model, feature_extractor, tokenizer):
    with torch.no_grad():
        # last_hidden_state: [batch_size, max_seq_len, embed_dim]
        if feature_extractor == 'bert':
            tokenizer_out = tokenize_and_pad(hypotheses, tokenizer, feature_extractor)
            text_feats = (model(**dictfilt(tokenizer_out.data, ("input_ids", "attention_mask"))).last_hidden_state,
                          tokenizer_out.data['length'])
        elif feature_extractor == 'glove':
            tokenizer_out = tokenize_and_pad(hypotheses, tokenizer, feature_extractor)
            text_feats = (pad_sequence([model.get_vecs_by_tokens(x).cuda()
                                        for x in tokenizer_out]).permute(1, 0, 2),
                          torch.tensor([len(x) for x in tokenizer_out]).cuda())
        elif feature_extractor == 'coca':
            with torch.no_grad():
                tokenizer_out = tokenizer(hypotheses).cuda()
                text_feats = model.module.encode_text(tokenizer_out)  # [batch_size, embed_dim=512]
            # b = text_feats.shape[0]  # batch_size
            text_feats = (text_feats.float(), None)
        elif feature_extractor == 'clip':
            tokenizer_out = clip.tokenize(hypotheses, truncate=True).cuda()
            text_feats = model.module.token_embedding(tokenizer_out).type(
                model.module.dtype)  # [batch_size, n_ctx, dim]
            text_feats = text_feats + model.module.positional_embedding.type(model.module.dtype)
            text_feats = text_feats.permute(1, 0, 2)  # NLD -> LND
            text_feats = model.module.transformer(text_feats)
            text_feats = text_feats.permute(1, 0, 2)  # LND -> NLD
            text_feats = model.module.ln_final(text_feats).type(model.module.dtype)
            text_feats = text_feats @ model.module.text_projection
            batch_size, _, dim = text_feats.shape
            prev_n_tokens = 20  # data['text'].shape[1]

            tokenizer_out = tokenizer_out[:, 1:]  # first token is a token of beginning of the sentence
            text_feats = text_feats[:, 1:]  # first token is a token of beginning of the sentence

            new_text = text_feats[:, :prev_n_tokens]  # 20 is max?
            new_text_length = torch.zeros(batch_size).cuda()
            for i in range(len(tokenizer_out)):
                # take features from the eot embedding (eot_token is the highest number in each sequence)
                n_eot = tokenizer_out[i].argmax().item()
                new_text_length[i] = min(n_eot, prev_n_tokens)
            # tuple: ([b, max_tokens, 512], [b])
            text_feats = (new_text.float(),
                          new_text_length)
        # elif feature_extractor == 'clip':
        #     tokenizer_out = clip.tokenize(hypotheses, truncate=True)
        #     tokenizer_out = clip.tokenize(hypotheses, truncate=True)
        #     text_feats = model.token_embedding(tokenizer_out).type(
        #         model.dtype)  # [batch_size, n_ctx, dim]
        #     text_feats = text_feats + model.positional_embedding.type(model.dtype)
        #     text_feats = text_feats.permute(1, 0, 2)  # NLD -> LND
        #     text_feats = model.transformer(text_feats)
        #     text_feats = text_feats.permute(1, 0, 2)  # LND -> NLD
        #     text_feats = model.ln_final(text_feats).type(model.dtype)
        #     text_feats = text_feats @ model.text_projection
        #     batch_size, _, dim = text_feats.shape
        #     prev_n_tokens = 20  # data['text'].shape[1]
        #
        #     tokenizer_out = tokenizer_out[:, 1:]  # first token is a token of beginning of the sentence
        #     text_feats = text_feats[:, 1:]  # first token is a token of beginning of the sentence
        #
        #     new_text = text_feats[:, :prev_n_tokens]  # 20 is max?
        #     new_text_length = torch.zeros(batch_size)
        #     for i in range(len(tokenizer_out)):
        #         # take features from the eot embedding (eot_token is the highest number in each sequence)
        #         n_eot = tokenizer_out[i].argmax().item()
        #         new_text_length[i] = min(n_eot, prev_n_tokens)
        #     # tuple: ([b, max_tokens, 512], [b])
        #     text_feats = (new_text.float(),
        #                   new_text_length)
    return text_feats
