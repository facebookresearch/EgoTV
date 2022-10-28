from dataset_utils import tokenize_and_pad, dictfilt
import math
import torch
from torch.nn.utils.rnn import pad_sequence


def extract_video_features(video_frames, model, feature_extractor, feat_size, finetune=False):
    video_features_batch = []  # transforms + visual model features
    vid_lengths = []
    if feature_extractor == 'resnet':
        vid_lengths.append(len(video_frames))
        if not finetune:
            with torch.no_grad():
                video_features_batch.append(model(video_frames).view(-1, feat_size))  # [t, 512]
        else:
            video_features_batch.append(model(video_frames).view(-1, feat_size))
    elif feature_extractor == 'i3d':
        video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [b, c, t, h, w]
        if not finetune:
            with torch.no_grad():  # [t, 1024]
                video_segments = model.module.extract_features(
                    video_frames).view(-1, feat_size)
        else:
            video_segments = model.module.extract_features(
                video_frames).view(-1, feat_size)

        video_features_batch.append(video_segments)
        vid_lengths.append(len(video_segments))
    elif feature_extractor == 'mvit':
        # here b=1 since we are processing one video at a time
        video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [b, c, t, h, w]
        b, c, t, h, w = video_frames.shape
        num_segments = math.ceil(t / 16)
        to_pad = num_segments * 16 - t
        video_frames = torch.cat((video_frames, torch.zeros(b, c, to_pad, h, w).cuda()), dim=2)
        video_frames = video_frames.reshape(b * num_segments, c, 16, h, w)
        if not finetune:
            with torch.no_grad():
                video_segments = model(video_frames).reshape(b * num_segments, feat_size)  # [num_segments, 768]
        else:
            video_segments = model(video_frames).reshape(b * num_segments, feat_size)  # [num_segments, 768]
        video_features_batch.append(video_segments)
        vid_lengths.append(len(video_segments))

    # pad_sequence(video_frames_batch) -> [max_seq_len, batch_size, resnet_dim]
    video_features_batch = (pad_sequence(video_features_batch), torch.tensor(vid_lengths).cuda())
    return video_features_batch


def extract_text_features(hypotheses, model, feature_extractor, tokenizer, glove_vec=None):
    with torch.no_grad():
        # last_hidden_state: [batch_size, max_seq_len, 768]
        tokenizer_out = tokenize_and_pad(hypotheses, tokenizer, feature_extractor)
        if feature_extractor == 'bert':
            hypotheses = (model(**dictfilt(tokenizer_out.data, ("input_ids", "attention_mask"))).last_hidden_state,
                          tokenizer_out.data['length'])
        elif feature_extractor == 'glove':
            hypotheses = (pad_sequence([glove_vec.get_vecs_by_tokens(x).cuda()
                                        for x in tokenizer_out]).permute(1, 0, 2),
                          torch.tensor([len(x) for x in tokenizer_out]).cuda())
    return hypotheses
