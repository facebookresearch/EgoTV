# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class NeSyBase(nn.Module):
    def __init__(self, vid_embed_size, hsize, hsize2, rnn_enc):
        super(NeSyBase, self).__init__()
        self.vid_ctx_rnn = rnn_enc(2 * vid_embed_size, hsize, bidirectional=True, dropout_p=0, n_layers=1,
                                   rnn_type="lstm")
        self.text_ctx_rnn = rnn_enc(vid_embed_size, hsize, bidirectional=True, dropout_p=0, n_layers=1,
                                    rnn_type="lstm")
        # positional encoding
        # self.positional_encode = PositionalEncoding(num_hiddens=2 * hsize, dropout=0.5)
        # multi-headed attention
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=2 * hsize,
        #                                             num_heads=10,
        #                                             dropout=0.5,
        #                                             batch_first=True)

        self.num_states = 4  # hot, cold, cleaned, sliced
        self.num_relations = 2  # InReceptacle, Holds

        self.state_query = nn.Sequential(nn.Linear(4 * hsize, hsize),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(hsize, 1))
        self.relation_query = nn.Sequential(nn.Linear(4 * hsize, hsize),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(hsize, 1))

    def forward(self, vid_feats, text_feats, segment_labels):
        ent_probs = []
        for vid_feat, text_feat, seg_labs in zip(vid_feats, text_feats, segment_labels):
            # each vid_feat is [num_segments, frames_per_segment, 512]
            b, vid_len, _ = vid_feat.shape
            vid_lens = torch.full((b,), vid_len).cuda()
            _, vid_feat = self.vid_ctx_rnn(vid_feat, vid_lens)  # aggregate
            vid_feat = vid_feat.unsqueeze(0)  # [1, num_segments, 512]
            #  vid_feat = [num_segments, 2*hsize]
            # vid_feat = self.positional_encode(vid_feat.unsqueeze(0))
            # integrating temporal component into each segment encoding
            # vid_feat = self.multihead_attn(vid_feat, vid_feat, vid_feat, need_weights=False)[0]
            # alignment_vid_feat = vid_feat[:, seg_labs <= 5, :]

            # each seg_text_feats is [num_segments, 20, 512]
            seg_text_feats, seg_text_lens = text_feat
            _, seg_text_feats = self.text_ctx_rnn(seg_text_feats, seg_text_lens)
            seg_text_feats = seg_text_feats.unsqueeze(0)  # [1, num_segments, 512]

            concat_feats = torch.cat((vid_feat, seg_text_feats), dim=-1)  # [1, num_segments, 1024]
            aligned_aggregated = torch.tensor(0.).cuda()
            for ind, seg_lab in enumerate(seg_labs):
                if seg_lab.item() <= 3:
                    aligned_aggregated += self.state_query(concat_feats[:,ind,:])[0][0] / len(seg_labs)
                elif seg_lab.item() in [4, 5]:
                    aligned_aggregated += self.relation_query(concat_feats[:,ind,:])[0][0] / len(seg_labs)

            ent_probs.append(torch.sigmoid(aligned_aggregated))

        return torch.stack(ent_probs).view(-1)
