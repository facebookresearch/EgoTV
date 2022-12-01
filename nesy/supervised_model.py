import itertools
import re
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
    def __init__(self, vid_embed_size, hsize, rnn_enc):
        super(NeSyBase, self).__init__()
        self.vid_ctx_rnn = rnn_enc(vid_embed_size , hsize, bidirectional=True, dropout_p=0, n_layers=1,
                                      rnn_type="lstm")
        # positional encoding
        self.positional_encode = PositionalEncoding(num_hiddens=2*hsize, dropout=0.5)
        # multi-headed attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=2*hsize,
                                                    num_heads=10,
                                                    dropout=0.5,
                                                    batch_first=True)

        self.num_states = 4  # hot, cold, cleaned, sliced
        self.num_relations = 2  # InReceptacle, Holds
        # TODO: make it deeper?
        # self.state_query = nn.Sequential(nn.Linear(hsize, self.num_states),
        #                                  nn.LogSoftmax())
        # self.relation_query = nn.Sequential(nn.Linear(hsize, 1),
        #                                     nn.LogSigmoid())
        self.num_labels = 7
        self.pred_layer = nn.Sequential(nn.Linear(2*hsize, self.num_labels))


    def forward(self, vid_feats):
        out_logits = []
        for vid_feat in vid_feats:
            # each vid_feat is [num_segments, frames_per_segment, 512]
            b, vid_len, _ = vid_feat.shape
            vid_lens = torch.full((b,), vid_len)
            _, vid_feat = self.vid_ctx_rnn(vid_feat, vid_lens)  # aggregate
            #  vid_feat = [num_segments, 2*hsize]
            vid_feat = self.positional_encode(vid_feat.unsqueeze(0))
            # integrating temporal component into each segment encoding
            vid_feat = self.multihead_attn(vid_feat, vid_feat, vid_feat, need_weights=False)[0]
            out_logits.append(self.pred_layer(vid_feat))  # logits
        return out_logits
