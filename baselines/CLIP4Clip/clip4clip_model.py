# this code is developed based on
# CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval (https://arxiv.org/abs/2104.08860)
import os
import sys

sys.path.append(os.environ['BASELINES'])

from end2end.rnn import RNNEncoder
import torch
from torch import nn
from torch.nn import LayerNorm


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CrossPooler(nn.Module):
    def __init__(self, hid_size):
        super(CrossPooler, self).__init__()
        self.ln_pool = LayerNorm(hid_size)
        self.dense = nn.Linear(hid_size, hid_size)
        self.activation = QuickGELU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = self.ln_pool(hidden_states)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens)).cuda()
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class TypeEncoding(nn.Module):
    """ Type Encoding: similar to Segmentation encoding in BERT """

    def forward(self, concat_feats, text_feats, vid_feats):
        type_encoding = torch.cat((torch.zeros_like(text_feats).cuda(),
                                   torch.ones_like(vid_feats).cuda()), dim=1)
        concat_feats += type_encoding
        return concat_feats


class CLIP4Clip(nn.Module):
    def __init__(self, embed_size, sim_type, temp=0.1):
        super(CLIP4Clip, self).__init__()
        self.embed_size = embed_size
        self.sim_type = sim_type  # [meanPool, seqLSTM, tightTransfer, hitchHiker]
        self.temp = temp

        if self.sim_type in ['meanPool', 'seqLSTM', 'hitchHiker']:
            if self.sim_type == 'seqLSTM':
                self.vid_ctx_rnn = RNNEncoder(embed_size,
                                              int(embed_size / 2),
                                              bidirectional=True,
                                              dropout_p=0,
                                              n_layers=1,
                                              rnn_type="lstm")
            # for concatenated text and video (pooled) feats
            self.final_fc = nn.Sequential(
                nn.Linear(2 * embed_size, embed_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(embed_size, 1),
                nn.Sigmoid()
            )
        elif self.sim_type == 'tightTransfer':
            # positional encoding
            self.positional_encode = PositionalEncoding(num_hiddens=embed_size, dropout=0.5)
            # type encoding
            self.type_encode = TypeEncoding()
            # multi-headed attention
            self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_size,
                                                        num_heads=8,
                                                        dropout=0.5,
                                                        batch_first=True)
            self.pooler = CrossPooler(hid_size=embed_size)
            # self.similarity_dense = nn.Sequential(nn.Linear(embed_size, 1),
            #                                       nn.Sigmoid())
            # for concatenated text and video (pooled) feats
            self.final_fc = nn.Sequential(
                nn.Linear(embed_size, int(embed_size/2)),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(int(embed_size/2), 1),
                nn.Sigmoid()
            )

    def _mean_pooling_for_similarity_visual(self, vid_feats):
        return torch.sum(vid_feats, dim=1) / vid_feats.shape[1]

    def _mean_pooling_for_similarity_text(self, text_feats):
        return torch.sum(text_feats, dim=1) / text_feats.shape[1]

    def _weighted_pooling_for_visual(self, vid_feats, text_feats):
        text_feats = text_feats.view(-1, 1, self.embed_size)
        weights = torch.softmax((vid_feats * text_feats).sum(dim=-1) / self.temp, dim=-1)
        return (weights.unsqueeze(-1) * vid_feats).sum(dim=1)

    def forward(self, vid_feats, text_feats):
        vid_feats, vid_lens = vid_feats
        batch_size = vid_feats.shape[0]
        # vid_feats = [b, max_seq_len, 512], vid_lens = [b]
        if self.sim_type in ['meanPool', 'seqLSTM', 'hitchHiker']:  # loose similarity
            if self.sim_type == 'seqLSTM':
                _, vid_feat_agg = self.vid_ctx_rnn(vid_feats, vid_lens)
            elif self.sim_type == 'meanPool':
                vid_feat_agg = self._mean_pooling_for_similarity_visual(vid_feats)  # [b, 512]
            elif self.sim_type == 'hitchHiker':
                # A CLIP-Hitchhiker’s Guide to Long Video Retrieval
                vid_feat_agg = self._weighted_pooling_for_visual(vid_feats, text_feats)

            vid_feat_normed = vid_feat_agg / vid_feat_agg.norm(dim=-1, keepdim=True)
            # text_feats = [b, 512]
            text_feats_normed = text_feats / text_feats.norm(dim=-1, keepdim=True)
            return self.final_fc(torch.cat([text_feats_normed, vid_feat_normed], dim=-1)).view(-1)
            # return torch.sigmoid((vid_feat_normed * text_feats_normed).sum(dim=-1))

        elif self.sim_type == 'tightTransfer':  # tight similarity
            text_feats = text_feats.view(batch_size, 1, self.embed_size)
            concat_feat = torch.cat((text_feats, vid_feats), dim=1)
            concat_feat = self.positional_encode(concat_feat)
            concat_feat = self.type_encode(concat_feat, text_feats, vid_feats)
            concat_feat = self.multihead_attn(concat_feat, concat_feat, concat_feat, need_weights=False)[0]
            pooled_feat = self.pooler(concat_feat)
            return self.final_fc(pooled_feat).view(-1)
