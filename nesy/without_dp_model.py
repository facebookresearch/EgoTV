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
    def __init__(self, vid_embed_size, hsize, hsize2, rnn_enc):
        super(NeSyBase, self).__init__()
        self.vid_ctx_rnn = rnn_enc(2 * vid_embed_size, hsize, bidirectional=True, dropout_p=0, n_layers=1,
                                   rnn_type="lstm")
        # # positional encoding
        # self.positional_encode = PositionalEncoding(num_hiddens=2 * hsize, dropout=0.5)
        # # multi-headed attention
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=2 * hsize,
        #                                             num_heads=10,
        #                                             dropout=0.5,
        #                                             batch_first=True)

        self.num_states = 4  # hot, cold, cleaned, sliced
        self.num_relations = 2  # InReceptacle, Holds
        # TODO: make it deeper?
        self.state_query = nn.Sequential(nn.Linear(2 * hsize, self.num_states),
                                         nn.LogSoftmax(dim=-1))
        self.relation_query = nn.Sequential(nn.Linear(2 * hsize, self.num_relations),
                                            nn.LogSoftmax(dim=-1))
        #TODO: could also use bidaf attention to combine the logits from clip queries and video segments
        self.aligned_agg = rnn_enc(2*hsize, hsize, bidirectional=False, n_layers=1,
                                   rnn_type="lstm")
        self.final_fc = nn.Sequential(
            nn.Linear(hsize, hsize2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hsize2, 1),
            nn.Sigmoid()
        )
        # self.action_query = nn.Sequential(nn.Linear(2 * hsize, 7),
        #                                     nn.LogSoftmax(dim=-1))

    def query(self, segment_feats, seg_labs):
        # log_prob = torch.tensor(0.).cuda()
        state_preds = []
        relation_preds = []
        state_labels = []
        relation_labels = []
        seg_feats_aligned = []
        for seg_feat, seg_lab in zip(segment_feats, seg_labs):
            # out = self.action_query(seg_feat)
            # log_prob += out[seg_lab]
            # state_preds.append(out)
            # state_labels.append(seg_lab)
            if seg_lab.item() <= 3:
                out = self.state_query(seg_feat)
                # log_prob += out[seg_lab]
                seg_feats_aligned.append(seg_feat)
                state_preds.append(out)
                state_labels.append(seg_lab)
            # TODO: relation query definitely needs negative labels
            elif seg_lab.item() in [4, 5]:
                out = self.relation_query(seg_feat)
                # log_prob += out[seg_lab - 4]
                seg_feats_aligned.append(seg_feat)
                relation_preds.append(out)
                relation_labels.append(seg_lab - 4)
            # TODO: use negative examples (like the ChangeIt paper) ?
        ent_prob = self.final_fc(self.aligned_agg(torch.stack(seg_feats_aligned).unsqueeze(0),
                                                  torch.tensor((len(seg_feats_aligned),)))[1])
        if len(state_labels) == 0:
            return ent_prob, \
                   torch.tensor([]), torch.tensor([]), \
                   torch.stack(relation_preds), torch.stack(relation_labels)
        elif len(relation_labels) == 0:
            return ent_prob, \
                   torch.stack(state_preds), torch.stack(state_labels), \
                   torch.tensor([]), torch.tensor([])
        return ent_prob, \
               torch.stack(state_preds), torch.stack(state_labels), \
               torch.stack(relation_preds), torch.stack(relation_labels)

    def forward(self, vid_feats, segment_labels):
        ent_probs = []
        state_log_probs, relation_log_probs = [], []
        state_labels, relation_labels = [], []
        for vid_feat, seg_labs in zip(vid_feats, segment_labels):
            # each vid_feat is [num_segments, frames_per_segment, 512]
            b, vid_len, _ = vid_feat.shape
            vid_lens = torch.full((b,), vid_len)
            _, vid_feat = self.vid_ctx_rnn(vid_feat, vid_lens)  # aggregate
            #  vid_feat = [num_segments, 2*hsize]
            # vid_feat = self.positional_encode(vid_feat.unsqueeze(0))
            # # integrating temporal component into each segment encoding
            # vid_feat = self.multihead_attn(vid_feat, vid_feat, vid_feat, need_weights=False)[0]
            # state_log_prob = self.action_query(vid_feat).squeeze(0)
            # state_log_probs.append(state_log_prob)
            # ent_log_probs.append(torch.take_along_dim(state_log_prob, seg_labs.unsqueeze(-1), dim=-1).sum())
            # state_labels.append(seg_labs)
            ent_prob, state_log_prob, state_label, relation_log_prob, relation_label = \
                self.query(vid_feat.squeeze(0), seg_labs)
            ent_probs.append(ent_prob)
            state_log_probs.append(state_log_prob)
            state_labels.append(state_label)
            relation_log_probs.append(relation_log_prob)
            relation_labels.append(relation_label)

        # flatten the batch -- [batch_size * num_segments, num_labels]
        # flatten the labels -- [batch_size * num_segments]
        return torch.stack(ent_probs).view(-1), \
               state_log_probs, state_labels, \
               relation_log_probs, relation_labels
