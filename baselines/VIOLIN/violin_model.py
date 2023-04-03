# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# this code is developed based on https://github.com/jimmy646/violin/blob/master/model/ViolinBase.py

import torch
from torch import nn
from rnn import RNNEncoder
from bidaf import BidafAttn


class ModelBase(nn.Module):
    def __init__(self, hsize1, hsize2, embed_size, vid_feat_size, attention=True):
        super(ModelBase, self).__init__()
        # hsize1 = 150
        # hsize2 = 300
        # embed_size = 768 (distilbert) or 300 (glove)
        # vid_feat_size = 512  (resnet) or 1024 (I3D)
        self.attention = attention

        if self.attention:
            self.bert_fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(embed_size, hsize1 * 2),
                nn.Tanh()
            )
            self.video_fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(vid_feat_size, embed_size),
                nn.Tanh()
            )
            self.lstm_raw = RNNEncoder(embed_size, hsize1, bidirectional=True,
                                       dropout_p=0, n_layers=1, rnn_type='lstm')
            self.bidaf = BidafAttn(hsize1 * 2, method="dot")
            self.vid_ctx_rnn = RNNEncoder(hsize1 * 2 * 3, hsize2, bidirectional=True, dropout_p=0, n_layers=1,
                                          rnn_type="lstm")
        else:
            self.state_rnn = RNNEncoder(embed_size, hsize1, bidirectional=True, dropout_p=0, n_layers=1,
                                        rnn_type="lstm")
            self.vid_ctx_rnn = RNNEncoder(vid_feat_size, hsize1, bidirectional=True, dropout_p=0, n_layers=1,
                                          rnn_type="lstm")

        self.final_fc = nn.Sequential(
            nn.Linear(hsize2 * 2, hsize2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hsize2, 1),
            nn.Sigmoid()
        )

    # def max_along_time(self, outputs, lengths):
    #     max_outputs = [outputs[i, :int(lengths[i]), :].max(dim=0)[0] for i in range(len(lengths))]
    #     ret = torch.stack(max_outputs, dim=0)
    #     assert ret.size() == torch.Size([outputs.size()[0], outputs.size()[2]])
    #     return ret

    def forward(self, vid_feats, text_feats):
        state_hidden, state_lens = text_feats
        vid_feat, vid_lens = vid_feats
        if self.attention:  # violin baseline
            state_encoded = self.bert_fc(state_hidden)
            vid_projected = self.video_fc(vid_feat)
            vid_encoded, _ = self.lstm_raw(vid_projected, vid_lens)
            u_va, _ = self.bidaf(state_encoded, state_lens, vid_encoded, vid_lens)
            # concat_vid = torch.cat([state_encoded, u_va], dim=-1)
            # _, vec_vid = self.vid_ctx_rnn(concat_vid, state_lens)
            concat_all = torch.cat([state_encoded, u_va, state_encoded * u_va], dim=-1)
            _, vec_vid = self.vid_ctx_rnn(concat_all, state_lens)
            return self.final_fc(vec_vid).view(-1)
        else:
            _, vid_agg = self.vid_ctx_rnn(vid_feat, vid_lens)
            _, state_agg = self.state_rnn(state_hidden, state_lens)
            concat = torch.cat([state_agg, vid_agg], dim=-1)
            return self.final_fc(concat).view(-1)
