import torch
import torch.nn as nn


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        att_w = torch.softmax(self.W(batch_rep).squeeze(-1), dim=1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)
        return utter_rep


class CocaVidModel(nn.Module):
    def __init__(self, embed_size):
        super(CocaVidModel, self).__init__()
        # attention pooling for images
        self.pooler = SelfAttentionPooling(input_dim=embed_size)
        self.final_fc = nn.Sequential(
            nn.Linear(2 * embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embed_size, 1),
            nn.Sigmoid()
        )

    def forward(self, vid_feats, text_feats):
        # text_feats are sentence encodings, hence already aggregated
        vid_feat, vid_lens = vid_feats
        vid_feat_agg = self.pooler(vid_feat)
        return self.final_fc(torch.cat([text_feats, vid_feat_agg], dim=-1)).view(-1)
