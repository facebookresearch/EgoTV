# VideoCLIP model
import torch
import torch.nn as nn


class VideoClipModel(nn.Module):
    def __init__(self, embed_size=768):
        super(VideoClipModel, self).__init__()
        self.final_fc = nn.Sequential(
            nn.Linear(2 * embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embed_size, 1),
            nn.Sigmoid()
        )

    def forward(self, vid_feats, text_feats):
        return self.final_fc(torch.cat([vid_feats, text_feats], dim=-1)).view(-1)
