import torch
import torch.nn as nn
from torch.nn import LayerNorm


class CocaVidModel(nn.Module):
    def __init__(self, dim):
        self.img_attn_pool_norm = LayerNorm(dim)
        self.img_attn_pool = CrossAttention(dim=dim, context_dim=image_dim, dim_head=dim_head, heads=heads,
                                            norm_context=True)

    def forward(self, img_queries, img_tokens):

        # attention pool image tokens
        img_queries = self.img_attn_pool(img_queries, image_tokens)
        img_queries = self.img_attn_pool_norm(img_queries)
