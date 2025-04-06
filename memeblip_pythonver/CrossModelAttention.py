# -*- coding: utf-8 -*-
import torch.nn as nn
class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, img_feat, txt_feat):
        # img_feat: (B, D), txt_feat: (B, D)
        q = self.query_proj(img_feat).unsqueeze(1)   # (B, 1, D)
        k = self.key_proj(txt_feat).unsqueeze(1)     # (B, 1, D)
        v = self.value_proj(txt_feat).unsqueeze(1)   # (B, 1, D)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, 1, 1)
        attn_weights = attn.softmax(dim=-1)
        output = attn_weights @ v  # (B, 1, D)
        return output.squeeze(1)   # (B, D)
