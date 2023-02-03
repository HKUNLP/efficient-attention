import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

import torch
import torch.nn as nn

class GatedMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., use_glu=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if use_glu:
            hidden_features = int(hidden_features * 2 // 3)
        self.use_glu = use_glu
        self.fc1 = nn.Linear(in_features, hidden_features * (2 if use_glu else 1))
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):

        if not self.use_glu:
            x = self.fc1(x)
            x = self.act(x)
        else:
            x, v = self.fc1(x).chunk(2, dim=-1)
            x = self.act(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x