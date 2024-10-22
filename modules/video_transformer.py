'''
Part of the code from https://github.com/wangxiang1230/OadTR
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .PositionalEncoding import (LearnedPositionalEncoding, FixedPositionalEncoding)
from .Attention import SelfAttention


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(
                                dim, heads=heads, dropout_rate=attn_dropout_rate
                            ),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class VideoTransformer(nn.Module):
    def __init__(self, config, flow_late=False):
        super().__init__()
        if config.data.rgb_features == 'ViT-H-14':
            input_dim = 1280
        elif config.data.rgb_features == 'tsn-inception_v3':
            input_dim = 2048
        elif config.data.rgb_features == 'tsn-bn_inpception':
            input_dim = 1024

        if config.data.flow_features == 'tsn-inception_v3':
            if flow_late:
                input_dim = 2048
            elif config.model.video_transformer.fusion == 'early':
                input_dim += 2048
        
        
        self.num_heads = config.model.video_transformer.num_heads
        self.embedding_dim = config.model.video_transformer.embedding_dim
        assert self.embedding_dim % self.num_heads == 0, 'embedding_dim must be divisible by num_heads'
    
        self.hidden_dim = config.model.video_transformer.hidden_dim

        self.cls_token = None
        if config.model.video_transformer.cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))

        self.linear_encoding = nn.Linear(input_dim, self.embedding_dim)

        if config.model.video_transformer.positional_encoding == 'learned':
            length = config.data.enc_steps + 1 if self.cls_token is not None else config.data.enc_steps
            self.positional_encoding = LearnedPositionalEncoding(length, self.embedding_dim, length) # +1 for cls token
        elif config.model.video_transformer.positional_encoding == 'fixed':
            self.positional_encoding = FixedPositionalEncoding(self.embedding_dim)
        else:
            self.positional_encoding = None

        self.pe_dropout = nn.Dropout(config.model.video_transformer.pe_dropout)
        self.dropout = nn.Dropout(config.model.video_transformer.dropout)

        self.encoder = TransformerModel(self.embedding_dim, config.model.video_transformer.num_layers,
                                        self.num_heads, self.hidden_dim, config.model.video_transformer.dropout)

    def forward(self, x):
        x = self.linear_encoding(x)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) # Class token is at the end
            x = torch.cat((x, cls_tokens), dim=1) # (B, S, 1024)
        
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
            x = self.pe_dropout(x)
        
        x = self.encoder(x)
        x = self.dropout(x)

        if self.cls_token is not None:
            x = x[:, -1] # (B,1024)
        else:
            x = x.mean(dim=1)

        return x