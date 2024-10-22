import torch.nn as nn
import torch.nn.functional as F
import torch
from .video_transformer import VideoTransformer
from .fusion_transformer import FusionTransformer


class AggregationTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text = False
        self.rgb = False
        self.flow_late = False
        self.flow = False
        self.fine_only = config.model.fine_only

        
        if config.data.rgb_features is not None:
            mlp_input_dim = config.model.video_transformer.embedding_dim
            linear_layer_dim = config.model.video_transformer.embedding_dim
            self.rgb = True
            self.video_transformer = VideoTransformer(config)
        else:
            mlp_input_dim = 0
            linear_layer_dim = 0
        if config.data.flow_features is not None:
            self.flow = True
            if config.model.video_transformer.fusion == 'late' or not self.rgb:
                self.flow_late = True
                self.flow_transformer = VideoTransformer(config, flow_late=True)
                mlp_input_dim += config.model.video_transformer.embedding_dim
                linear_layer_dim += config.model.video_transformer.embedding_dim
        if config.data.text_features:
            mlp_input_dim += 768
            self.text = True


        self.type = config.model.aggregation.type
        self.text_only_for_coarse = config.data.text_only_for_coarse
        if self.text_only_for_coarse:
            mlp_input_dim -= 768
        elif self.type == 'transformer' and ((self.rgb or self.flow) and self.text):
            self.linear_visual = nn.Sequential(nn.Linear(linear_layer_dim, 768), nn.LayerNorm(768))
            self.fusion_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=config.model.aggregation.embedding_dim, nhead=config.model.aggregation.num_heads), num_layers=config.model.aggregation.num_layers)
            mlp_input_dim = config.model.aggregation.embedding_dim

        self.common_mlp = nn.Sequential(nn.Linear(mlp_input_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1), 
                        nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
                        nn.Linear(256, 51))
        if not config.model.fine_only:
            self.coarse_mlp = nn.Sequential(nn.Linear(768, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
                                            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
                                            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
                                            nn.Linear(128, 7))
    

    def forward(self, x):
        rgb_t, flow_t, text = x
        
        if self.flow:
            if self.flow_late or not self.rgb:
                flow_fts = self.flow_transformer(flow_t)
            else:
                rgb_t = torch.cat([rgb_t, flow_t], dim=-1)
        if self.rgb:
            fts = self.video_transformer(rgb_t)

        if self.flow_late:
            if self.rgb:
                fts = torch.cat((fts, flow_fts), dim=-1)
            else:
                fts = flow_fts

        if self.text and (self.rgb or self.flow) and not self.text_only_for_coarse:
            if self.type == 'transformer':
                fts = self.linear_visual(fts) 
                combined_seq = torch.stack([fts, text], dim=1)
                output = self.fusion_transformer(combined_seq)
                output = torch.mean(output, dim=1)
            else:
                output = torch.cat((fts, text), dim=-1)
        elif self.text and not self.text_only_for_coarse:
            output = text
        else:
            output = fts

        fine = self.common_mlp(output)

        if not self.fine_only:
            coarse = self.coarse_mlp(text)
        else:
            coarse = torch.tensor([0])

        return coarse, fine