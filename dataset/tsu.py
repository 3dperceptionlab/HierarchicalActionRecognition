import torch
from torch.utils.data import Dataset
import os, pickle
import numpy as np

import pandas as pd

class ToyotaSmartHomeDataset(Dataset):
    def __init__(self, config, split="train", num_coarse_actions=7, num_fine_actions=51, real_inference=False):
        self.real_inference = real_inference
        self.inputs = []
        self.anno_path = config.data.anno_path
        self.enc_steps = config.data.enc_steps
        self.split = split
        self.rgb_fts = config.data.rgb_features
        if self.rgb_fts is not None:
            print("Using RGB features")
        self.flow_fts = config.data.flow_features
        if self.flow_fts is not None:
            print("Using Flow features")
        self.text_fts = config.data.text_features
        if self.text_fts is not None:
            print("Using Text features")
        self.fts_model = config.data.fts_model
        self.data_path = config.data.data_path
        self.num_coarse = num_coarse_actions
        self.num_fine = num_fine_actions

        with open(os.path.join(self.anno_path, split + "_split.pkl"), "rb") as f:
            data = pickle.load(f)
        
        for video,anno in data.items():
            for a in anno:
                anno_id, start, end, coarse, fine = a
                length = end - start
                if length < self.enc_steps:
                    continue
                if length > self.enc_steps:
                    start = np.random.randint(0, length - self.enc_steps)
                else:
                    start = 0
                self.inputs.append((video[:-4], start, coarse, fine, anno_id))

    
    def __getitem__(self, index):
        video_name, start, coarse, fine, anno_id = self.inputs[index]
        rgb_t = torch.tensor([0])
        flow_t = torch.tensor([0])
        text = torch.tensor([0])
        if self.rgb_fts is not None:
            file = os.path.join(self.data_path, "rgb", self.rgb_fts, video_name + "_rgb.npy")
            if not os.path.exists(file):
                raise ValueError("File not found: {}".format(file))
            features = np.load(file)
            rgb_t = torch.from_numpy(features[start:start+self.enc_steps])
            del features
        if self.flow_fts is not None:
            file = os.path.join(self.data_path, "flow", self.flow_fts, video_name + "_flow.npy")
            features = np.load(file)
            flow_t = torch.from_numpy(features[start:start+self.enc_steps])
            del features
        if self.text_fts is not None:
            file = os.path.join("/workspace/predicted_actions_16", video_name + "_phrases.txt_features.pkl")
            with open(file, 'rb') as f:
                features = pickle.load(f)

            text = torch.from_numpy(features[anno_id]).squeeze()
            del features


        fts = (rgb_t, flow_t, text)
        
        coarse_t = np.zeros(self.num_coarse)
        coarse_t[coarse] = 1
        coarse_t = torch.from_numpy(coarse_t)

        fine_t = np.zeros(self.num_fine)
        fine_t[fine] = 1
        fine_t = torch.from_numpy(fine_t)
        
        return rgb_t, flow_t, text, coarse_t, fine_t, anno_id, video_name
    
    def __len__(self):
        return len(self.inputs)