import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir):
        # data_names = glob.glob(os.path.join(data_dir, '*f.npy'))
        CLIP_pair = torch.load(f"{data_dir}/feature_CLIP_pair.pt")
        # self.instance_feature = CLIP_pair['feature']
        instance_feature = CLIP_pair['feature']/(torch.norm(CLIP_pair['feature'], dim=-1, keepdim=True)+1e-8)
        self.instance_feature = instance_feature.cpu().numpy()
        # print("self.instance_feature.shape")
        ## TODO
        # import ipdb
        # ipdb.set_trace()
        # self.clip = self.clip/
        self.clip = CLIP_pair['clip'].cpu().numpy()
        
        # self.data = data

    def __getitem__(self, index):
        # data = torch.tensor(self.data[index])
        instance_feature_item = torch.tensor(self.instance_feature[index])
        clip_item = torch.tensor(self.clip[index])
        return instance_feature_item, clip_item

    def __len__(self):
        return self.instance_feature.shape[0] 