import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import os

class AirBnbDataset(Dataset):
    def __init__(self, df, base_img_path, transform=None, target_transform=None, im_size = (224,224), im_lib = 'PIL', num_channels = 3, data_type = 'float'):
        self.df = df
        print(len(self.df))

        self.transform = transform
        self.target_transform = target_transform
        self.im_size = im_size
        self.im_lib = im_lib
        self.base_img_path = base_img_path
        self.num_channels = num_channels
        if data_type == 'float':
            self.dt = torch.float32

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df['img_path'][idx]
        if self.im_lib == 'PIL':
            image = torch.tensor(np.array(Image.open(os.path.join(self.base_img_path,img_path)).resize(self.im_size)).transpose(2, 0, 1)).to(self.dt)
        else: 
            raise ValueError("Enter correct imaging library")
        label = torch.tensor(self.df['categorical_label'][idx])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label