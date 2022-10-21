# -*- coding: utf-8 -*-
"""
Created on Mon Nov 8 2021

@author: Aline Sindel
"""

import os
from PIL import Image
from math import *
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from data_utils import is_image_file
    
class ImageDataset(Dataset):

    def __init__(self, dataset_dir, in_size):
        self.in_size = in_size
        
        self.image_filenames = []        
        self.image_filenames.extend(os.path.join(dataset_dir, x)
                                     for x in sorted(os.listdir(dataset_dir)) if is_image_file(x))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):        
        image = Image.open(self.image_filenames[index]).convert("RGB")
        image = TF.resize(image, (self.in_size, self.in_size))  
        
        #normalize by 0.5
        image = TF.to_tensor(image)
        image = TF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                 
        return image    