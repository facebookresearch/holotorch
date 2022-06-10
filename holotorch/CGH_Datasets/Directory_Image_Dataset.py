########################################################
# Copyright (c) 2022 Meta Platforms, Inc. and affiliates
#
# Holotorch is an optimization framework for differentiable wave-propagation written in PyTorch 
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
#
# Contact:
# florianschiffers (at) gmail.com
# ocossairt ( at ) fb.com
#
########################################################

import torch
import tifffile
import os
import torch
import pandas as pd
import numpy as np
import os, torch
import torchvision.io
import pandas as pd
import numpy as np
import glob
from torch.utils.data import Subset
from holotorch.CGH_Datasets.DataTransformer import DataTransformer

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from holotorch.CGH_Datasets.CGH_Dataset import CGH_Dataset

class Directory_Image_Dataset(CGH_Dataset):

    # Pre-initialize current batch to none
    current_batch = None

    def __init__(self,
        img_dir : str,
        grayscale,
        num_pixel_x,
        data_sz = 1,
        batch_size = 5,
        num_pixel_y  :int = None,
        border_x   : int = 0,
        border_y   : int = 0,
        flag_time_multiplexed : bool = False
        ):
        """
        
        Args:
        
        """
        super().__init__()

        # Set the directory where images are stored                
        self.img_dir = img_dir
        
        self.transform = DataTransformer(
            num_pixel_x =num_pixel_x,
            num_pixel_y = num_pixel_y,
            border_x    = border_x,
            border_y    = border_y,
            grayscale   = grayscale,
        )
        
        self.flag_time_multiplexed = flag_time_multiplexed
        # Get the Filnames
        
        files = []
        for ext in ('*.gif', '*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
            tmp_files = glob.glob(os.path.join(self.img_dir, ext))
            names = [os.path.basename(x) for x in tmp_files]

            files.extend(names)
        self.img_labels = pd.DataFrame(files)

        self.data_sz = data_sz
        self.train_dataset = Subset(self, np.arange(data_sz))

        self.pre_loaded = False         


    def pre_load_dataset(self, save_dir : str = None):
        '''
        save transformed data in tensor format
        '''
        
        if save_dir == None:
            save_dir = self.img_dir
        
        for idx in range(len(self)):
            image = self.__getitem__(idx)
            tensor_path = os.path.join(save_dir, self.img_labels.iloc[idx,0] + '.pt')
            torch.save(image, tensor_path)
        self.pre_loaded = True

    def get_preloaded(self,
                      idx : int
                      ):
        # print("loading pre-transformed tensor: ", str(idx), end = '')
        tensor_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0] + '.pt')
        images = torch.load(tensor_path)
        self.current_batch = images
        return images



    def get_image_path(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        return img_path
    
    def read_and_process_image(self, idx):
        img_path = self.get_image_path(idx)
        try:
            image = torchvision.io.read_image(img_path)
        except: 
            image = torch.tensor(tifffile.imread(img_path))

        image = image / 255.0
        image = self.transform(image)
        
        return image


    def __getitem__(self, idx):
        """
        
        """
        if idx >= len(self):
            raise IndexError
            
        if self.pre_loaded:
            image = self.get_preloaded(idx)
        else:
            image = self.read_and_process_image(idx)
                
        self.current_batch = image
        
        if self.flag_time_multiplexed:
            image = image.permute(2,0,1,3,4)

        return image
                
    def __len__(self):
        return self.data_sz

    def set_test_dataset(self):
        pass
    
    def set_train_dataset(self):
        pass
        
    def show_sample(self):
        pass

    def imshow_current_batch(self):
        """
        
        
        """

        if self.current_batch is None:
            raise ValueError("Current Batch has not been loaded yet. Please load.")

        self.current_batch.visualize()