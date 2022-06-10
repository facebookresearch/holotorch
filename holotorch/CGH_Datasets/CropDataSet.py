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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from holotorch.CGH_Datasets.CGH_Dataset import CGH_Dataset

class CropDataSet(CGH_Dataset):

    def __init__(self,
        dataset         : CGH_Dataset,
        center_x        : int = 0,
        center_y        : int = 0,
        num_pixel_x     : int = 100,
        num_pixel_y     : int = 100
        ):
        """
        
        Args:
        
        """
        super().__init__()
        
        self.dataset    = dataset
        self.center_x   = center_x
        self.center_y   = center_y
        self.new_num_pixel_x     = num_pixel_x
        self.new_num_pixel_y      = num_pixel_y

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        
        """
        if idx >= len(self):
            raise IndexError
            
        tmp = self.dataset[idx]
        
        if not ( isinstance(tmp,tuple) or  isinstance(tmp,list)) :
            tmp = [tmp]
        
        tmp2 = []
        # Sometimes datasets returns tuples, so we need to move every element in this tuple to the GPU            
        for foo in tmp:
            foo = self.transform_img(foo)
            tmp2.append(foo)
        
        # Let's remove the tuple option if we have only one element
        if len(tmp2) == 1:
            tmp2 = tmp2[0]
        
        return tmp2
            
    def transform_img(self, tmp : torch.Tensor):
        """CROP THE IMAGE

        """        
        x_l = self.center_x - self.new_num_pixel_x // 2
        x_r = self.center_x + self.new_num_pixel_x // 2

        y_l = self.center_y - self.new_num_pixel_y // 2
        y_r = self.center_y + self.new_num_pixel_y // 2
        
        crop = tmp[..., x_l:x_r, y_l:y_r]
        return crop