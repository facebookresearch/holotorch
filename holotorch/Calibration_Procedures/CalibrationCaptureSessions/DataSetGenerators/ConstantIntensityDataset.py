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
from holotorch.CGH_Datasets.CGH_Dataset import CGH_Dataset

class ConstantIntensityDataset(CGH_Dataset):
    
    def __init__(self,
            num_pixel_x= 1080,
            num_pixel_y= 1920,
            int_min = 0.0,
            int_max = 1.0,
            n_ims = 10,
            ) -> None:

        self.n_ims = n_ims        
        self.int_range = torch.linspace(int_min,int_max,n_ims)
        self.im = torch.ones((num_pixel_x,num_pixel_y))    

    def __getitem__(self, idx):
        """
        
        """
        intensity = self.int_range[idx]
        return (intensity * self.im)[None,None,None]
                
    def __len__(self):
        return self.n_ims 


