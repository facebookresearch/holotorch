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

from holotorch.CGH_Datasets.CGH_Dataset import CGH_Dataset

class PatternDataset(CGH_Dataset):
    
    def __init__(self,
            data,
            ) -> None:

        if not type(data) == list:
            data = [data]
        self.data = data

    def __getitem__(self, idx):
        """
        
        """
        if idx < len(self):
            return self.data[idx]
        else:
            return IndexError()

    def __len__(self):
        return len(self.data) 


