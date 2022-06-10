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

from typing import Tuple

from holotorch.CGH_Datasets.CGH_Dataset import CGH_Dataset

class CITL_Dataset(CGH_Dataset):
    
    def __init__(self,
            dataset_measured : CGH_Dataset,
            dataset_pattern : CGH_Dataset,
            dataset_target : CGH_Dataset
            ):
        
        super().__init__()

        self.dataset_measured = dataset_measured
        self.dataset_pattern  = dataset_pattern
        self.dataset_target   = dataset_target
            
    def __len__(self):
        return len(self.dataset_measured)
    
    @property
    def pattern_dataset(self) -> CGH_Dataset:
        return self.dataset_pattern

    @property
    def measured_dataset(self) -> CGH_Dataset:
        return self.dataset_measured

    @property
    def target_dataset(self) -> CGH_Dataset:
        return self.dataset_target
    
    def __getitem__(self,idx) -> Tuple:
        
        outs = []
        
        measured = self.dataset_measured[idx]
        outs.append(measured)
        
        pattern  = self.dataset_pattern[idx]
        outs.append(pattern)
        try:
            target   = self.dataset_target[idx]
            outs.append(target)
        except IndexError:
            pass
        except TypeError:
            pass
    
        return outs

        
        