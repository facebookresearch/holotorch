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

class DummyDisplay:
    def __init__(self):
        pass

    def display_pattern(self,
                        pattern,
                        sleep  = 0,
                        bit_depth : int = 8

    ) -> torch.Tensor:
        #NOTE: pattern is assumed to be \in [0,1]
        if pattern.max() > 1 or pattern.min() < 0:
            raise ValueError('SLM pattern values not between 0-1!')

        if torch.is_tensor(pattern):
            pattern = pattern.detach().cpu().numpy().squeeze()
        
        pattern = pattern.squeeze()

        # pattern = pattern * 255 / max_phase
        
        # quantize pattern 
        pattern = pattern * (2**bit_depth - 1)
            
        if pattern.dtype is not "uint8":
            pattern = pattern.astype('uint8')
        
        return pattern

    def get_resolution(self):
        # To override
        pass

    def set_resolution(self, resolution):
        # To override
        pass

    def quit_and_close(self):
        # To override
        pass

    def getStatus(self):
        # To override
        pass