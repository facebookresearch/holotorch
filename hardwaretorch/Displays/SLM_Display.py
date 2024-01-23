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

from hardwaretorch.Displays.Display import Display
import slmpy as slmpy
import numpy as np
import torch

__slm__ = None
def get_slmpy(monitor = 1, isImageLock = True, alwaysTop = False):
    global __slm__
    if __slm__ is None:
        print("Create new SLM DIsplay")
        __slm__ = slmpy.SLMdisplay(monitor = monitor, isImageLock = isImageLock, alwaysTop = alwaysTop)

    return __slm__
    

class SLM_Display(Display):
    
    def __init__(self, monitor = 1,  isImageLock = True, alwaysTop = False) -> None:

        global __slm__
        self.slm = get_slmpy(monitor = monitor, isImageLock = isImageLock, alwaysTop = alwaysTop)
        
    @property
    def slm(self) -> slmpy.SLMdisplay:
        """Returns the SLM object in case more specific changes needed to be made.
        
        In General we wrap all necessary methods into this Display class
        so that each DISPLAY is implemented the same way and similair abstract
        functions can be called.
        
        Hopefully you never have to call this method.
        
        Returns:
            slmpy.SLMdisplay: [description]
        """        
        return self._slm
    
    @slm.setter
    def slm(self, slm : slmpy.SLMdisplay) -> None:
        self._slm = slm

    def get_resolution(self):
        """Return resolution of SLM
        """ 
        return self.slm.getSize()
    
    def display_pattern(self,
            pattern : torch.Tensor or np.ndarray,
            sleep = 0.2,
            bit_depth : int = 8
            # max_phase = 2*np.pi
                        ):
        
        #NOTE: pattern is assumed to be \in [0,1]
        if pattern.max() > 1 or pattern.min() < 0:
            raise ValueError('SLM pattern values not between 0-1!')

        if torch.is_tensor(pattern):
            pattern = pattern.detach().cpu().numpy().squeeze()
        
        pattern = pattern.squeeze()

        # pattern = pattern * 255 / max_phase
        
        # quantize pattern 
        # pattern = pattern * (2**bit_depth - 1)

        pattern = np.round(pattern * (2.**bit_depth - 1))  *  255 / (2.**bit_depth)  


        if pattern.dtype is not "uint8":
            pattern = pattern.astype('uint8')
            
        
        self.slm.updateArray(pattern, sleep = sleep)

        return pattern
        
    # def get_random_image(self):
    #     out = np.random.rand(self.get_resolution()[1], self.get_resolution()[0])
    #     out = out * 255
    #     out = out.astype('uint8')
    #     return out
