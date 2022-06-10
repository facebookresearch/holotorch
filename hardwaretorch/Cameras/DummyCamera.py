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


from hardwaretorch.Cameras.Camera import Camera
from holotorch.utils.tictoc import *

import torch


from holotorch.utils.units import *

class DummyCamera(Camera):

    def __init__(self,
                 exposure = 10*s,
                 white_balance = False,
                 auto_focus = False,
                 fps = 1,
                 resolution = [1020,1980],
                 grayscale = True
                 ):
        super().__init__(exposure, white_balance, auto_focus, fps, resolution, grayscale)
        
    def getImage(self):
        # To override: Capture image, return frame and save in corresponding folder in specified file format
        return torch.rand(self.resolution[0],self.resolution[1])