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

from holotorch.ComponentWrapper.PARAM_COMPONENT import PARAM_COMPONENT
class PARAM_CONTENT_INDEPENDENT_DETECTOR_FIELD(PARAM_COMPONENT):
    
    N_pixel_x : int = None
    N_Pixel_y : int = None
    N_time : int = 3
    N_channel : int = 0