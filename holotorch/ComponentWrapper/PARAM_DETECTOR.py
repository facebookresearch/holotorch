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
from holotorch.utils.Enumerators import *

class PARAM_DETECTOR(PARAM_COMPONENT):


    def __init__(self) -> None:
        super().__init__()
        
    color_flag              = ENUM_SENSOR_TYPE.BAYER
    RGB_multiplexing        = ENUM_TIME_MULTIPLEXING.TIME_MULTIPLEX
    num_pixel_x : int       = None
    num_pixel_y : int       = None
    zero_pad_flag : bool    = False
