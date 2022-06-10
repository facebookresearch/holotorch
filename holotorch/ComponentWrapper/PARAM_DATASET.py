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

from __future__ import annotations
from holotorch.ComponentWrapper.PARAM_COMPONENT import PARAM_COMPONENT

from holotorch.utils.Enumerators import *

class PARAM_DATASET(PARAM_COMPONENT):

    def __init__(self) -> None:
        super().__init__()

    data_folder            = None
    data_sz                = None
    num_pixel_x : int      = None
    num_pixel_y : int      = None
    batch_size      = None
    border_x        = 0
    border_y        = 0
    path            = None
    TYPE_dataloader = ENUM_DATASET.DIV2K_Dataset
    color_flag    :     ENUM_SENSOR_TYPE  = None
