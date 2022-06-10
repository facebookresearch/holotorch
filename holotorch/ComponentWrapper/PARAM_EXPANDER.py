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
from holotorch.utils.units import *

class PARAM_EXPANDER(PARAM_COMPONENT):
    num_pixel_x : int          = None
    num_pixel_y : int
    material          = ENUM_MATERIAL.HOLOGRAFIX
    spacing           = None
    center_wavelength = None
    holo_type : ENUM_HOLO_TYPE = ENUM_HOLO_TYPE.complex
    init_type : ENUM_HOLO_INIT = ENUM_HOLO_INIT     