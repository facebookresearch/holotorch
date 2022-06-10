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

class PARAM_PROPAGATOR(PARAM_COMPONENT):


    def __init__(self) -> None:
        super().__init__()
        
    # create the Hologram -> Detector propagation model
    focal_length    = 35*mm
    pad             = False
    prop_type       = ENUM_PROP_TYPE.FOURIER
    init_distance   = 0.0
    pad_size        = None
    
    focal_length_1 = None
    focal_length_2 = None
    aperture_radius = None