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

from holotorch.ComponentWrapper import PARAM_PROPAGATOR
from holotorch.Optical_Components.FT_Lens import FT_Lens
from holotorch.Optical_Propagators.ASM_Prop import ASM_Prop
from holotorch.utils.Enumerators import *

def create_propagator(prop : PARAM_PROPAGATOR):
    
    if prop.prop_type == ENUM_PROP_TYPE.FOURIER:
        focal_length = prop.focal_length
        
        prop =  FT_Lens(
            focal_length    = focal_length,
            distance        = None,
            pad             = False
        )

    elif prop.prop_type == ENUM_PROP_TYPE.ASM:
        linear_conv = not prop.pad
        
        prop = ASM_Prop(
                        init_distance   = prop.init_distance, 
                        linear_conv     = linear_conv,
                        pad_size        = prop.pad_size
        )


    return prop