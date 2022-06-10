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
from holotorch.ComponentWrapper import PARAM_PUPIL
from holotorch.Pupils.Identity_Pupil import Identity_Pupil

try:
    from holotorch.Pupils.Complex_Pupil import Complex_Pupil
except:
    pass

import holotorch.utils.Dimensions as Dimensions

def create_pupil(param : PARAM_PUPIL):
    
    
    if param.FLAG_identity_pupil == True:
        
        pupil = Identity_Pupil()
        return pupil
        
    n_pupils        = param.n_pupils
    height          = param.height
    width           = param.width
    channel         = param.channel
    radius          = param.init_rad
    cropped         = param.cropped

    pupil_dimension = Dimensions.PCHW(
        channel                 = channel,
        pupil                   = n_pupils,
        height                  = height,
        width                   = width
    )
    
    pupil = Complex_Pupil(
            pupil_sz        = pupil_dimension,
            radii           = torch.tensor([radius]),
            cropped         = cropped,
        )
    return pupil
