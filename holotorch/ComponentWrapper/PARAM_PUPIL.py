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


class PARAM_PUPIL(PARAM_COMPONENT):

    def __init__(self) -> None:
        super().__init__()
        
    FLAG_identity_pupil = True
    init_rad            = 1
    n_pupils            = 1
    channel             = 1 
    height              = 100
    width               = 100
    cropped             = True

