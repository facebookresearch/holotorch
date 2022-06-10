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

from holotorch.ComponentWrapper import PARAM_CONTENT_INDEPENDENT_DETECTOR_FIELD
from holotorch.HolographicComponents import Content_independent_detector_field
import holotorch.utils.Dimensions as Dimensions

def create_content_independent_detector_field(
    constant_field : PARAM_CONTENT_INDEPENDENT_DETECTOR_FIELD
            ):
    

    N_pixel_x = constant_field.N_pixel_x    
    N_pixel_y = constant_field.N_pixel_y
    N_time = constant_field.N_time

    tensor_dimension = Dimensions.TCHW(
        n_time = N_time ,
        n_channel = 1,
        height = N_pixel_x,
        width = N_pixel_y
    )
    
    content_indepdent  = Content_independent_detector_field(
        tensor_dimension = tensor_dimension
    )
    
    return content_indepdent