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

from holotorch.ComponentWrapper import PARAM_DETECTOR
from holotorch.Sensors.Detector import Detector

def create_detector(detector : PARAM_DETECTOR) -> Detector:
    
    
    RGB_multiplexing    = detector.RGB_multiplexing
    color_flag          = detector.color_flag
    N_pixel_out_x       = detector.num_pixel_x
    N_pixel_out_y       = detector.num_pixel_y
    zero_pad_flag       = detector.zero_pad_flag
    # create the detector
    detector = Detector(
        RGB_multiplexing    = RGB_multiplexing,
        color_flag          = color_flag,
        N_pixel_out_x       = N_pixel_out_x,
        N_pixel_out_y       = N_pixel_out_y,
    )
    
    return detector
    