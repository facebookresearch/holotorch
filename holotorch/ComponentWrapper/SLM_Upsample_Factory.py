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

from holotorch.ComponentWrapper import PARAM_SLM_UPSAMPLE

from holotorch.HolographicComponents.SLM_Upsampler import SLM_Upsampler


def create_slm_upsample(slm_upsample : PARAM_SLM_UPSAMPLE) -> SLM_Upsampler:
        
        # create the SLM_Upsample model
        upsample = SLM_Upsampler(replicas = slm_upsample.replicas, 
                                pixel_fill_ratio = slm_upsample.pixel_fill_ratio)
        
        return upsample