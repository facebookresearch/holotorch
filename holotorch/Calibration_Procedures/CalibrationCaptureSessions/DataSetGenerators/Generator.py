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

def compute_radial_map(
        num_pixel_x= 1080,
        num_pixel_y= 1920,
):
    aspect_ratio = num_pixel_y/num_pixel_x
    x = torch.linspace(-1,1, num_pixel_x)
    y = aspect_ratio*torch.linspace(-1,1,num_pixel_y)

    X,Y = torch.meshgrid(x,y)

    R = torch.sqrt(X**2 + Y**2)
    
    return R

def compute_quadratic_phase_targets(
    R : torch.Tensor,
    scale = torch.arange(1,10),
    amplitude_scale = torch.linspace(0.4,1,5),
    flag_blazed_grating = True,
):
    quadratic_phase = R **2
    quadratic_phase / quadratic_phase.max()
    
    img = quadratic_phase[None,:,:] * scale[:,None,None]
    
    if flag_blazed_grating:
        img = img.remainder(1)
    else:
        img = 0.5 *(torch.sin(img) + 1)

    img = img[None,:,:,:] * amplitude_scale[:,None,None,None]

    return img