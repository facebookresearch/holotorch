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
    
def transform_6D_to_4D(tensor_in:torch.Tensor):
    """
    
    Transforms the 5D-dimensional into a 4D Dimensional tensor

    NOTE: This is done to enable easier processing with PyTorch framework
    since 5D-processing with 3 batch dimensions are not really supported
    
    From : batch x time x  color x height x width
    To: batch * time x  color x height x width?
    
    B x T x P x C x H X W
    
    (B * T * P ) x C x H X W

    """
    
    my_shape = tensor_in.shape
    #
    new_shape = [my_shape[0]*my_shape[1]*my_shape[2], my_shape[3] , my_shape[4], my_shape[5]]
    #
    torch_out = torch.reshape(input = tensor_in, shape = new_shape )

    return torch_out


def transform_4D_to_6D(tensor_in:torch.Tensor, newshape:torch.Size):
    """
    
    Sister method of transform_5D_to_4D

    transform_4D_to_5D reverts the dimensions collapse

    From : batch * time x  color x height x width
    to: batch x time x  color x height x width?

    """
    torch_out = torch.reshape(input = tensor_in, shape = newshape )
    return torch_out