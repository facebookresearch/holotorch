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

def convert_integer_into_string(number, depth = 5):
    """Converts a number into a string

    Args:
        number (_type_): _description_

    Returns:
        _type_: _description_
    """    
    test = '%0'+ str(depth) + 'd'
    return str(test % number)