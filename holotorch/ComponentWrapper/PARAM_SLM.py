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
import numpy as np

class PARAM_SLM(PARAM_COMPONENT):


    def __init__(self) -> None:
        super().__init__()
        
    material            = ENUM_MATERIAL.Default
    num_pixel_x         : int
    num_pixel_y         : int               
    feature_size_slm    : float
    data_sz             : int = 1
    #n_batches           : int = 1
    n_time              : int = 1
    n_channel           : int = 1
    n_slm_batches       : int = 1
    slm_material        = ENUM_MATERIAL.HOLOGRAFIX,
    
    SLM_TYPE            = ENUM_SLM_TYPE.phase_only,
    
    field_fring_type    = ENUM_SLM_FIELD_FRINGE_TYPE.SIMPLE_CONVOLUTION
    field_fringe_sigma  = 1.3
    
    slm_per_batch       = False,
    SLM_INIT            = ENUM_SLM_INIT.RANDOM,
    init_variance       = 2*np.pi
    replicas            = 1
    pixel_fill_ratio    = 1.0
    pixel_fill_ratio_opt= False

    slm_tmp_dir : str    = ".slm"
    slm_id : int        = 0
    
    store_on_gpu : bool  = False