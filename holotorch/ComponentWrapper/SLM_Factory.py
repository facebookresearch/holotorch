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

import numpy as np
from holotorch.ComponentWrapper import PARAM_SLM
from holotorch.HolographicComponents.Modulator_Container import Modulator_Container
from holotorch.HolographicComponents.SLM_PhaseOnly import SLM_PhaseOnly
from holotorch.HolographicComponents.SLM_Complex_DPAC import SLM_Complex_DPAC
from holotorch.HolographicComponents.SLM_Complex_Cartesian import SLM_Complex_Cartesian
from holotorch.HolographicComponents.SLM_Complex_Polar import SLM_Complex_Polar
import holotorch.utils.Dimensions as Dimensions
from holotorch.utils.Enumerators import *

try:
    from holotorch_dev.HolographicComponents.SLM_Voltage import SLM_Voltage
    from holotorch_dev.HolographicComponents.Look_Up_Table_Phase import Look_Up_Table_Phase
except:
    pass

def create_slm(slm : PARAM_SLM) -> Modulator_Container:
    
        
        SLM_container_dimension = Dimensions.BTCHW(
            n_batch         = slm.data_sz, # Total number of images for Modualator_Container
            n_time          = slm.n_time,
            n_channel       = slm.n_channel,
            height          = slm.num_pixel_x,
            width           = slm.num_pixel_y
        )
        
        if slm.SLM_TYPE == ENUM_SLM_TYPE.voltage:

            slm_lut_curve = Look_Up_Table_Phase(
                    max_phase = 4*np.pi,
                    min_phase = 0*np.pi,
                    gamma = 1,
                    
            )

            # NOTE: currently assumes data_sz is divisible by n_batches
            # create the SLM models
            modulator =  SLM_Voltage(
                tensor_dimension                = SLM_container_dimension,
                feature_size                    = slm.feature_size_slm  ,
                slm_lut_model                   =  slm_lut_curve,
                n_slm_batches                   = slm.n_slm_batches,
                replicas                        = 1,
                pixel_fill_ratio                = 1.0,                  
                pixel_fill_ratio_opt      = False,     
                init_type                 = slm.SLM_INIT,
                init_variance             = slm.init_variance,
                FLAG_optimize             = True,
                field_fringe_type  = slm.field_fring_type,
                slm_directory             = slm.slm_tmp_dir,
                slm_id                    = slm.slm_id
            )
            
            if slm.field_fringe_sigma is not None:
                modulator.field_fringe_blur.sigma = slm.field_fringe_sigma
            
            return modulator

        elif slm.SLM_TYPE == ENUM_SLM_TYPE.phase_only:
            
            modulator = SLM_PhaseOnly(
                tensor_dimension = SLM_container_dimension,
                feature_size    = slm.feature_size_slm  ,
                n_slm_batches                   = slm.n_slm_batches,
                replicas           = 1,
                pixel_fill_ratio     = 1.0,                  
                pixel_fill_ratio_opt = False,     
                init_type     = slm.SLM_INIT,
                init_variance = slm.init_variance,
                FLAG_optimize             = True,
                slm_directory             = slm.slm_tmp_dir,
                slm_id                    = slm.slm_id,
                store_on_gpu              = slm.store_on_gpu,
            )

        elif slm.SLM_TYPE == ENUM_SLM_TYPE.complex_dpac:
            
            modulator = SLM_Complex_DPAC(
                tensor_dimension = SLM_container_dimension,
                feature_size    = slm.feature_size_slm  ,
                n_slm_batches                   = slm.n_slm_batches,
                replicas           = 1,
                pixel_fill_ratio     = 1.0,                  
                pixel_fill_ratio_opt = False,     
                init_type     = slm.SLM_INIT,
                init_variance = slm.init_variance,
                FLAG_optimize             = True,
                slm_directory             = slm.slm_tmp_dir,
                slm_id                    = slm.slm_id
            )

        elif slm.SLM_TYPE == ENUM_SLM_TYPE.complex_polar:
            
            modulator = SLM_Complex_Polar(
                tensor_dimension        = SLM_container_dimension,
                feature_size            = slm.feature_size_slm  ,
                n_slm_batches           = slm.n_slm_batches,
                replicas                = 1,
                pixel_fill_ratio        = 1.0,                  
                pixel_fill_ratio_opt    = False,     
                init_type               = slm.SLM_INIT,
                init_variance           = slm.init_variance,
                FLAG_optimize           = True,
                slm_directory             = slm.slm_tmp_dir,
                slm_id                    = slm.slm_id
            )
            

        elif slm.SLM_TYPE == ENUM_SLM_TYPE.complex_cart:
            
            modulator = SLM_Complex_Cartesian(
                tensor_dimension        = SLM_container_dimension,
                feature_size            = slm.feature_size_slm  ,
                n_slm_batches           = slm.n_slm_batches,
                replicas                = 1,
                pixel_fill_ratio        = 1.0,                  
                pixel_fill_ratio_opt    = False,     
                init_type               = slm.SLM_INIT,
                init_variance           = slm.init_variance,
                FLAG_optimize           = True,
                slm_directory             = slm.slm_tmp_dir,
                slm_id                    = slm.slm_id
            )
        
        return modulator