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

from __future__ import annotations
import torch

from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.utils.Enumerators import *
from holotorch.utils.Dimensions import BTPCHW, TensorDimension, BTCHW
from holotorch.HolographicComponents.Modulator_Container import Modulator_Container

class SLM_PhaseOnly(Modulator_Container):
    """Implements a SLM that maps from Voltage Values to Phase

    Args:
        CGH_Component (_type_): _description_
    """
      
    def __init__(self,
            tensor_dimension : TensorDimension,
            feature_size    : float,
            n_slm_batches                   = 1,
            replicas :int               = 1,
            pixel_fill_ratio: float     = 1.0,                  
            pixel_fill_ratio_opt: bool  = False,     
            init_type       : ENUM_SLM_INIT = None,
            init_variance   : float = 0,
            FLAG_optimize   : bool          = True,
            slm_directory : str         = ".slm",
            slm_id : int                = 0,
            store_on_gpu                = False,
            ):
        
        super().__init__(
            tensor_dimension            = tensor_dimension,
            feature_size                = feature_size,
            n_slm_batches               = n_slm_batches,
            replicas                    = replicas,
            pixel_fill_ratio            = pixel_fill_ratio,                  
            pixel_fill_ratio_opt        = pixel_fill_ratio_opt,  
            init_type                   = init_type,
            init_variance               = init_variance,
            slm_directory               = slm_directory,
            slm_id                      = slm_id,
            store_on_gpu                = store_on_gpu,
        )
    
    @classmethod
    def create_slm(cls,
            height  : int,
            width   : int, 
            feature_size    : float,
            replicas :int               = 1,
            pixel_fill_ratio: float     = 1.0,                  
            pixel_fill_ratio_opt: bool  = False,     
            init_type       : ENUM_SLM_INIT = None,
            init_variance   : float = 0,
            FLAG_optimize   : bool          = True,
            n_batch  : int = 1, # Total number of images for Modualator_Container
            n_time   : int = 1,
            n_channel : int = 1,
            n_slm_batches                   = 1,
        ) -> SLM_PhaseOnly:
                
        SLM_container_dimension = BTCHW(
            n_batch         = n_batch, # Total number of images for Modualator_Container
            n_time          = n_time,
            n_channel       = n_channel,
            height          = height,
            width           = width
        )
        
        return SLM_PhaseOnly(
            tensor_dimension = SLM_container_dimension,
            n_slm_batches = n_slm_batches,
            feature_size    = feature_size,
            replicas  = replicas,
            pixel_fill_ratio = pixel_fill_ratio,                  
            pixel_fill_ratio_opt = pixel_fill_ratio_opt,     
            init_type     = init_type,
            init_variance  = init_variance,
            FLAG_optimize  = FLAG_optimize,
        )
            
    def forward(self,
                field : ElectricField = None,
                batch_idx = None,
                bit_depth : int = None,
                ) -> ElectricField:
        
        phases, scale = super().forward(batch_idx = batch_idx)
        
        slm_field = torch.exp(1j*phases)
        
        if field is None:
            return slm_field
        

        field_data = field.data * slm_field[:,:,None] # Expand Slm Field for pupil dimension

        # We need a scale for batch

        scale_shape = self.batch_tensor_dimension.get_new_shape(BTPCHW)

        scale = scale.view(scale_shape[:-2])
        
        field_data = field_data * scale[...,None,None]

        out = ElectricField(
            data = field_data,
            wavelengths = field.wavelengths,
            spacing = field.spacing
        ) 


        
        return out

  
    