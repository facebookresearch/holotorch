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
import matplotlib.pyplot as plt

from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.utils.Enumerators import *
from holotorch.utils.Dimensions import TensorDimension
from holotorch.HolographicComponents.Modulator_Container import Modulator_Container
from holotorch.utils.Visualization_Helper import add_colorbar

class SLM_Complex_Cartesian(Modulator_Container):
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
            slm_id                      = slm_id
        )

        

    def visualize_slm(self, figsize=(10,10),
                      batch_idx = 0,
                      sub_batch_idx = 0,
                    x0 = None,
            y0 = None,
            width = None,
            height = None,
            vmax = None,
            vmin = None,
            title1 = "",
            ):
        
        self.load_single_slm(batch_idx=batch_idx)
        
        plt.figure(figsize=figsize)
        if height == None:
            x1 = None
        else:
            x1 = x0 + height

        if width == None:
            y1 = None
        else:
            y1 = y0 + width

        img1 = self.forward()
        img1 = img1[sub_batch_idx,:,:,x0:x1,y0:y1]
        
        if torch.is_tensor(img1):
            img1 = img1.detach().cpu().squeeze()
            
        if img1.ndim == 2:
            
            plt.subplot(121)
            _im = plt.imshow(img1.abs(), vmax = vmax, vmin=vmin, cmap = 'gray')
            add_colorbar(_im)
            plt.title("Magnitude")
            plt.subplot(122)
            _im = plt.imshow(img1.angle(), vmax = vmax, vmin=vmin, cmap = 'gray')
            add_colorbar(_im)
            plt.title("Phase")
            
            plt.tight_layout()
            
        elif img1.ndim == 3:
            
            for k in range(img1.shape[0]):
                plt.subplot(1,img1.shape[0],k+1)
                _im = plt.imshow(img1[k], vmax = vmax, vmin=vmin, cmap = 'gray')
                plt.title(title1)
            plt.tight_layout()
            
    
    def reset_values(self, noise_variance = 0.1):
        self.set_new_values(noise_variance* (torch.rand_like(self.values.data_tensor) + torch.rand_like(self.values.data_tensor)*1j))

    def forward(self,
                field : ElectricField = None,
                batch_idx = None,
                bit_depth : int = None,
                ) -> ElectricField:
        
        slm_field, _ = super().forward()
       
       
        if field is None:
            return slm_field
        

        field_data = field.data * slm_field[:,:,None] # Expand Slm Field for pupil dimension
        
        out = ElectricField(
            data = field_data,
            wavelengths = field.wavelengths,
            spacing = field.spacing
        ) 
        
        return out

  
    