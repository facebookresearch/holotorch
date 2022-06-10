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
from holotorch.utils.Dimensions import BTPCHW, TensorDimension
from holotorch.Optical_Components.AbstractMask import AbstractMask
from holotorch.utils.Enumerators import *

class SimpleMask(AbstractMask):
    def __init__(self,
            tensor_dimension : TensorDimension               = None,
            init_type : INIT_TYPE                            = INIT_TYPE.ZEROS,
            mask_model_type     : MASK_MODEL_TYPE            = MASK_MODEL_TYPE.REAL,
            mask_forward_type        : MASK_FORWARD_TYPE     = MASK_FORWARD_TYPE.MULTIPLICATIVE
            ) -> None:
        super().__init__()
        
        self.add_attribute('mask')
        
        self.mask_opt = True
        
        self.tensor_dimension = tensor_dimension

        if tensor_dimension is not None:
            if init_type == INIT_TYPE.ZEROS:
                mask = torch.zeros(tensor_dimension.shape)
            elif init_type == INIT_TYPE.ONES:
                mask = torch.ones(tensor_dimension.shape)

            if mask_model_type == MASK_MODEL_TYPE.COMPLEX:
                mask = mask + 0j

            self.mask = mask
        
        self.mask_model_type = mask_model_type

        self.mask_forward_type = mask_forward_type
        
        self.spacing = None
        
    def forward(self, field : ElectricField) -> ElectricField:

        if self.mask_forward_type is MASK_FORWARD_TYPE.MULTIPLICATIVE:
            new_field = field.data * self.mask
        elif self.mask_forward_type is MASK_FORWARD_TYPE.ADDITIVE:
            new_field = field.data + self.mask

        field = ElectricField(
            data        = new_field,
            spacing     = field.spacing,
            wavelengths = field.wavelengths            
        )
        
        return field
    
    def visualize(self,
            figsize=(14,4),
            x0 = None,
            y0 = None,
            width = None,
            height = None,
            vmax = None,
            vmin = None,
            flag_axis : bool = True,
            flag_colorbar = False,
            adjust_aspect : bool = False,
            title = None
            ):
        
        self.renitialize_attributes()
        plt.figure(figsize=figsize)


        new_shape = self.tensor_dimension.get_new_shape(BTPCHW)

        mask = self.mask.expand(new_shape)

        field_mask = ElectricField(
            data        = mask,
            spacing     = self.spacing,
        )

        plt.subplot(121)
        mytitle = "Amplitude"
        if title is not None:
            mytitle = title + ": " + mytitle

        field_mask2 = field_mask
        if torch.is_complex(self.mask):
            field_mask2 = field_mask2.abs()

        field_mask2.abs().visualize(
            flag_axis  = flag_axis,
            title   = mytitle ,
            flag_colorbar= flag_colorbar,
            adjust_aspect = adjust_aspect
        )

        if torch.is_complex(self.mask):

            plt.subplot(122)
            mytitle = "Phase (deg)"      
            if title is not None:
                mytitle = title + ": " + mytitle

            field_mask.angle().visualize(
                flag_axis  = flag_axis,
                title = mytitle,
                flag_colorbar= flag_colorbar,
                adjust_aspect = adjust_aspect
            )

        plt.tight_layout()
