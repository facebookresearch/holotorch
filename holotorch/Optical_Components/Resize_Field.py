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
from torch.types import _size
from torch.nn.functional import pad
import warnings

from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.utils.Dimensions import *
from holotorch.utils.Helper_Functions import * 
import holotorch.utils.transformer_6d_4d as transformer_6d_4d
from holotorch.CGH_Datatypes.ElectricField import ElectricField

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
    
class Resize_Field(torch.nn.Module):
    
    def __init__(self,
                 scale_factor : float or int    = None, 
                 size : _size                   = None ,
                 recompute_scale_factor : bool  = False ,
                 mode : str                     = "bicubic",
                 frequency_domain_flag : bool   = False  
                 ):
        """
        Resizes a field 
        Parameters
        ==========
        scale_factor   : float
                        resizing scale factor - optional with size (see torch documentation for interpolate)
                           
        size           :   float
                        new field size - optional with scale factor
                          
        recompute_scale_factor  : bool                  
                          
                       

        """
        super().__init__()

        self.scale_factor           = scale_factor
        self.recompute_scale_factor = recompute_scale_factor
        self.size                   = size
        self.mode                   = mode
        self.frequency_domain_flag  = frequency_domain_flag

    def zero_pad_frequency(self, field_data : torch.Tensor) -> torch.Tensor:
        _, _, H, W = field_data.shape
        pad_y = int(self.scale_factor * H/2)
        pad_x = int(self.scale_factor * W/2)

        field_data = ft2(field_data)
        field_data = pad(field_data, (pad_x,pad_x,pad_y,pad_y), mode='constant', value=0)
        field_data = ift2(field_data)

        return field_data

    def interpolate_spatial(self, field_data : torch.Tensor) -> torch.Tensor:
        
        if torch.is_complex(field_data):

            B=torch.nn.functional.interpolate(field_data.real, scale_factor =self.scale_factor, size =self.size, mode=self.mode ) 
            Bi=torch.nn.functional.interpolate(field_data.imag, scale_factor =self.scale_factor, size =self.size, mode=self.mode ) 
            Eout =   B + 1j*Bi
        else:
            Eout=torch.nn.functional.interpolate(field_data, scale_factor =self.scale_factor, size =self.size, mode=self.mode ) 

        return Eout

    def forward(self, field : ElectricField or torch.Tensor) -> ElectricField:
        """
        In this function we interpolate a complex field to a new grid
        
        Parameters
        ==========
        field            : torch.complex128
                           Complex field (MxN).

                       
        """
        is_field = True
        if isinstance(field, torch.Tensor):
            tmp_shape = field.shape       
            data = field
            is_field = False
        else:
            tmp_shape = field.data.shape       
            data = field.data

        if data.ndim == 2:
            tmp_shape = [1,1,1,1,*tmp_shape]
            data = data[None, None, None, None,:,:]
        if data.ndim == 4:
            tmp_shape = [1,1,*tmp_shape]
            data = data[None, None, :,:,:,:]
        if data.ndim == 5:
            tmp_shape = [1,*tmp_shape]
            data = data[None, :,:,:,:,:]

        tmp_field = transformer_6d_4d.transform_6D_to_4D(tensor_in=data)

        # perform interpolation in the spatial domain or zero padding in frequency domain
        if self.frequency_domain_flag:
            Eout = self.zero_pad_frequency(tmp_field)       
        else:
            Eout = self.interpolate_spatial(tmp_field)       

        # release the temporary field from gpu memory
        del tmp_field
        torch.cuda.empty_cache()

        new_shape = (*tmp_shape[:4],*Eout.shape[-2:])
        Eout = transformer_6d_4d.transform_4D_to_6D(tensor_in=Eout, newshape = new_shape)

        if field.data.ndim == 2:
            Eout = Eout.squeeze(dim=0).squeeze(dim=0).squeeze(dim=0).squeeze(dim=0)
        if field.data.ndim == 4:
            Eout = Eout.squeeze(dim=0).squeeze(dim=0)
        if field.data.ndim == 5:
            Eout = Eout.squeeze(dim=0)

        if is_field == False:
            return Eout
        
        if self.scale_factor is None:
            scale_factor = self.size[0] / field.height
        else:
            scale_factor = self.scale_factor
        
        
        if field.spacing is not None:
            new_spacing = field.spacing.data_tensor / scale_factor
        
            new_spacing = SpacingContainer(
                            spacing = new_spacing,
                            tensor_dimension= H(height=len(new_spacing)))
        else:
            new_spacing = field.spacing
        
        Eout = ElectricField(
            data = Eout,
            wavelengths=field.wavelengths,
            spacing = new_spacing
        )
        return Eout 


