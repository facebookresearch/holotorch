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
from torch.types import _device, _dtype, _size

from holotorch.CGH_Datatypes.Light import Light
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.Spectra.SpacingContainer import SpacingContainer

from holotorch.utils.Enumerators import *

class IntensityField(Light):

    def __init__(self, 
                data : torch.Tensor,
                wavelengths : WavelengthContainer = None,
                spacing : SpacingContainer = None,
                requires_grad = None,
                identifier : FIELD_IDENTIFIER = FIELD_IDENTIFIER.NOTHING,
                ) -> IntensityField:
        
        assert torch.is_tensor(data), "Data must be a torch tensor"

        super().__init__(
            data = data,
            wavelengths = wavelengths,
            spacing = spacing,
            identifier= identifier
            )
                


    def like(self, data : torch.Tensor) -> IntensityField:

        assert data.ndim == 6

        return IntensityField(
            data = data,
            wavelengths = self.wavelengths,
            spacing = self.spacing,
            identifier = self.identifier
        )
    

    def __getitem__(self, keys) -> IntensityField:
        
        mylight = super().__getitem__(keys)
        
        
        return self.like(data=mylight.data)

    def __sub__(self, other) -> IntensityField:
        
        new_data = super().__sub__(other)
        
        return self.like(data=new_data.data)

        
    def __add__(self, other : IntensityField) -> IntensityField:
        
        new_data = super().__add__(other)
        
        return self.like(data=new_data.data)

        
    def __mul__(self, other : IntensityField) -> IntensityField:
        
        new_data = super().__mul__(other)
        
        return self.like(data=new_data.data)


    def detach(self) -> IntensityField:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data = self.data.detach()
        
        if self.wavelengths is not None:
            wavelengths = self.wavelengths.detach()
        else:
            wavelengths = self.wavelengths
        
        return IntensityField(data = data, wavelengths=wavelengths,            spacing= self.spacing, identifier=self.identifier
        )

    def log(self) -> IntensityField:
        
        data = self.data.log()
        
        return self.like(data=data.data)



    def cpu(self) -> IntensityField:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data = self.data.cpu()
        
        if self.wavelengths is not None:
            wavelengths = self.wavelengths.cpu()
        else:
            wavelengths = self.wavelengths
        
        return IntensityField(data = data, wavelengths=wavelengths,             spacing= self.spacing, identifier= self.identifier
        )
    
    @staticmethod
    def zeros(size : _size, wavelengths, device : _device = None ,dtype : _dtype = None, **param):
        """[summary]

        Args:
            size (_size): [description]
            wavelengths ([type]): [description]
            device (_device, optional): [description]. Defaults to None.
            dtype (_dtype, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """        
        assert wavelengths is not None, "Wavelenghts cannot be None"
        tmp = torch.zeros(size = size,dtype=dtype, device=device, **param)
        return IntensityField(data = tmp, wavelengths=wavelengths, **param)
    
    @staticmethod
    def ones(size : _size, wavelengths, device : _device = None ,dtype : _dtype = None, **param):
        """[summary]

        Args:
            size (_size): [description]
            wavelengths ([type]): [description]
            device (_device, optional): [description]. Defaults to None.
            dtype (_dtype, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """        
        assert wavelengths is not None, "Wavelenghts cannot be None"

        tmp = torch.ones(size = size,dtype=dtype, device=device, **param)
        return IntensityField(data = tmp, wavelengths=wavelengths, **param)