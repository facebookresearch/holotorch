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
import torch.nn as nn

from holotorch.utils.Dimensions import *

class WavelengthContainer(nn.Module):
    
    def __init__(self,
                 wavelengths      : torch.Tensor or float,
                 tensor_dimension : TensorDimension = None,
                 center_wavelength : torch.Tensor = None
                 ) -> None:
        super().__init__()
        
        if torch.is_tensor(wavelengths) == False:
            if isinstance(wavelengths, list):
                wavelengths = torch.tensor(wavelengths)
            else:
                wavelengths = torch.tensor([wavelengths])
        elif wavelengths.ndim == 0:
            wavelengths = torch.tensor([wavelengths])

            
        if tensor_dimension is None:
            if wavelengths.ndim != 1:
                raise ValueError("If wavelengths dimension is larger than 1, we need to provide tensor dimension information.")
            
            # Default to time tensor dimension
            tensor_dimension = T(n_time = wavelengths.shape[0])
            
        self.data_tensor = wavelengths
        
        self.tensor_dimension = tensor_dimension
        
        if center_wavelength is None:
            # If there's no center wavelength given the center
            # wavelength just corresponds to to the wavelength tensor given
            center_wavelength = wavelengths

        # Tensors should be one dimension so that they provide a shape
        if center_wavelength.ndim == 0:
                center_wavelength = torch.tensor([center_wavelength])
                
        self.center_wavelength = center_wavelength
        
    def detach(self) -> WavelengthContainer:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data_tensor = self.data_tensor.detach()

        return WavelengthContainer(
            wavelengths = data_tensor,
            tensor_dimension = self.tensor_dimension,
            center_wavelength = self.center_wavelength
        )

        
    def cpu(self) -> WavelengthContainer:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data_tensor = self.data_tensor.cpu()

        return WavelengthContainer(
            wavelengths = data_tensor,
            tensor_dimension = self.tensor_dimension,
            center_wavelength = self.center_wavelength
        )

    def __str__(self) -> str:
        return "WaveLengthContainer: "  + str(self.data_tensor.cpu().numpy() / nm) + "nm"
    
    def __repr__(self):
        return self.__str__()
    
    @property
    def time(self):
        return self.tensor_dimension.time

    @property
    def channel(self):
        return self.tensor_dimension.channel

    @property
    def data_tensor(self) -> torch.Tensor:
        return self._data_tensor
        
    @data_tensor.setter
    def data_tensor(self, data : torch.Tensor) -> None:
        self.register_buffer("_data_tensor", data)
        
    def write_data_tensor(self, data : torch.Tensor):
        
        # Ensure that data has the same type and cast if necessary
        data = data.type_as(self.data_tensor)
        
        with torch.no_grad():
            self._data_tensor.copy_(data)
        
    @property
    def shape(self) -> torch.Size:
        return self.data_tensor.shape

    @property
    def tensor_dimension(self) -> TensorDimension:
        return self._tensor_dimension
    
    @tensor_dimension.setter
    def tensor_dimension(self, dim : TensorDimension) -> None:
        self._tensor_dimension = dim
        
    def forward(self):
        
        new_shape = self.tensor_dimension.get_new_shape(new_dim = BTPC)
        
        expanded_wavelengths = self.data_tensor.view(new_shape)
        
        return expanded_wavelengths

        
