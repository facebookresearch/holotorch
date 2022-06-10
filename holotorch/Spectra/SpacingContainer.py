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

class SpacingContainer(nn.Module):
    
    def __init__(self,
                spacing : torch.Tensor or float,
                tensor_dimension : TensorDimension = None,
                 ) -> None:
        super().__init__()

        
        if torch.is_tensor(spacing) == False:
            spacing = torch.tensor([[spacing]]).expand([1,1,2])


        if tensor_dimension is None:         
            # Default to time tensor dimension
            tensor_dimension = TCD(n_time = spacing.shape[0],n_channel=spacing.shape[1],height=2)

        self.data_tensor = spacing
        self.tensor_dimension = tensor_dimension

    def detach(self) -> SpacingContainer:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data_tensor = self.data_tensor.detach()

        return SpacingContainer(
            spacing = data_tensor,
            tensor_dimension = self.tensor_dimension,
        )

    def cpu(self) -> SpacingContainer:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data_tensor = self.data_tensor.cpu()

        return SpacingContainer(
            spacing = data_tensor,
            tensor_dimension = self.tensor_dimension,
        )

    @property
    def data_tensor(self) -> torch.Tensor:
        return self._data_tensor
    
    def __str__(self) -> str:
        mystr = "Spacing Container: "
        mystr += str(self.data_tensor.shape)
        mystr += "\n" + str(self.data_tensor)

        return mystr
    
    def __repr__(self):
        return self.__str__()
        
    @data_tensor.setter
    def data_tensor(self, data : torch.Tensor) -> None:
        self.register_buffer("_data_tensor", data)
        
    @property
    def shape(self) -> torch.Size:
        return self.data_tensor.shape

    @property
    def tensor_dimension(self) -> TensorDimension:
        return self._tensor_dimension
    
    @tensor_dimension.setter
    def tensor_dimension(self, dim : TensorDimension) -> None:
        self._tensor_dimension = dim
    
    def check_if_non_square_pixels(self) -> bool:
        pass
    
    def get_spacing_xy(self):
        return self.data_tensor[:,:,0], self.data_tensor[:,:,1]
    
    @property
    def spacing_x(self):
        return self.data_tensor[:,:,0]
    
    @property
    def spacing_y(self):
        return self.data_tensor[:,:,1]
    
    def set_spacing_center_wavelengths(self, spacing):
        self._spacing_center_wavelengths = spacing
    
    def get_spacing_center_wavelengths(self):
        return self._spacing_center_wavelengths
    
    

