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
import warnings

from holotorch.utils.Dimensions import TensorDimension
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.LightSources.GaussianMixture import GaussianMixture
import holotorch.utils.Dimensions as Dimensions

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class Source(CGH_Component):
    """[summary]


    Source is 4D - (T or 1) x C x H X W
    Field out: 1 x (T or 1) x 1 x C x H X W
    
    """    
    
        
    def __init__(self,
            tensor_dimension    : TensorDimension,
            wavelengths         : WavelengthContainer,
            grid_spacing        : SpacingContainer,
            num_gaussians       : int = None
                ) -> None:
        

        super().__init__()
        
        self.tensor_dimension   = tensor_dimension
        self.wavelengths        = wavelengths
        self.grid_spacing       = grid_spacing


        
        if num_gaussians is not None:
            # Color information is currently stored in the time dimension because we do time multiplexing
            num_colors = tensor_dimension.time
            self.gaussian_mixture = GaussianMixture(n_channels=num_colors, num_gaussians=num_gaussians)
        else:
            self.gaussian_mixture = None

        

    def _init_dimensions(self):
        pass

    def __str__(self) -> str:
        
        mystr = super().__str__()
        mystr += "\n-------------------------------------------------------------\n"
    
        return mystr
    
    def __repr__(self):
        return self.__str__()
    
    ##############################################################################
    #
    # SETTER METHODS
    #
    ##############################################################################
    

    @property
    def tensor_dimensions(self) -> TensorDimension:
        return self._tensor_dimension
    
    @tensor_dimensions.setter
    def tensor_dimensions(self, dim : TensorDimension) -> None:
        self._tensor_dimension = dim

    @property
    def num_times(self) -> int:
        return self.tensor_dimension.time

    @property
    def shape(self) -> int:
        return self.tensor_dimension.shape
       
    @property
    def num_channels(self) -> int:
        """ Returns the number of channels used for simulation

        Returns:
            [type]: [description]
        """        
        
        return self.tensor_dimension.channel

    @property
    def height(self) -> int:
        return self.tensor_dimension.height
    
    @property
    def width(self) -> int:
        return self.tensor_dimension.width
       
    ##############################################################################
    #
    # FORWARD METHOD
    #
    ##############################################################################
    
    _scale = 1.0
    
    @property
    def scale(self):
        return self._scale
        
    @scale.setter
    def scale(self, val):
        self._scale = val
    
    def forward(self,
                channel_idxs : torch.Tensor = None
                ) -> ElectricField:
        """[summary]

            Source is 4D - (T or 1) x C x H X W
            Field out: 1 x (T or 1) x 1 x C x H X W
        """
        
        # Get the device type of the model
        device = self.wavelengths._data_tensor.device 
        
        # Calculate the intensity:
        
        if self.gaussian_mixture is None:
            
            out = torch.ones(self.tensor_dimension.shape, device = device)
        else:
            field_shape = [self.height,self.width]
            out = self.gaussian_mixture.forward(field_shape)
            # Make it a 6D-object
            out = out[None,:,None,None,:,:]
        
        # Need to expand for the batch and pupil dimension
        new_shape = self.tensor_dimension.get_new_shape(Dimensions.BTPCHW)
        out = out.view(new_shape)
        
        out = out * self.scale
        
        out = ElectricField(
                data = out,
                wavelengths = self.wavelengths,
                spacing = self.grid_spacing
                )
        
        return out     



