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
from torch.types import _size

from holotorch.CGH_Datatypes.Light import Light
from holotorch.CGH_Datatypes.IntensityField import IntensityField
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.utils.Enumerators import *

class ElectricField(Light):

    def __init__(self, 
                data : torch.Tensor,
                wavelengths : WavelengthContainer or float = None,
                spacing     : SpacingContainer or float = None,
                requires_grad = None,
                identifier : FIELD_IDENTIFIER = FIELD_IDENTIFIER.NOTHING,
                **param) -> ElectricField:

        super().__init__(
            data = data,
            wavelengths = wavelengths,
            spacing = spacing,
            identifier=identifier
            )
        

    def like(self, data : torch.Tensor) -> ElectricField:

        assert data.ndim == 6

        return ElectricField(
            data = data,
            wavelengths = self.wavelengths,
            spacing = self.spacing,
            identifier = self.identifier
        )
    
    
    def __getitem__(self, keys) -> ElectricField:
        
        mylight = super().__getitem__(keys)
        
        return self.like(data=mylight.data)

    
    @staticmethod
    def zeros(size : _size,
              wavelengths : torch.Tensor,
              spacing : torch.Tensor,
              **param):
        """[summary]

        Args:
            size (_size): [description]
            wavelengths ([type]): [description]
            device (_device, optional): [description]. Defaults to None.
            dtype (_dtype, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """        
        assert wavelengths is not None, "wavelengths cannot be None"
        tmp = torch.zeros(size = size, **param)
        return ElectricField(
            data = tmp,
            wavelengths=wavelengths,
            spacing = spacing,
            **param)
    
    @staticmethod
    def ones(
            size : _size,
            wavelengths : torch.Tensor,
            spacing : torch.Tensor,
            **param):
        """[summary]

        Args:
            size (_size): [description]
            wavelengths ([type]): [description]
            device (_device, optional): [description]. Defaults to None.
            dtype (_dtype, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """        
        assert wavelengths is not None, "wavelengths cannot be None"

        tmp = torch.ones(size = size, **param)
        
        return ElectricField(
            data = tmp,
            wavelengths=wavelengths,
            spacing = spacing,
            **param
            )
   

        
    def detach(self) -> ElectricField:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data = self.data.detach()
        wavelengths = self.wavelengths.detach()
        spacing = self.spacing.detach()
        
        return ElectricField(data = data, wavelengths=wavelengths, spacing = spacing, identifier=self.identifier)
  
    def cpu(self) -> ElectricField:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data = self.data.cpu()
        wavelengths = self.wavelengths.cpu()
        spacing = self.spacing.cpu()
        
        return ElectricField(data = data, wavelengths=wavelengths, spacing = spacing, identifier=self.identifier)
    
    def abs(self) -> IntensityField:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data = self.data.abs()
        wavelengths = self.wavelengths   
        spacing = self.spacing
             
        return IntensityField(data = data, wavelengths=wavelengths, spacing = spacing, identifier=self.identifier)    
    

    def angle(self) -> IntensityField:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data = self.data.angle()
        wavelengths = self.wavelengths   
        spacing = self.spacing
             
        return IntensityField(data = data, wavelengths=wavelengths, spacing = spacing, identifier=self.identifier)    
    
    def get_intensity(self):
        """Returns the intensity field from the Electric Field

        Returns:
            Intensity Field: The intensity field after the abs() operation
        """        
        data = self.data.abs()
        
        new_intensity_field = IntensityField(
            data=data,
            wavelengths=self.wavelengths,
            spacing = self.spacing,
            identifier=self.identifier
        )
        
        return new_intensity_field

    def visualize_time_series_gif(self,
                    figsize=(5,3),
                    ):       
        
        new_intensity_field = self.get_intensity()


        return new_intensity_field.visualize_time_series_gif()

    
    def visualize(self,
                  title: str = "",
                  flag_colorbar: bool = True,
                  flag_axis: str = False, 
                  cmap='gray',
                  index=None,
                  open_new_figure=False,
                  figsize=None,
                  vmax=None,
                  vmin=None,
                  plot_type : ENUM_PLOT_TYPE = ENUM_PLOT_TYPE.MAGNITUDE,
                  flag_log : bool = False,
                  adjust_aspect : bool    = False,
                  rescale_factor : float = 1,
                    ):       
        
        if plot_type is ENUM_PLOT_TYPE.MAGNITUDE:
            new_intensity_field = self.get_intensity()
        elif plot_type is ENUM_PLOT_TYPE.PHASE: 
            new_intensity_field = self.angle()
            
        if flag_log == True:
            new_intensity_field = new_intensity_field.log()
            
        new_intensity_field.visualize(
            title = title,
            flag_colorbar = flag_colorbar,
            flag_axis = flag_axis,
            cmap = cmap,
            index = index,
            open_new_figure = open_new_figure,
            figsize = figsize,
            vmax = vmax,
            vmin = vmin,
            adjust_aspect = adjust_aspect,
            rescale_factor = rescale_factor,
        )

    def visualize_grid_gif (self,
            flag_axis       = "off",
            time_idx        = 0,
            suptitle        = "Collection",
            flag_colorbar   = True,
            title           = None,
            max_images      = 9,
            figsize         = (12,7),
            cmap            = 'gray'
                                   ) -> None:
        new_intensity_field = self.get_intensity()

        return new_intensity_field.visualize_grid_gif(
            flag_axis       = flag_axis,
            time_idx        = time_idx,
            suptitle        = suptitle,
            flag_colorbar   = flag_colorbar,
            title           = title,
            max_images      = max_images,
            figsize         = figsize,
            cmap            = cmap
        )
    

    def visualize_time_series_grid(self,
            flag_axis       = "off",
            time_idx        = 0,
            suptitle        = "Collection",
            flag_colorbar   = True,
            title           = None,
            max_images      = 9,
            figsize         = (12,7),
            cmap            = 'gray'
                                   ) -> None:
        new_intensity_field = self.get_intensity()

        return new_intensity_field.visualize_time_series_grid(
            flag_axis       = flag_axis,
            time_idx        = time_idx,
            suptitle        = suptitle,
            flag_colorbar   = flag_colorbar,
            title           = title,
            max_images      = max_images,
            figsize         = figsize,
            cmap            = cmap
        )
