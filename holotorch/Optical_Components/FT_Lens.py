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
import numpy as np
import warnings
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.utils import Dimensions
from holotorch.utils.Helper_Functions import *
from holotorch.utils.units import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class FT_Lens(CGH_Component):

    def __init__(self,
                focal_length,
                distance=None,
                pad = False,
                flag_ifft = False,
                ):

        """
        This takes an image in the rear focal plane and computes the (properly sampled) image in the  Fourier plane
        This implementation rescales the coordinates in the fourier plane and stores the new coordinates as self.image_coords_x and image_coords_y

        Adapted from: Numerical simulation of optical wave simulation , Jason Schmidt 

        Args:

            focal_length(nd.array)      : Focal length of lens (matching units with wavelength)
            wavelength(np.array)        : wavelength


        """

        super().__init__()

        # d is object position behind lens> If none assume object is one focal length
        if distance == None:
            self.distance = focal_length
        else:
            self.distance = distance

       
        self.pad = pad
        self.focal_length = focal_length
        self.flag_ifft = flag_ifft



    def _init_dimensions(self):
        """Creates coordinate system for detector
        """        

    def __str__(self, ):
        """
        Creates an output for the Material class.
        """
        
        mystr = super().__str__()
        mystr += "\n-------------------------------------------------------------\n"
        mystr += "Focal Length: " + str(self.focal_length/mm) + " mm \n"
        mystr += "Padding: " + str(self.pad)

        return mystr
    
    def __repr__(self):
        return self.__str__()

    @staticmethod
    def calculate_output_spacing(
        field : ElectricField,
        focal_length
                ) -> SpacingContainer:
        """Computes the output spacing after the fourier transform

        Args:
            wavelengths (torch.Tensor): [description]
            dx_input (torch.Tensor): [description]
            focal_length ([type]): [description]
            num_pixel (int): [description]

        Returns:
            torch.Tensor: [description]
        """       
        
        num_pixel_x = field.height
        num_pixel_y = field.width
           
        wavelengths_shape = field.wavelengths.tensor_dimension.get_new_shape(Dimensions.TCD)
        wavelengths_expanded = field.wavelengths.data_tensor.view(wavelengths_shape)
        
        center_wavelength_expanded = field.wavelengths.center_wavelength.view(wavelengths_shape)

        if num_pixel_x == num_pixel_y:
            pixel_dim = Dimensions.HW(height=num_pixel_x, width=num_pixel_y)
            pixel_shape = torch.tensor(pixel_dim.shape).to(device=field.data.device)
        else:
            pixel_dim = Dimensions.HW(height=num_pixel_x, width=num_pixel_y)
            pixel_shape = torch.tensor(pixel_dim.shape).to(device=field.data.device)

        dx_input = field.spacing.data_tensor.to(device=field.data.device)
        

        # If we have partially coherent light (i.e. occupied in the wavelength dimension)
        # we need to find a common coordinate system which 
        # we define to be normalized to the respective center wavelengths
        center_wavelength = center_wavelength_expanded.to(device=field.data.device)
        dx_output_center_wavelengths = center_wavelength * np.abs(focal_length) / dx_input / pixel_shape
        
        # Calculate the new coordinate system
        # 
        # NOTE: This is an important equation since it relates the coordinates
        # of a FT between spectral and spatial domain
        #
        dx_output =  wavelengths_expanded * np.abs(focal_length) / dx_input / pixel_shape
        
        dx_output = SpacingContainer(spacing=dx_output, tensor_dimension=Dimensions.TCD)
        
        dx_output.set_spacing_center_wavelengths(dx_output_center_wavelengths)

        return dx_output        

    def set_calculate_output_spacing(self,
            field : ElectricField
                ) -> torch.Tensor:
        """[summary]

        Args:
            wavelengths (torch.Tensor): [description]
        """    
      
        dx_output = FT_Lens.calculate_output_spacing(
            field = field,
            focal_length    = self.focal_length,
                )
        
        return dx_output


    def forward(self,
                field : ElectricField,
                norm = "ortho",
                pad = False
                ) -> ElectricField:
        """
        In this function we apply a phase delay which depends on the local intensity of the electric field at that point

        Args:
            field(torch.complex128) : Complex field (MxN).
            norm(str)               : check pytorch documentation for FT normalization

        """
        
        dx_output = self.set_calculate_output_spacing(
            field = field
        )
        

        data = field.data
        
        if self.flag_ifft:
            my_ft = ift2
        else:
            my_ft = ft2
            
        out = my_ft(
            input = data,
            norm = norm,
            pad = pad
            )


        Eout = ElectricField(
                data=out,
                wavelengths=field.wavelengths,
                spacing = dx_output
                )
        
        return Eout
