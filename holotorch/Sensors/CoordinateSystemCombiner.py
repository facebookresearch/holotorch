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
from torchvision import transforms

from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.CGH_Datatypes.IntensityField import IntensityField
from holotorch.utils import Dimensions

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class CoordinateSystemCombiner(CGH_Component):


    def __init__(self,
                ):
        super().__init__()

    def _init_dimensions(self):
        """Creates coordinate system for detector
        """        

    def forward(self, intensity : IntensityField) -> IntensityField:
        """
        Input: # B x T x P x C x H X W
        Output: # B x P x C x H X W

        """

        #---------------------------------------------------------------------------
        # Scale the outputs
        #
        # NOTE: This might happen if the output fields are multdimensional
        # e.g. multi-spectral and the outputgrid live on different coordinate systems
        # This can e.g. happen with a fourier transform lens.
        #
        #---------------------------------------------------------------------------

        # NOTE: resize input to 4D for interpolation
        B, T, P, C, N_pixel_in_x, N_pixel_in_y = intensity.shape        
        data = intensity.data
        
        new_shape = intensity.wavelengths.tensor_dimension.get_new_shape(Dimensions.TC)
        wavelengths = intensity.wavelengths.data_tensor.view(new_shape)
        
        dx_input = intensity.spacing.data_tensor[:,:,0]
        dy_input = intensity.spacing.data_tensor[:,:,1]
        
        spacing_center = intensity.spacing.get_spacing_center_wavelengths()
        dx_center = spacing_center[:,:,0]
        dx_center = spacing_center[:,:,1]

        min_dx = dx_center.min()
        min_dy = dx_center.min()

        dx_input = dx_input.expand(new_shape)
        dy_input = dy_input.expand(new_shape)

        # Create the Crop-Functions which will happen after the interpolation
        crop_shape = [intensity.height, intensity.width]
        crop = transforms.CenterCrop(crop_shape)

        # Iterative over all wavelength
        for idx_time in range(wavelengths.shape[0]):
            for idx_lambda in range(wavelengths.shape[1]):

                # Compute the scale factor
                scale_factor_x = dx_input[idx_time,idx_lambda] / min_dx        
                scale_factor_y = dy_input[idx_time,idx_lambda] / min_dy        
               
                input = data[:,idx_time,:,idx_lambda]
                
                assert (scale_factor_x - scale_factor_y).abs() < 0.01
                
                scale_factor = scale_factor_x
                
                if scale_factor == 1:
                    output = input
                else:
                    output =  torch.nn.functional.interpolate(
                                    input,
                                    scale_factor = float(scale_factor),
                                    mode = "bilinear"
                                    )
                    output = crop(
                                    output
                                )   

                # Update the intensities
                data[:,idx_time,:,idx_lambda] = output
        
        new_spacing = SpacingContainer(torch.tensor([min_dx,min_dy]), Dimensions.H(height = 2))
        
        size_x_mm = N_pixel_in_x * new_spacing.data_tensor[0]
        size_y_mm = N_pixel_in_y * new_spacing.data_tensor[1]
        
        if isinstance(intensity.wavelengths.tensor_dimension,Dimensions.T):
            new_tensor_dimension = Dimensions.T(n_time = intensity.wavelengths.tensor_dimension.time)
        elif isinstance(intensity.wavelengths.tensor_dimension,Dimensions.C):
            new_tensor_dimension = Dimensions.C(n_channel = intensity.wavelengths.tensor_dimension.channel)
        
        
        new_wavelengths, _ = wavelengths.max(axis=1)
        
        WavelengthContainer(
            wavelengths=new_wavelengths,
            tensor_dimension=new_tensor_dimension,
            center_wavelength = intensity.wavelengths.center_wavelength
            )

        # After resizing the physical size in x-dimension and y-dimension should be the same
        # NOTE: This is only true if there has been a full FFT involved
        # assert size_x_mm == size_y_mm
        
        output = IntensityField(
                data = data,
                wavelengths = new_wavelengths,
                spacing = new_spacing
            )
        
        return output

        