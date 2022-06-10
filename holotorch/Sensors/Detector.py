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

from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Sensors.IncoherentAverager import IncoherentAverager
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.utils.Helper_Functions import *
from holotorch.Sensors.IntensityOperator import IntensityOperator
from holotorch.Sensors.CoordinateSystemCombiner import CoordinateSystemCombiner
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.CGH_Datatypes.IntensityField import IntensityField
from holotorch.Optical_Components.Resize_Field import Resize_Field
from holotorch.utils import Dimensions
from holotorch.utils.Enumerators import *
from holotorch.utils.units import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class Detector(CGH_Component):


    def __init__(self,
                color_flag : ENUM_SENSOR_TYPE,
                N_pixel_out_x : int,
                RGB_multiplexing : ENUM_TIME_MULTIPLEXING = ENUM_TIME_MULTIPLEXING.ONE_SHOT,
                N_pixel_out_y : int     = None,
                zero_pad_flag : bool    = False
                ):
        super().__init__()

        if N_pixel_out_y is None:
            N_pixel_out_y = N_pixel_out_x
        
        self.color_flag         = color_flag
        self.RGB_multiplexing   = RGB_multiplexing

        self.zero_pad_flag      = zero_pad_flag

        self.n_channels_out = Detector.compute_output_dimension(
                        color_flag= self.color_flag,
                        )
        # Scale_ratios provides the ratios between the
        # lowest and longest wavelengths
        # Which then will define how much resizing we 
        # need for each wavelength

        self.intensity_operator = IntensityOperator()
        self.coordinateSystemCombiner = CoordinateSystemCombiner()

        #---------------------------------------------------------------------------
        # Downsampling operator
        #---------------------------------------------------------------------------
        self.set_pooling_operator(N_pixel_out_x, N_pixel_out_y)

        #---------------------------------------------------------------------------
        # optional zero padding before computing intensity
        #---------------------------------------------------------------------------
        if self.zero_pad_flag:
            self.RF = Resize_Field(scale_factor=2, frequency_domain_flag=True)
            
        self.wavelength_integrator = IncoherentAverager()

    def set_pooling_operator(self,
            N_pixel_out_x : int,
            N_pixel_out_y : int
                             ):
        input_size = [N_pixel_out_x, N_pixel_out_y]
        self.N_pixel_out_x = N_pixel_out_x
        self.N_pixel_out_y = N_pixel_out_y
        self.pool = torch.nn.AdaptiveAvgPool2d(input_size)

    def _init_dimensions(self):
        """Creates coordinate system for detector
        """        

        

    def forward(self, field : ElectricField) -> IntensityField:
        """
        Input: # B x T x P x C x H X W
        Output: # B x P x C x H X W

        """
        orig_wavelengths = field.wavelengths

        # zero pad in the frequency domain before computing intensity
        if self.zero_pad_flag:
            field = self.RF(field)
        #---------------------------------------------------------------------------
        # Calculate the intensity
        #---------------------------------------------------------------------------
        
        intensity = self.intensity_operator(field)

        #---------------------------------------------------------------------------
        # Account for possible changes in coordinate systems
        # NOTE: coordinateSystemCombiner also performs relative scaling
        # NOTE: After coordinateSystemCombiner
        #---------------------------------------------------------------------------
        
        intensity = self.coordinateSystemCombiner.forward(intensity)

        #---------------------------------------------------------------------------
        # Wavelength integrator
        #---------------------------------------------------------------------------
        if isinstance(orig_wavelengths.tensor_dimension,Dimensions.C):
            pass
        else:
            intensity = self.wavelength_integrator(intensity)
        
        data = intensity.data
        
        # Get the shape of date
        B, T, P, C, H , W = data.shape
        
        # Transform 6D into 4D for pooling
        data = data.view(B*P*C,T,H,W)
        # Apply the pool
        output = self.pool(data)
        # Transform 4D back into 6D (NOTE: Channel dimension is one here)
        output = output.view(B,T,P,C,self.N_pixel_out_x,self.N_pixel_out_y)
        
        # PERMUTE TIME and CHANNEL dimension for now so that 
        # Time-multiplexed RGB image goes into the Channel dimension
        # NOTE: This should probably be handled different in the future
        
        if self.RGB_multiplexing == ENUM_TIME_MULTIPLEXING.TIME_MULTIPLEX and ( self.color_flag == ENUM_SENSOR_TYPE.BAYER or self.color_flag == ENUM_SENSOR_TYPE.TIME_MULTIPLEXED):
            output = output.permute(
                    field.BATCH,
                    field.CHANNELS,
                    field.PUPIL,
                    field.TIMES,
                    field._HEIGHT,
                    field._WIDTH
                )
        
        new_dx = intensity.spacing.data_tensor[0] * self.N_pixel_out_x / H
        new_dy = intensity.spacing.data_tensor[1] * self.N_pixel_out_y / W
        
        new_spacing = SpacingContainer(torch.tensor([new_dx,new_dy]), Dimensions.H(height = 2))
        
                
        output = IntensityField(
                data = output,
                wavelengths = intensity.wavelengths,
                spacing = new_spacing
            )
        
        return output
        
    @property
    def dx_input(self) -> torch.Tensor:
        """Returns the magnitude of the pupil objects


        Returns:
            torch.Tensor: [description]
        """    
        return self._dx_input

    @dx_input.setter
    def dx_input(self, data : torch.Tensor) -> None:
        self._dx_input = data
        
    @staticmethod
    def compute_output_dimension(
            color_flag : ENUM_SENSOR_TYPE,
            ) -> int:
        
        if color_flag == ENUM_SENSOR_TYPE.BAYER or color_flag == ENUM_SENSOR_TYPE.TIME_MULTIPLEXED :
            return 3
        if color_flag == ENUM_SENSOR_TYPE.MONOCHROMATIC:
            return 1
                
        else:
            raise NotImplementedError(
                "This sensor type is not yet implemented")