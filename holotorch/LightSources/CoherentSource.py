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
import warnings

from holotorch.utils.Dimensions import TensorDimension
from holotorch.utils.Helper_Functions import *
from holotorch.utils.units import *
import holotorch.utils.Dimensions as Dimensions
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.Spectra.SpacingContainer import SpacingContainer
from .Source import Source

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class CoherentSource(Source):

    def __init__(self,
            tensor_dimension    : TensorDimension,
            wavelengths         : WavelengthContainer or torch.Tensor,
            grid_spacing        : SpacingContainer
                ) -> None:

        if type(wavelengths) is not WavelengthContainer:
            wavelengths_dim = Dimensions.TC(
                n_time=tensor_dimension.time,
                n_channel=tensor_dimension.channel
            )
            wavelengths = WavelengthContainer(wavelengths=wavelengths,
                                              tensor_dimension=wavelengths_dim)
            
            
        super().__init__(
            tensor_dimension    = tensor_dimension,
            wavelengths         = wavelengths,
            grid_spacing        = grid_spacing
        )

    @classmethod
    def create(cls,
        height : int,
        width  : int,
        spacing : float = 8 * um,
        wavelengths : tuple = [432 * nm, 530 * nm, 630 * nm],
    ) -> CoherentSource:

        # Define the wavelengths
        wavelength_container = WavelengthContainer(
            wavelengths = [432 * nm, 530 * nm, 630 * nm],
            tensor_dimension = Dimensions.C(n_channel=3)
            # We need to tell the container at which dimension the data should work on
            # This resolves confusion for more complicates tasks (such as partial coherence)
            )

        # Define the spacing
        spacing_container = SpacingContainer(
            spacing = 8 * um
            )

        # The spacing will automatically extend to other dimensions (by default Time,Channel and xy)
        # (this is required for more complicated setups and happens "behind the scenes")
                
        # Defines the pixel resolution of the source
        source_dim      = Dimensions.CHW(
                n_channel    = wavelength_container.channel,
                height       = 1000,
                width        = 1400,
            )

        source = CoherentSource(
            tensor_dimension = source_dim,
            wavelengths      = wavelength_container,
            grid_spacing     = spacing_container
            )

        #print(wavelength_container)
        #print("Spacing", spacing_container)
        #print("Spacing", spacing_container.shape) 
        #print("Source Dim: ", source_dim)
        #print(source)
        #print("Source Shape", source_out.shape)

        # Get the output of the source
        return source

