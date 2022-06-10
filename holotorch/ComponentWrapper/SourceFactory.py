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
from holotorch.LightSources.Source import Source
from holotorch.LightSources.CoherentSource import CoherentSource
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.utils.Enumerators import *
from holotorch.utils.units import *
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.ComponentWrapper import PARAM_SOURCE
import holotorch.utils.Dimensions as Dimensions
import holotorch.Spectra as Spectra

def create_source(source : PARAM_SOURCE) -> Source:
    
    source_type     = source.source_type 
    color_flag     = source.color_flag 
    spectrum_type   = source.spectrum_type
    
    source_dim      = Dimensions.HW(
            height       = source.height,
            width        = source.width,
    )
    
    if spectrum_type is ENUM_SPECTRUM.NO_SPECTRUM:
        
        source = create_no_spectrum_source(
            color_flag        = color_flag,
            source_dimension  = source_dim,
            grid_spacing      = source.grid_spacing,
            wavelengths       = source.wavelengths
            )
        
    else:
        
    
        source = create_source_from_param(
            source_type       = source_type,
            color_flag        = color_flag,
            spectrum_type     = spectrum_type,
            source_dimension  = source_dim,
            grid_spacing      = source.grid_spacing,
            num_modes_per_center_wavelength = source.num_modes_per_center_wavelength,
            focal_length_collimating_lens              = source.f_col,
            # Diameter of the source
            source_diameter    = source.source_diameter,
            # Standard deviation of the gaussian amplitude
            spatial_coherence_sampling_type = source.spatial_coherence_sampling_type,
            temporal_coherence_sampling_type = source.temporal_coherence_sampling_type,
            wavelengths       = source.wavelengths,
            amplitudes        = source.amplitudes,
            sigma             = source.sigma
        )
    
    return source

def create_no_spectrum_source(
        color_flag : ENUM_SENSOR_TYPE,
        source_dimension : Dimensions.HW,
        grid_spacing : float,
        wavelengths
        ) -> Source:
    
        if not torch.is_tensor(wavelengths):
            wavelengths = torch.tensor(wavelengths)
            
        # NOTE: make sure that grid spacing always has TCD dimensions 
        grid_spacing = SpacingContainer(
                            spacing=torch.tensor([grid_spacing]).expand(1,1,2),
                            tensor_dimension= Dimensions.TCD(1,1,2)
                            )
        
        if color_flag == ENUM_SENSOR_TYPE.BAYER:
            n_time = 3
        elif color_flag == ENUM_SENSOR_TYPE.MONOCHROMATIC:
            n_time = 1
        else:
            raise NotImplementedError("this color flag doesnt exist. please implement or set.")
        
        tensor_dimension = Dimensions.T(n_time = n_time)
        
        wavelengths = WavelengthContainer(
                            wavelengths = wavelengths,
                            tensor_dimension = tensor_dimension,
                            center_wavelength=wavelengths
                            )

        coherent_size = Dimensions.TCHW(
            n_time=wavelengths.tensor_dimension.time,
            n_channel=1,
            height = source_dimension.height,
            width = source_dimension.width,
        )
        
        source = CoherentSource(
            tensor_dimension=coherent_size,
            wavelengths=wavelengths,
            grid_spacing=grid_spacing
        )
        
        
        return source
    
    

def create_source_from_param(
        amplitudes,
        sigma,
        spectrum_type : ENUM_SPECTRUM,
        color_flag : ENUM_SENSOR_TYPE,
        source_type : ENUM_SOURCE_TYPE,
        source_dimension : Dimensions.HW,
        grid_spacing : float,
        num_modes_per_center_wavelength : int,
        wavelengths,
        source_diameter     : float,
        focal_length_collimating_lens : float,
        spatial_coherence_sampling_type : ENUM_SPATIAL_COHERENCE_SAMPLER = ENUM_SPATIAL_COHERENCE_SAMPLER.PLANE_WAVES,
        temporal_coherence_sampling_type : ENUM_TEMPORAL_COHERENCE_SAMPLER = ENUM_TEMPORAL_COHERENCE_SAMPLER.UNIFORM,
        ) -> Source:      


    if spectrum_type is ENUM_SPECTRUM.STANFORD_LED:
        spectrum = Spectra.SpectrumRGB.stanford_LED()
    elif spectrum_type is ENUM_SPECTRUM.STANFORD_SLED:
        spectrum = Spectra.SpectrumRGB.stanford_sLED()
    elif spectrum_type is ENUM_SPECTRUM.PARAMETERIZED:
        spectrum = Spectra.SpectrumRGB.get_analytic(
            center_wavelengths=wavelengths,
            amplitudes=amplitudes,
            sigma_center_wavelengths=sigma
        )
    else:
        raise NotImplementedError("This spectrum type has not been implemented.")

    # NOTE: make sure that grid spacing always has TCD dimensions 
    grid_spacing = SpacingContainer(
                        spacing=torch.tensor([grid_spacing]).expand(1,1,2),
                        tensor_dimension= Dimensions.TCD(1,1,2)
                        )

    # This works because stanford LED exports these
    center_wavelengths = spectrum.center_wavelengths
    sigma_wavelengths = spectrum.sigma_wavelengths
    
    if color_flag is ENUM_SENSOR_TYPE.MONOCHROMATIC:
        center_wavelengths = center_wavelengths[0]
        sigma_wavelengths = sigma_wavelengths[0]
        
        if center_wavelengths.ndim == 0:
            center_wavelengths = torch.tensor([center_wavelengths])
            sigma_wavelengths = torch.tensor([sigma_wavelengths])
        
    if source_type == ENUM_SOURCE_TYPE.COHERENT:

        tensor_dimension = Dimensions.T(n_time = len(center_wavelengths))
        
        wavelengths = WavelengthContainer(
                            wavelengths=center_wavelengths,
                            tensor_dimension=tensor_dimension,
                            )
        
        
        coherent_size = Dimensions.TCHW(
            n_time=wavelengths.tensor_dimension.time,
            n_channel=1,
            height = source_dimension.height,
            width = source_dimension.width,
        )
        
        source = CoherentSource(
            tensor_dimension                = coherent_size,
            wavelengths         = wavelengths,
            grid_spacing        = grid_spacing            
        )

    elif source_type == ENUM_SOURCE_TYPE.PARTIALLY_COHERENT:

        from holotorch.LightSources.PartialCoherentSource import PartialCoherentSource

        source = PartialCoherentSource(
            tensor_dimension    = source_dimension,
            spectrum            = spectrum,
            num_modes_per_center_wavelength = num_modes_per_center_wavelength,
            grid_spacing        = grid_spacing,
            source_diameter     = source_diameter,
            focal_length_collimating_lens = focal_length_collimating_lens,
            spatial_coherence_sampling_type = spatial_coherence_sampling_type,
            temporal_coherence_sampling_type = temporal_coherence_sampling_type,
        )
        
    return source