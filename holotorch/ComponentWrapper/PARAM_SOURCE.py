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

from holotorch.ComponentWrapper.PARAM_COMPONENT import PARAM_COMPONENT
from holotorch.utils.Enumerators import *
from holotorch.utils.units import *
import torch
class PARAM_SOURCE(PARAM_COMPONENT):


    def __init__(self) -> None:
        super().__init__()
        
    spectrum_type       = ENUM_SPECTRUM.NO_SPECTRUM 
    source_type         = None
    
    color_flag          = ENUM_SENSOR_TYPE.MONOCHROMATIC
    source_type         = ENUM_SOURCE_TYPE.COHERENT
    height              = None
    width               = None
    grid_spacing        = None
    num_modes_per_center_wavelength = 1
    
    wavelengths         = torch.tensor([459, 532, 633])*nm
    bandwidth           = 10*nm
    sigma               = torch.tensor([bandwidth, bandwidth, bandwidth])
    amplitudes          = torch.tensor([1.0, 1.0, 1.0])
    
    # Collimating lens focal length
    f_col              = 100 * mm
    # Diameter of the source
    source_diameter    = 100 * um
    # Standard deviation of the gaussian amplitude
    source_amp_sigma   = 10  * um

    spatial_coherence_sampling_type : ENUM_SPATIAL_COHERENCE_SAMPLER = ENUM_SPATIAL_COHERENCE_SAMPLER.PLANE_WAVES
    temporal_coherence_sampling_type : ENUM_TEMPORAL_COHERENCE_SAMPLER = ENUM_TEMPORAL_COHERENCE_SAMPLER.UNIFORM