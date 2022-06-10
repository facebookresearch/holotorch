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

from holotorch.ComponentWrapper import PARAM_EXPANDER
import holotorch.utils.Dimensions as Dimensions
from holotorch.Material.MaterialCreator import create_material
from holotorch.HolographicComponents.Hologram_Model import Hologram_Model

def create_expander(expander : PARAM_EXPANDER):
    
    spacing = expander.spacing
    
    num_pixel_x = expander.num_pixel_x
    num_pixel_y  = expander.num_pixel_y

    center_wavelength = expander.center_wavelength
    material = expander.material
    
    material = create_material(material_type=material) 

    holo_type = expander.holo_type
    init_type = expander.init_type

    hologram_dimension = Dimensions.HW(
                height          = num_pixel_x,
                width           = num_pixel_y
    )

    holo_model = Hologram_Model(
        center_wavelength   = center_wavelength,
        hologram_dimension  = hologram_dimension,
        feature_size        = spacing,
        holo_type           = holo_type,
        material            = material,
        init_type           = init_type,
        )
    
    return holo_model