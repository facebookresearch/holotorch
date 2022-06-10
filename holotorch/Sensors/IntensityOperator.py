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

import warnings

from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.CGH_Datatypes.IntensityField import IntensityField

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class IntensityOperator(CGH_Component):

    def __init__(self):
        super().__init__()
    
    def _init_dimensions(self):
        pass

    def forward(self,
            field : ElectricField,
            ) -> IntensityField:
        """
        Input: # B x T x P x C x H X W
        Output: # B x P x C x H X W

        """

        #---------------------------------------------------------------------------
        # Calculate the intensity
        #---------------------------------------------------------------------------

        intensity = field.data.abs().pow(2)        
        
        field = IntensityField(
            data        = intensity,
            spacing     = field.spacing,
            wavelengths = field.wavelengths
        )
        
        return field
