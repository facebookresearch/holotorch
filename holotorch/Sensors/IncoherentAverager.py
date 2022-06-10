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


from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.CGH_Datatypes.IntensityField import IntensityField

class IncoherentAverager(CGH_Component):

    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, field : IntensityField):
                        
        # Integrate over the wavelength axis to collapse each center wavelength
        data = field.data.mean(axis=3)
        # Adds the 4th axis again to recreate the 6D tensor
        data = data[:,:,:,None,:,:]
        
        out = IntensityField(
                data = data,
                wavelengths=field.wavelengths,
                spacing = field.spacing
                )
        
        return out