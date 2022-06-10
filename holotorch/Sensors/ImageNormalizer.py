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
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.utils.Enumerators import *

class ImageNormalizer(CGH_Component):

    def __init__(self,
            type_normalize : NORMALIZE_TYPE = NORMALIZE_TYPE.MEAN    
        ) -> None:
        
        super().__init__()
        self.type_normalize = type_normalize

        
    def forward(self, field : IntensityField or ElectricField):
        

        # normalize output by mean intensity
        mean_intensity = field.data.mean(dim=(4,5))
        field.data.div_(mean_intensity[:,:,:,:,None,None])
               
        return field