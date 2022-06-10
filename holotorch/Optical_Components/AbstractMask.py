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

from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Components.CGH_Component import CGH_Component

class AbstractMask(CGH_Component):
    def __init__(self,
            ) -> None:
        super().__init__()
        
        
    def forward(self, field : ElectricField) -> ElectricField:
        pass