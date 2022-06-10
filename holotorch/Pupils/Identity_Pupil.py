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

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

class Identity_Pupil(CGH_Component):
    """
    
    """
    
    def __init__(self,
            ) -> None:

        super().__init__()
        self.dimension = 1
        self.input_dimensions = 1

    def _init_dimensions(self):
                     
        self.output_dimensions = 1

    def forward(self, field : ElectricField) -> ElectricField:    
        """[summary]

        Args:
            field (ElectricField): [description]

        Returns:
            ElectricField: [description]
        """        
        new_field = field
        return new_field