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

import sys
sys.path.append('../')

import numpy as np
import torch

from holotorch.Material.CGH_Material import CGH_Material

class DefaultMaterial(CGH_Material):
    """Represents a Material class
    """    
    
    # Name of the material
    name = None

    # The model parameters
    refractive_index = None # A-parameter in Cauchy Model
    

    def __init__(self, n = 1, k = 0, name="DEFAULT"):
        super().__init__(name=name)
        self.n = n
        self.k = k

        self.refractive_index = n + 1j*k


    def __str__(self) -> str:

        return "Default Material: n = " + str(self.n) + " , k = " + str(self.k)

    def __repr__(self) -> str:

        return self.__str__()



    def calc_phase_shift(self,
                thickness:torch.Tensor or np.ndarray,
                wavelengths:torch.Tensor or np.ndarray) -> torch.Tensor or np.ndarray:

        refractive_indices = self.get_complex_index_of_refraction(wavelengths=wavelengths)

        phi = CGH_Material.calc_phase_shift_equation(refractive_indices=refractive_indices,wavelengths=wavelengths,thickness=thickness)
        
        return phi
        

    def get_complex_index_of_refraction(self, wavelengths:torch.Tensor or np.ndarray) -> torch.Tensor or np.ndarray:
        
        if type(wavelengths) is torch.Tensor:
            refractive_index = torch.ones(wavelengths.shape, device = wavelengths.device) * self.refractive_index
        else:
            refractive_index = np.ones(wavelengths.shape, device = wavelengths.device) * self.refractive_index

        return refractive_index
