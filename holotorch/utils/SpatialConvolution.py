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
import kornia


from holotorch.CGH_Datatypes.Light import Light

from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.utils.transformer_6d_4d import transform_4D_to_6D, transform_6D_to_4D
import holotorch.utils.Visualization_Helper as VH

import matplotlib.pyplot as plt

class SpatialConvolution(CGH_Component):
    
    def __init__(self,
            kernel : torch.Tensor = None
                 ):
        """

        """
        super().__init__()

        self.add_attribute("kernel")
        self.kernel_opt = True
        self.kernel = kernel

    @classmethod
    def gaussian_kernel(cls,
            kernel_size : int = 3,
            sigma   : float = 1.0,
            sigma_x : float = None,
            sigma_y : float = None
    ) -> SpatialConvolution:
    
        kernel = SpatialConvolution.compute_gaussian_kernel(
            kernel_size = kernel_size,
            sigma       = sigma,
            sigma_x     = sigma_x,
            sigma_y     = sigma_y
        )
        return SpatialConvolution(kernel=kernel)


    @staticmethod
    def compute_gaussian_kernel(
        kernel_size : int = 3,
        sigma   : float = 1.0,
        sigma_x : float = None,
        sigma_y : float = None
    ) -> torch.Tensor:
        if sigma_x is None:
            sigma_x = sigma

        if sigma_y is None:
            sigma_y = sigma_x

        kernel = kornia.filters.get_gaussian_kernel2d((kernel_size, kernel_size), (sigma_x, sigma_y))

        kernel = kernel/kernel.sum()

        return kernel

    def forward(self,
            field : Light
            ) -> Light:

        kernel = self.kernel[None] # Need to expend for kornia
    
        values = field.data
        old_shape = values.shape
        
        values = transform_6D_to_4D(values)
        
        out = kornia.filters.filter2d(
            input = values,
            kernel = kernel,
            border_type='reflect',
            normalized=False,
            padding='same'
            )

        values = transform_4D_to_6D(tensor_in=out, newshape=old_shape)

        new_field = Light(
            data = values,
            wavelengths = field.wavelengths,
            spacing = field.spacing
        )

        return new_field

    def visualize_kernel(self, title : str = None):
        _im = plt.imshow(self.kernel.detach().cpu())
        if title is not None:
            plt.title(title)
        VH.add_colorbar(_im)
