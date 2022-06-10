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
import kornia
import numpy as np
from holotorch.CGH_Datasets.CGH_Dataset import CGH_Dataset

from holotorch.Calibration_Procedures.CalibrationCaptureSessions.DataSetGenerators.Generator import compute_radial_map

class SpeckleDataset(CGH_Dataset):
    
    def __init__(self,
            num_pixel_x= 1080,
            num_pixel_y= 1920,
            speckle_min = 0.01,
            speckle_max = 1,
            gamma = 3,
            num_images_per_seckle_size = 10,
            n_speckl_sizes = 5,
            flag_histogram_equalized = False
            ) -> None:
        
        # self.speckle_range = speckle_min + speckle_max * torch.linspace(0,1,n_speckl_sizes) ** gamma
        self.speckle_range = gamma ** torch.linspace(speckle_min,speckle_max,n_speckl_sizes)
    
        self.flag_histogram_equalized = flag_histogram_equalized
        
        self.data_sz = num_images_per_seckle_size * n_speckl_sizes

        self.R = compute_radial_map(
            num_pixel_x = num_pixel_x,
            num_pixel_y = num_pixel_y
        )

        self.num_images_per_seckle_size = num_images_per_seckle_size
        self.n_speckl_sizes = n_speckl_sizes
        
        new_shape = [num_images_per_seckle_size,*self.R.shape]
        self.random_phase = torch.exp(1j * 2*np.pi * torch.rand(new_shape))


    def __getitem__(self, idx):
        """
        
        """
        idx_count = idx % self.num_images_per_seckle_size
        idx_speckle_size = idx // self.num_images_per_seckle_size
        
        speckle_radius = self.speckle_range[idx_speckle_size]
        
        speckle = self.compute_speckle(
            speckle_radius=speckle_radius,
            index_random = idx_count,
            flag_histogram_equalized= self.flag_histogram_equalized
        )
        return speckle
                
    def __len__(self):
        return self.data_sz 
    


    def compute_speckle(self,
            speckle_radius : float,
            index_random : str,
            flag_histogram_equalized = False,
                ):

        mask = torch.zeros_like(self.R)
        mask[self.R < speckle_radius] = 1

        img_random = self.random_phase[index_random]

        speckle = torch.fft.fftshift(torch.fft.fft2(img_random * mask)).abs()
        
        speckle = speckle / speckle.abs().max()
        
        if flag_histogram_equalized:
            speckle = kornia.enhance.equalize(speckle)
            
        return speckle

