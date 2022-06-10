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

from holotorch.Calibration_Procedures.CalibrationCaptureSessions.DataSetGenerators.Generator import compute_radial_map
from holotorch.CGH_Datasets.CGH_Dataset import CGH_Dataset


class QuadraticPhaseDataset(CGH_Dataset):
    
    def __init__(self,
            num_pixel_x= 1080,
            num_pixel_y= 1920,
            scale = torch.arange(1,10),
            amplitude_scale = torch.ones(1),
            flag_blazed_or_not = True,
            ):
        
          

        self.R = compute_radial_map(
            num_pixel_x = num_pixel_x,
            num_pixel_y = num_pixel_y
        )
        
        self.scale = scale
        self.amplitude_scale = amplitude_scale
        
        self.data_sz = len(scale) * len(amplitude_scale)
        
        self.flag_blazed_or_not = flag_blazed_or_not



    def __getitem__(self, idx):
        """
        
        """
        num_phase_scales = len(self.scale)
        #num_amplitudes   = len(self.amplitude_scale)
        
        idx_count = idx % num_phase_scales
        idx_speckle_size = idx // num_phase_scales
        
        scale = self.scale[idx_count]
        amplitude_scale = self.amplitude_scale[idx_speckle_size]
        
        img = QuadraticPhaseDataset.compute_quadratic_phase_targets(
            R = self.R,
            scale = scale,
            amplitude_scale=amplitude_scale,
            flag_blazed_grating = self.flag_blazed_or_not
        )
        
        return img
                
    def __len__(self):
        return self.data_sz 
    


    @staticmethod
    def compute_quadratic_phase_targets(
        R : torch.Tensor,
        scale : float,
        amplitude_scale : float,
        flag_blazed_grating : bool = True,
    ):
        quadratic_phase = R **2
        quadratic_phase / quadratic_phase.max()
        
        img = quadratic_phase * scale
        
        if flag_blazed_grating:
            img = img.remainder(1)
        else:
            img = 0.5 *(torch.sin(img) + 1)

        img = img * amplitude_scale

        return img