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
from holotorch.CGH_Datasets.CGH_Dataset import CGH_Dataset

import numpy as np

class BlazedGratingDataset(CGH_Dataset):
    
    def __init__(self,
            num_pixel_x : int =1080,
            num_pixel_y : int = 1920,
            freq_low  : torch.Tensor = 2,
            freq_high : torch.Tensor = None,
            freq_step : torch.Tensor = 1,
            angle_rad : torch.Tensor = 0,
            flag_blazed_or_sin : bool = True
                ):
        
        
        self.flag_blazed_or_sin = flag_blazed_or_sin
        
        x = torch.arange(0,num_pixel_x)*1.0 / num_pixel_x
        y = torch.arange(0,num_pixel_y)*1.0 / num_pixel_y
        self.X, self.Y = torch.meshgrid(x,y)


        if torch.is_tensor(angle_rad) == False:
            angle_rad = torch.tensor(angle_rad)
            _tmp = torch.ones(1)
            angle_rad = angle_rad.to(_tmp.device)
        self.angle_rad = angle_rad
        
        if freq_high is None:
            freq_high = freq_low + 1
        
        self.freqs = torch.arange(freq_low,freq_high,freq_step) * 1.00
        
        self.data_sz = len(angle_rad) * len(self.freqs)

    def __getitem__(self, idx):
        """
        
        """
        num_phase_scales = len(self.freqs)
        #num_amplitudes   = len(self.amplitude_scale)
        
        idx_count = idx % num_phase_scales
        idx_speckle_size = idx // num_phase_scales
        
        freq = self.freqs[idx_count]
        angle_rad = self.angle_rad[idx_speckle_size]
               
        blazed_gratings = BlazedGratingDataset.compute_blazed_gratings(
                X                  = self.X,
                Y                  = self.Y,
                angle_rad          = angle_rad,
                freq               = freq,
                flag_blazed_or_sin = self.flag_blazed_or_sin,
                )    
        return blazed_gratings
                
    def __len__(self):
        return self.data_sz 
    
    @staticmethod
    def compute_blazed_gratings(
                X           : torch.Tensor,
                Y           : torch.Tensor,
                angle_rad   : float,
                freq        : float,
                flag_blazed_or_sin = False,
        ):
        
        resX, resY = X.shape
        
        xi0 = resX / freq * torch.sin(angle_rad)
        nu0 = resY / freq * torch.cos(angle_rad)

        Z = xi0 * X + nu0 * Y

        if flag_blazed_or_sin:
            Z = torch.remainder(Z,1.0)
        else:
            Z = 0.5 * (torch.sin(2*np.pi*Z) + 1 )

        return Z

