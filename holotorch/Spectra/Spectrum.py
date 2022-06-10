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
import torch.nn as nn
import numpy as np

from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.utils.torchinterp1d.torchinterp1d.interp1d import Interp1d


class Spectrum(nn.Module):

    def __init__(self,
                 wavelengths        : WavelengthContainer,
                 values             : torch.Tensor,
                 requires_grad          = False
                 ) -> None:

        super().__init__()

        values.requires_grad        = requires_grad
        
        assert torch.is_tensor(wavelengths), "wavelengths needs to be a tensor"
        assert torch.is_tensor(values), "values needs to be a tensor"
        
        self.wavelengths    = wavelengths
        self.values         = values

    @property
    def wavelengths(self) -> WavelengthContainer:
        return self._wavelengths
    
    @wavelengths.setter
    def wavelengths(self, wavelengths : WavelengthContainer):
        self.register_buffer("_wavelengths", wavelengths)
        
    @property
    def num_wavelengths(self):
        return self.wavelengths.shape[0]
    
        
    @property
    def num_sources(self) -> int:
        if self.values.ndim == 1:
            return 1
        else:
            return self.values.shape[1]
    

        
    @property
    def values(self) -> torch.Tensor:
        return self._values

    @values.setter
    def sigma_wavelengths(self, data : torch.Tensor) -> None:
        self.register_buffer("_values", data)
                
    def __getitem__(self, keys):
        pass
    

    
    @property
    def values(self) -> torch.Tensor:
        return self._values
    
    @values.setter
    def values(self, myvalues : torch.Tensor):
        self._values = myvalues
        
    def get(self,
        wavelengths : float or np.array or torch.tensor,
        source_idx : int or np.ndarray or torch.tensor
        ) -> torch.Tensor:
        """ Returns irradiance at specific wavelengths given by 

        Args:
            wavelengths (float or np.array or torch.tensor): [description]

        Returns:
            [type]: [description]
        """
        
        # Values should be a 2D tensor where the first dimension
        # denotes the temporal dimension which e.g. could be
        # different sources
        values = torch.zeros(wavelengths.shape).type_as(wavelengths)
        
        for idx_source in range(values.shape[0]):
            
            tmp_val = self.values[:,source_idx[idx_source]][None]
            tmp_wavelengths = wavelengths[idx_source][None]
            x = self.wavelengths[:,source_idx[idx_source]][None]
            
            y_new = Interp1d()(
                x    = x,
                y    = tmp_val,
                xnew = tmp_wavelengths
                    )
            values[idx_source] = y_new.squeeze()
        
        return values
        
    def to(self, device) -> Spectrum:
        """Returns a new Spectrum object with device

        Args:
            device ([type]): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """        
        
        if self.device is device:
            return self
        else:
            return Spectrum(
                wavelengths=self.wavelengths.to(device),
                values=self.values.to(device),
            )
    
    @staticmethod
    def make_norm_dist(x, mean, sd):
        return 1.0/(sd*np.sqrt(2*np.pi))*np.exp(-(x - mean)**2/(2*sd**2))
    
    def sample_spectrum(self):
        raise NotImplementedError