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
import numpy as np

from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.CGH_Datatypes.IntensityField import IntensityField
from holotorch.CGH_Datatypes import Light
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.utils.Helper_Functions import ft2, ift2
from holotorch.utils.Enumerators import *

class DPAC(CGH_Component):
    
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,
                field : ElectricField,
                bit_depth : int =8,
                max_phase : float = 2*np.pi,
                ) -> torch.Tensor:
    
        if isinstance(field,ElectricField) or  isinstance(field, Light):
            field = field.data

        out = DPAC.double_phase(field, three_pi=True, mean_adjust=True)

        return out

    def compute_dpac_phase(        self,
                                target_field : IntensityField or ElectricField,
                                # NOTE: these are magic numbers for now!!!
                                scale       = 1.0,
                                off         = 0*np.pi,
                                max_phase   = 2*np.pi,):

        voltage_modulation = self.compute_dpac_voltage_map_from_target(
            target_field=target_field,
            scale=scale,
            off=off,
            max_phase=max_phase
        )
        
        # Voltage modulation is between 0 and 1 (full SLM-bitdepth). We need to scale back to the 2*pi range
        phase_modulation = voltage_modulation * max_phase

        return phase_modulation

    def compute_dpac_voltage_map_from_target(
                                self,
                                target_field : IntensityField or ElectricField,
                                # NOTE: these are magic numbers for now!!!
                                scale       = 1.0,
                                off         = 0*np.pi,
                                max_phase   = 2*np.pi,
                            ):
        """ Computes a Double-Phase Amplitude Encoding

        Args:
            target_field (IntensityField): _description_

        Returns:
            _type_: _description_
        """
        
        # Dpac generator returns a tensor for now
        phase_only_modulation   = self.forward(target_field)

        phase_only_modulation   = scale * phase_only_modulation + off
        voltage_modulation      = phase_only_modulation / max_phase
        # We need to wrap voltage modulation into [0,1] range
        voltage_modulation      = voltage_modulation % 1.0

        if voltage_modulation.ndim == 3:
            voltage_modulation = voltage_modulation[:,None,None,None]
        # remove pupil dimension
        voltage_modulation = voltage_modulation[:,:,0,:,:,:]

        return voltage_modulation

    def bandlimit_field(self, field: ElectricField):
        field_data = field.data

        B,T,P,C,H,W = field_data.shape
        field_data = field_data.view(B*T*P,C,H,W)
        new_field_data = torch.zeros_like(field_data)

        x       = torch.linspace(-1,1,W)
        y       = torch.linspace(-1,1,H)
        Y,X     = torch.meshgrid(y,x)
        R2      = X**2 + Y**2
        rad     = 1/4
        mask    = R2 < rad**2  

        # convert to frequency domain
        Field_data = ft2(field_data)

        # set border of zeros in frequency domain
        new_field_data = Field_data * mask[None,None]

        # convert back to spatial domain
        field_data = ift2(new_field_data).abs()

        field_data = field_data.view(B,T,P,C,H,W)
        field.data = field_data
        return field

    @staticmethod
    def double_phase(
                     field,
                     three_pi=True,
                     mean_adjust=True
                     ):
        """Converts a complex field to double phase coding
        field: A complex64 tensor with dims [..., height, width]
        three_pi, mean_adjust: see double_phase_amp_phase
        """
        
        if torch.is_complex(field) == False:
            field = field + 1j - 1j
        
        dpac = DPAC.double_phase_amp_phase(
                field.abs(),
                field.angle(),
                three_pi,
                mean_adjust
                )
        return dpac

    @staticmethod
    def double_phase_amp_phase(
                amplitudes : torch.Tensor,
                phases : torch.Tensor,
                three_pi=True,
                mean_adjust=False,
                encoding_type : ENUM_DPAC_ENCODING = ENUM_DPAC_ENCODING.CHECKERBOARD
                ):
        """converts amplitude and phase to double phase coding
        amplitudes:  per-pixel amplitudes of the complex field
        phases:  per-pixel phases of the complex field
        three_pi:  if True, outputs values in a 3pi range, instead of 2pi
        mean_adjust:  if True, centers the phases in the range of interest
        """
        # normalize
        if amplitudes.max() > 1:
            amplitudes = amplitudes / amplitudes.max()

        phases_a = phases - torch.acos(amplitudes)
        phases_b = phases + torch.acos(amplitudes)

        phases_out = phases_a

        if encoding_type is ENUM_DPAC_ENCODING.HORIZONTAL:
            # Horizontal row encoding
            phases_out[..., ::2, :] = phases_b[..., ::2, :]
        elif encoding_type is ENUM_DPAC_ENCODING.VERTICAL:
            # Vertical row encoding
            phases_out[..., :, ::2] = phases_b[..., :, ::2]
        elif encoding_type is ENUM_DPAC_ENCODING.CHECKERBOARD:
            # Checkerboard encoding
            phases_out[..., ::2, 1::2] = phases_b[..., ::2, 1::2]
            phases_out[..., 1::2, ::2] = phases_b[..., 1::2, ::2]

        if three_pi:
            max_phase = 3 * np.pi
        else:
            max_phase = 2 * np.pi

        if mean_adjust:
            phases_out -= phases_out.mean()

        return (phases_out + max_phase / 2) % max_phase - max_phase / 2
