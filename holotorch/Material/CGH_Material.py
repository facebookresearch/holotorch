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
import matplotlib.pyplot as plt
from holotorch.utils.units import *
import torch


from abc import ABC, abstractmethod


class CGH_Material(ABC):
    """Represents a Material class
    """    
    
    # Name of the material
    name = None


    def __init__(self, name=None):
        self.name = name

    @abstractmethod
    def calc_phase_shift(self,
                thickness:torch.Tensor or np.ndarray,
                wavelengths:torch.Tensor or np.ndarray):
        pass

    @abstractmethod
    def get_complex_index_of_refraction(self, wavelengths:torch.Tensor or np.ndarray):
        pass

    @staticmethod
    def calc_phase_shift_equation(refractive_indices, thickness, wavelengths) -> np.ndarray or torch.Tensor:

        assert refractive_indices.device == thickness.device, "REF INDEX = " + str(refractive_indices.device) + ", THICKNESS = " + str(thickness.device) +  " wavelengths = " + str(wavelengths.device) + ")"

        assert refractive_indices.device == wavelengths.device, "REF INDEX = " + str(refractive_indices.device) + ", THICKNESS = " + str(thickness.device) +  " wavelengths = " + str(wavelengths.device) + ")"

        if wavelengths.ndim == 0:
            wavelengths = torch.tensor([wavelengths])
            refractive_indices = torch.tensor([refractive_indices])

        if wavelengths.ndim == 1:
            phi = 2 * np.pi * refractive_indices[:,None,None] * thickness[None,:,:] / wavelengths[:,None,None]
        if wavelengths.ndim == 2:
            phi = 2 * np.pi * refractive_indices[:,:,None,None] * thickness[None,None,:,:] / wavelengths[:,:,None,None]            
        

        return phi


    def plot_refractive_index(self, xlim_left = 380*nm, xlim_right = 750*nm, figsize=(5,3),title=None):

        xrange = np.linspace(xlim_left,xlim_right,50)

        refactive_index = self.get_complex_index_of_refraction(wavelengths=xrange)

        if type(refactive_index) is torch.TensorType:
            n = refactive_index.real
            k = refactive_index.imag
        else:
            n = np.real(refactive_index.real)
            k = np.real(refactive_index.imag)

        fontsize = 16
        plt.figure(figsize=figsize)

        plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

        wavelengths_in_nm = xrange/nm
        plt.plot(wavelengths_in_nm,n,color='blue')
        plt.xlabel("Wavelength [nm]",fontsize=fontsize)
        plt.ylabel("Refractive n",color='blue',fontsize=fontsize)

        ax = plt.gca()

        ax.yaxis.label.set_color('blue')        #setting up X-axis label color to yellow
        ax.tick_params(axis='y', colors='blue')  #setting up Y-axis tick color to black

        # twin object for two different y-axis on the sample plot
        ax = plt.twinx()
        # make a plot with different y-axis using second axis object
        plt.plot(wavelengths_in_nm,k,color='red')
        plt.ylabel("Absorption k",color='red',fontsize=fontsize)
        ax.spines['right'].set_color('red')  
        ax.spines['left'].set_color('blue')  

        ax.xaxis.label.set_color('red')        #setting up X-axis label color to yellow
        ax.tick_params(axis='y', colors='red')  #setting up Y-axis tick color to black


        plt.xlim(xlim_left/nm,xlim_right/nm)


        if title is None:
            title = self.name
        plt.title(title,fontsize=fontsize)


        plt.tight_layout()