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
import pathlib
import numpy as np
import torch
from scipy.optimize import curve_fit
sys.path.append('../')

from holotorch.Material.CGH_Material import CGH_Material
from holotorch.Material.CGH_Material import CGH_Material
from holotorch.utils.units import *

class CauchyMaterial(CGH_Material):
    """Represents a Material class
    """    

    # The model parameters
    A = None # A-parameter in Cauchy Model
    B = None # B-parameter in Cauchy Model
    


    def __init__(self, A, B, name=None):
        """[summary]

        Args:
            A ([type]): [description]
            B ([type]): [description]
            name ([type], optional): [description]. Defaults to None.
        """
        super().__init__(name=name)
        self.A = A
        self.B = B


    def __str__(self):

        name = ""
        if self.name is not None:
            name = "(" + self.name + " )"
        return "CauchyMaterial" + name +  ": A = " + str(self.A) + " , B = " + str(self.B)


    def __repr__(self):
        return "CauchyMaterial: A = " + str(self.A) + " , B = " + str(self.B)

    @staticmethod
    def cauchy_equation(wavelength:torch.Tensor or np.ndarray or np.ScalarType,
                        A,
                        B):
        """Computes Cauchy Equation

        Args:
            wavelength ([torch.Tensor or np.ndarray or np.ScalarType]): Wavelengths
            A ([Scalar]): A value in Cauchy Equation
            B ([Scalar]): B value in Cauchy Equation
        Returns:
            [type(wavelength)]: The refractive index corresponding to the wavelength
        """    
        wavelength = wavelength / um

        ref_index = A + B/(wavelength**2)
        return ref_index


    @staticmethod
    def fit_material(wavelengths, n, k = None):
        """Fits the Cauchy Parameters to a wavelength and refractive index 

        Args:
            wavelengths ([type]): [description]
            refractive_index ([type]): [description]

        Returns:
            [type]: [description]
        """    
        popt,_ = curve_fit(CauchyMaterial.cauchy_equation, wavelengths, n)
        
        A_n = popt[0]
        B_n = popt[1]

        A_k = 0
        B_k = 0

        A = 1j*A_k + A_n
        B = 1j*B_k + B_n

        return A, B

    @staticmethod
    def create_from_txt(path:str or pathlib.Path, wavelength_left = 400*nm, wavelength_right = 800*nm):
        """Creates a static Cauchy Material from a txt file

        Args:
            path (strorpathlib.Path): [description]
            wavelength_left ([type], optional): [description]. Defaults to 400*nm.
            wavelength_right ([type], optional): [description]. Defaults to 800*nm.

        Returns:
            [type]: [description]
        """        
        data = np.loadtxt(path, skiprows=1)

        wavelengths = data[:,0] * nm
        n = data[:,1]
        k = data[:,2]

        k = k[wavelengths > wavelength_left]
        n = n[wavelengths > wavelength_left ]
        wavelengths = wavelengths[wavelengths > wavelength_left ]


        k = k[wavelengths < wavelength_right]
        n = n[wavelengths < wavelength_right ]
        wavelengths = wavelengths[wavelengths < wavelength_right ]


        return CauchyMaterial.create_from_values(wavelengths=wavelengths,n=n, k=k, wavelength_left=wavelengths, wavelength_right=wavelength_right)

    @staticmethod
    def holografix(): 
        """ A decorator for the holografix material. """
        holografix_wavelengths = np.array([450,517,660])*nm
        holografix_refractive_idx = np.array([1.5223 , 1.5159 , 1.5081])

        return CauchyMaterial.create_from_values(
            wavelengths=holografix_wavelengths,
            n=holografix_refractive_idx,
            name='holografix')

    @staticmethod
    def create_from_values(wavelengths, n, k = None, name = None, wavelength_left = 400*nm, wavelength_right = 800*nm): 
        """ A static method to create a material object.
        """

        A,B = CauchyMaterial.fit_material(wavelengths=wavelengths, n = n, k = k)
        material = CauchyMaterial(A=A,B=B,name=name)
        return material

    # Implements the abstractmethod from CGH_Material
    def calc_phase_shift(self,
                thickness:torch.Tensor or np.ndarray,
                wavelengths:torch.Tensor or np.ndarray):
        """[summary]

        Args:
            thickness (torch.Tensorornp.ndarray): [description]
            wavelengths (torch.Tensorornp.ndarray): [description]
        """

        assert isinstance(thickness,type(wavelengths)), "Thickness and Wavelengths need to same type."


        refractive_indices = self.get_complex_index_of_refraction(wavelengths=wavelengths)

        phi = CGH_Material.calc_phase_shift_equation(refractive_indices=refractive_indices,
                            thickness=thickness,
                            wavelengths=wavelengths)
                        
        return phi

    # Implements the abstractmethod from CGH_Material
    def get_complex_index_of_refraction(self, wavelengths:torch.Tensor or np.ndarray):
        """[summary]

        Args:
            wavelengths (torch.Tensorornp.ndarray): [description]
        """        

        complex_index_of_refraction = CauchyMaterial.cauchy_equation(wavelength=wavelengths, A = self.A, B = self.B)

        return complex_index_of_refraction


