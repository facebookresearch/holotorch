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


from holotorch.utils.PyTorchPoly.poly import legendre, chebyshev, hermite

import torch
import torch.nn as nn
from holotorch.Optical_Components.CGH_Component import CGH_Component

class UnivariatePoly(nn.Module):
    """ Univariate Legendre Polynomial 
    
    NOTE: This class is adapted from https://github.com/goroda/PyTorchPoly
    """
    def __init__(self, PolyDegree, poly_type):
        super(UnivariatePoly, self).__init__()
        self.degree = PolyDegree
        self.linear = nn.Linear(PolyDegree+1, 1, bias=False)
        self.linear_y = nn.Linear(PolyDegree+1, 1, bias=False)
        
        weights = torch.zeros(PolyDegree + 1) + 1
        with torch.no_grad():
            self.linear.weight.copy_(weights)

        weights_y = torch.zeros(PolyDegree + 1) + 1
        with torch.no_grad():
            self.linear_y.weight.copy_(weights_y)
            
        self.poly_type = poly_type

    def forward(self, x, y):

        if self.poly_type == "legendre":
            vand = legendre(x, self.degree)
        elif self.poly_type == "chebyshev":
            vand = chebyshev(x, self.degree)
        elif self.poly_type == "hermite":
            vand = hermite(x, self.degree)            
        else:
            print("No Polynomial type ", self.poly_type, " is implemented")
            exit(1)
            
        # print("vand = ", vand)
        retvar = self.linear(vand)

        return retvar


class LegendrePoly2D(CGH_Component):
    """ Univariate Legendre Polynomial """
    def __init__(self,
                PolyDegree,
                flag_opt       : torch.tensor = torch.tensor([True]),
                ):
        super().__init__()
        
        self.degree = PolyDegree

        self.num_bases = (self.degree + 1) ** 2

        self.add_attribute( attr_name="legendre_coef")

        self.legendre_coef_opt = True
        # Set weights to 0 for beginning
        self.legendre_coef = torch.zeros(self.num_bases)
        
            
    def set_random_weights(self):

        coef = torch.randn(self.num_bases)
        with torch.no_grad():
            self.legendre_coef.copy_(coef)
            
    def set_zero_weights(self):

        with torch.no_grad():
            weights = torch.zeros(self.num_bases)
            weights = weights.type_as(self.legendre_coef)
            self.legendre_coef.copy_(weights)
            
    def set_weights(self, weights : torch.Tensor):

        with torch.no_grad():
            weights = weights.type_as(self.legendre_coef)
            self.legendre_coef.copy_(weights)


    def forward(self, x, y):

        Lxf = legendre(x, self.degree, device = self.legendre_coef.device)
        Lyf = legendre(y, self.degree, device = self.legendre_coef.device)
        
        Nx = len(x)
        Ny = len(y)
        
        # Apply outer product
        Legendre_Basis = Lxf[:,None,:,None] * Lyf[None,:,None,:]
        # View as 3D vector ( NOTE: Dimensions are H x W x #Bases)
        Legendre_Basis = Legendre_Basis.view(Nx, Ny, self.num_bases)
        
        out = self.legendre_coef[None,None,:] * Legendre_Basis
        out = torch.sum(out, axis = 2)

        return out    
