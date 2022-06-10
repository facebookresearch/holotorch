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
from holotorch.Optical_Components.CGH_Component import CGH_Component



class GaussianMixture(CGH_Component):
    
    def __init__(self,
                num_gaussians = 3,
                n_channels = 1 
                 ) -> None:
        super().__init__()

        self.add_attribute(attr_name="weights")
        self.add_attribute(attr_name="mu")
        self.add_attribute(attr_name="sigma")
        self.add_attribute(attr_name="bias")

        self.set_opt(True)
        # self.bias_opt = False
        # self.weights_opt = False
        # self.mu_opt = False
        # self.sigma_opt = False
        
        self.bias    = torch.ones(n_channels)
        
        self.weights = torch.zeros(n_channels, num_gaussians) / num_gaussians
        self.mu      = 0.01*torch.randn(2, n_channels, num_gaussians)
        self.sigma   = 4*torch.ones(2, n_channels, num_gaussians)
             

    def set_opt(self, opt : bool):
        self.bias_opt       = opt
        self.weights_opt    = opt
        self.mu_opt         = opt
        self.sigma_opt      = opt

    def compute_gaussian_mixture(self, XY_grid, weights, mu, sigma) -> torch.Tensor:

        prefac = torch.sqrt ( ( 2 * np.pi) ** 2 * torch.abs(sigma.sum(dim=0)) )
        
        inside = (XY_grid[:, None, None,:,:] - mu[:,:,:,None,None])**2. / (2 * sigma[:,:,:,None,None] )
        exponent = -1 * torch.sum(inside, dim=0) 
        gaussian = torch.exp(exponent) / prefac[:,:,None,None]
        
        gaussian = gaussian * weights[:,:,None,None]
        gaussian = gaussian.sum(dim = 1)
        
        gaussian = gaussian + self.bias
        
        return gaussian
        
    def forward(self, field_shape : torch.Tensor):
        
        device = self.mu.device
        
        Nx = field_shape[0]
        Ny = field_shape[1]

        aspect_ratio = Nx/Ny
        y_size = 1/aspect_ratio
        x = torch.linspace(-1,1,Nx, device=device)
        y = torch.linspace(-y_size,y_size,Ny, device=device)
        
        X, Y = torch.meshgrid(x,y)
        
        XY = torch.stack((X,Y))
        
        gaussian = self.compute_gaussian_mixture(XY_grid = XY, weights = self.weights, mu = self.mu, sigma = self.sigma)
        
        return gaussian
    
    def __str__(self) -> str:
        mystr = "Bias : " + str(self.bias.detach().cpu().numpy())
        mystr += "weights : " + str(self.weights.detach().cpu().numpy())
        mystr += "mu : " + str(self.mu.detach().cpu().numpy())
        mystr += "sigma : " + str(self.sigma.detach().cpu().numpy())

        return mystr
        
        