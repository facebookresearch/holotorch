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

from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Components.FT_Lens import FT_Lens
from holotorch.Optical_Components.Radial_Optical_Aperture import Radial_Optical_Aperture
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.utils.Dimensions import BTPCHW
from holotorch.utils.Enumerators import *
from holotorch.Optical_Components.SimpleMask import SimpleMask

class Four_F_system(CGH_Component):
    
    def __init__(self,
                 focallength_1 : float,
                 focallength_2 : float,
                 aperture_radius : float,
                 aperture_type : ENUM_SOURCE_APERTURE_TYPE = ENUM_SOURCE_APERTURE_TYPE.DISK,
                 flag_is_learnable : bool =  True,
                 flag_flip : bool = False,
                 ) -> None:
        super().__init__()
        
       
        self.lens1 : FT_Lens = FT_Lens(focal_length=focallength_1)
        self.lens2 : FT_Lens = FT_Lens(focal_length=focallength_2)

        self.init_aperture_radius : float = aperture_radius

        if flag_is_learnable == True:
            self.aperture : SimpleMask = SimpleMask()
        elif aperture_type is ENUM_SOURCE_APERTURE_TYPE.DISK:
            self.aperture : SimpleMask = Radial_Optical_Aperture(aperture_radius=aperture_radius)
        else:
            raise NotImplementedError("Not yet supported.")


        self.aperture_type : ENUM_SOURCE_APERTURE_TYPE = aperture_type

        # just use a dummy size for initialization
        self.shape      = torch.Size([1,1,1,1,64,64])
        self.spacing_in_FT_plane    = None #SpacingContainer(spacing=torch.tensor((1,1)))
        self.spacing_in_FT_plane    = None
        self.wavelength = None        
        self.flag_flip = flag_flip
        
    def init_abberrations(self, H_new, W_new, N_channel):
        flat_field              = torch.ones(N_channel,H_new, W_new) + 0j   
        
        field                   = ElectricField(
            data        = flat_field[None,:,None,None,:,:],
            spacing     = self.spacing_in_FT_plane,
            wavelengths = self.wavelength            
        )

        self.aperture.mask = self.make_gaussian_mask(field).data

        self.aperture.tensor_dimension = BTPCHW.from_shape(self.aperture.mask.shape)

        self.aperture.spacing = self.spacing_in_FT_plane

    def make_gaussian_mask(self, field : ElectricField):

        spacing : SpacingContainer = field.spacing
        
        x = torch.linspace(-0.5,0.5, field.height, device = field.data.device)
        y = torch.linspace(-0.5,0.5, field.width, device = field.data.device)
        
        X,Y = torch.meshgrid(x,y)
        X = field.height * spacing.data_tensor[:,:,0][:,:,None,None].to(field.data.device) * X[None,None]
        Y = field.width * spacing.data_tensor[:,:,1][:,:,None,None].to(field.data.device) * Y[None,None]
        R2 = X**2 + Y**2
        
        sig = self.init_aperture_radius
        
        if self.aperture_type is ENUM_SOURCE_APERTURE_TYPE.DISK:
            mask = R2 < (sig ** 2)
            mask = mask.float()
        elif self.aperture_type is ENUM_SOURCE_APERTURE_TYPE.GAUSSIAN:
            mask = torch.exp(-R2/sig**2/2)

        new_field = field.data * mask[None,:,None,:,:,:]
        
      
        field = ElectricField(
            data = new_field,
            spacing=field.spacing,
            wavelengths=field.wavelengths            
        )

        return field

    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self, shape):
        try:
            _,N_channel,_,_,H_new,W_new = shape
            _,N_channel,_,_,H_old,W_old = self.shape
            self._shape  = shape
            if H_old != H_new or W_old != W_new:
                self.init_abberrations(H_new,W_new,N_channel)
        except AttributeError:
            self._shape  = shape
        
    def forward(self, field : ElectricField) -> ElectricField:
        """Computes a 4F transform. If lens1 and lens2 is different
        the coordinates will change

        Args:
            field (ElectricField): [description]

        Returns:
            ElectricField: [description]
        """

        # Apply the first fourier transform
        field = self.lens1(field)

        # store the shape - the setter automatically creates a new aberration tensor if dimensions changed
        self.spacing_in_FT_plane    = field.spacing
        self.wavelength = field.wavelengths
        self.shape      = field.shape     
        
        field = self.aperture(field)

        # Apply the second fourier transform
        field = self.lens2(field)
        
        if self.flag_flip:
            # flip up/down & left/right for 4f sytem
            field.data = torch.flip(field.data,dims=[-1,-2]) 
        
        return field
    
    def visualize_aperture(self,
            figsize=(8,6),
            x0 = None,
            y0 = None,
            width = None,
            height = None,
            vmax = None,
            vmin = None,
            flag_colorbar : bool = False,
            adjust_aspect :bool = True
            ):
        
        self.aperture.visualize(
            figsize = figsize,
            x0 = x0,
            y0 = y0,
            width = width,
            height = height,
            vmax = vmax,
            vmin = vmin,
            flag_colorbar = flag_colorbar,
            adjust_aspect = True
        )