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

import numpy as np
import torch
import torchvision
from torch.nn.functional import pad
import matplotlib.pyplot as plt

from holotorch.Optical_Components.CGH_Component import CGH_Component
import holotorch.utils.Dimensions as Dimension
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.utils.Helper_Functions import ft2, ift2
from holotorch.utils.Enumerators import *

class ASM_Prop(CGH_Component):
    
    def __init__(self, 
                    init_distance             : float = 0.0,
                    z_opt                     : bool = False,
                    linear_conv               : bool = True,
                    pad_size                  : torch.Tensor = None,
                    pad_factor                : float = None,
                    prop_kernel_type          : ENUM_PROP_KERNEL_TYPE = ENUM_PROP_KERNEL_TYPE.FULL_KERNEL
                ):
        """
        Angular Spectrum method with bandlimited ASM from Digital Holographic Microscopy
        Principles, Techniques, and Applications by K. Kim 
        Eq. 4.22 (page 50)

        Args:
            init_distance (float, optional): initial propagation distance. Defaults to 0.0.
            z_opt (bool, optional): is the distance parameter optimizable or not. Defaults to False
            linear_conv (bool, optional): perform linear convolution (cheap) or zero padding (expensive). Defaults to True.
            paraxial_kernel (bool, optional): use paraxial approximation for kernel. Defaults to False.
        """                
        super().__init__()

        self.add_attribute( attr_name="z")

        if not torch.is_tensor(pad_size):
            if pad_size == None:
                pad_size = torch.tensor([0,0])
            else:
                pad_size = torch.tensor(pad_size)

        # store the input params
        self.linear_conv        = linear_conv
        self.prop_kernel_type   = prop_kernel_type   
        self.z_opt              = z_opt
        self.z                  = init_distance
        self.pad_size           = pad_size
        self.pad_factor         = pad_factor

        # the normalized frequency grid
        # we don't actually know dimensions until forward is called
        self.Kx = None
        self.Ky = None

        # initialize the shape
        self.shape = torch.Size([1,1,1,1,64,64])

    def compute_pad_size(self, H, W):
        # Get the shape for processing
        if self.linear_conv == True:
            padding_x = 0
            padding_y = 0
        else:
            if self.pad_size == None:
                padding_x = 1    
                padding_y = 1    
            else:
                padding_x = 2 * self.pad_size[0] / W
                padding_y = 2 * self.pad_size[1] / H
            
        padW =int( (1 + padding_x) * W)
        padH =int( (1 + padding_y) * H)
        
        return padW, padH

    def create_frequency_grid(self, H, W):

        padW, padH = self.compute_pad_size(H,W)

        # precompute frequency grid for ASM defocus kernel
        with torch.no_grad():
            # Creates the frequency coordinate grid in x and y direction
            kx = torch.linspace(-1/2, 1/2, padH)
            ky = torch.linspace(-1/2, 1/2, padW)        
            self.Kx, self.Ky = torch.meshgrid(kx, ky)

    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self, shape):
        try:
            _,_,_,_,H_new,W_new = shape
            _,_,_,_,H_old,W_old = self.shape
            self._shape  = shape
            if H_old != H_new or W_old != W_new:
                self.create_frequency_grid(H_new,W_new)
        except AttributeError:
            self._shape  = shape

    @property
    def Kx(self):
        return self._Kx
    
    @Kx.setter
    def Kx(self, Kx):
        self.register_buffer("_Kx", Kx)

    @property
    def Ky(self):
        return self._Ky
    
    @Ky.setter
    def Ky(self, Ky):
        self.register_buffer("_Ky", Ky)


    def visualize_kernel(self,
            field : ElectricField,
        ):
        kernel = self.create_kernel(field = field)

        plt.subplot(121)
        plt.imshow(kernel.abs().cpu().squeeze(),vmin=0)
        plt.title("Amplitude")
        plt.subplot(122)
        plt.imshow(kernel.angle().cpu().squeeze())
        plt.title("Phase")
        plt.tight_layout()

    def create_kernel(self,
        field : ElectricField,
            ):

        # store the shape - the setter automatically creates a new grid if dimensions changed
        self.shape = field.shape

        # extract dx, dy spacing into T x C tensors
        spacing = field.spacing.data_tensor
        dx      = spacing[:,:,0]
        if spacing.shape[2] > 1:
            dy = spacing[:,:,1]
        else:
            dy = dx

        # extract the data tensor from the field
        wavelengths = field.wavelengths

        #################################################################
        # Prepare Dimensions to shape to be able to process 6D data
        # ---------------------------------------------------------------
        # NOTE: This just means we're broadcasting lower-dimensional
        # tensors to higher dimensional ones
        #################################################################

        # get the wavelengths data as a TxC tensor 
        new_shape       = wavelengths.tensor_dimension.get_new_shape(new_dim=Dimension.TC)        
        wavelengths_TC  = wavelengths.data_tensor.view(new_shape) # T x C
        # Expand wavelengths for H and W dimension
        wavelengths_TC  = wavelengths_TC[:,:,None,None]
            
        # do the same for the spacing
        dx_TC   = dx.expand(new_shape)
        dx_TC   = dx_TC[:,:,None,None] # Expand to H and W
        dy_TC   = dy.expand(new_shape)
        dy_TC   = dy_TC[:,:,None,None] # Expand to H and W


        #################################################################
        # Compute the Transfer Function Kernel
        #################################################################
        
        # Expand the frequency grid for T and C dimension

        self.Kx = self.Kx.to(device=field.data.device)
        self.Ky = self.Ky.to(device=field.data.device)


        Kx = 2*np.pi * self.Kx[None,None,:,:] / dx_TC
        Ky = 2*np.pi * self.Ky[None,None,:,:] / dy_TC

        # create the frequency grid for each T x C wavelength/spacing combo
        K2 = Kx**2 + Ky**2 

        # compute ASM kernel on the fly for the right wavelengths
        K_lambda = 2*np.pi /  wavelengths_TC # T x C x H x W
        K_lambda_2 = K_lambda**2  # T x C x H x W

        #
        # MORE on ASM Kernels here in this book: 
        # Digital Holographic Microscopy
        # Principles, Techniques, and Applications
        #

        if self.prop_kernel_type is ENUM_PROP_KERNEL_TYPE.PARAXIAL_KERNEL:
            # avoid sqrt operation
            ang = self.z * K_lambda[:,:,None,None] + self.z/(2*K_lambda)*K2 # T x C x H x W
        elif self.prop_kernel_type is ENUM_PROP_KERNEL_TYPE.FULL_KERNEL:
            ang = - self.z * torch.sqrt(K_lambda_2 - K2) # T x C x H x W
            if ang.is_complex():
                ang = ang.real

        #################################################################
        # Bandlimit the kernel
        # see band-limited ASM - Matsushima et al. (2009)
        # K. Matsushima and T. Shimobaba, 
        # "Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields,"
        #  Opt. Express  17, 19662-19673 (2009).
        #################################################################

        # size of the field
        # # Total field size on the hologram plane
        length_x = field.height * dx_TC 
        length_y = field.width  * dy_TC

        # band-limited ASM - Matsushima et al. (2009)
        f_y_max = 2*np.pi / torch.sqrt((2 * self.z * (1 / length_x) ) **2 + 1) / wavelengths_TC
        f_x_max = 2*np.pi / torch.sqrt((2 * self.z * (1 / length_y) ) **2 + 1) / wavelengths_TC


        H_filter = torch.zeros_like(ang)
        H_filter[ ( torch.abs(Kx) < f_x_max) & (torch.abs(Ky) < f_y_max) ] = 1

        ASM_Kernel =  H_filter * torch.exp(1j*H_filter * ang)

        return ASM_Kernel

    def forward(self,
            field : ElectricField,
            ) -> ElectricField:
        """ 
        Takes in optical field and propagates it to the instantiated distance using ASM from KIM
        Eq. 4.22 (page 50)

        Args:
            field (ElectricField): Complex field 6D tensor object 

        Returns:
            ElectricField: The electric field after the rotate field propagation model
        """
        # extract the data tensor from the field
        wavelengths = field.wavelengths
        field_data  = field.data

        ASM_Kernel  = self.create_kernel(field)

        #################################################################
        # Apply the convolution in Angular Spectrum
        #################################################################
        
        # convert field to 4D tensor for batch processing
        B,T,P,C,H,W = field_data.shape
        field_data = field_data.view(B*T*P,C,H,W)
        
        pad_type = ENUM_LOSS_TYPE.MAX_VALUE_PAD
        
        pad_val = float(field_data.abs().max().cpu().detach().numpy())
        
        if pad_type is ENUM_LOSS_TYPE.ZERO_VALUE_PAD:
            pad_val = 0
                    
        # If not linear convolution, pad the image to avoid wrap around effects
        if not self.linear_conv:
            padW, padH = self.compute_pad_size(H,W)
            pad_x = int((padH - H)/2)
            pad_y = int((padW - W)/2)
            field_data = pad(field_data, (pad_x,pad_x,pad_y, pad_y), mode='constant', value=pad_val)

        _, _, H_pad,W_pad = field_data.shape

        # convert to angular spectrum
        field_data = ft2(field_data)
        
        # Convert 4D into 6D so that 6D-ASM Kernel can be applied
        field_data = field_data.view(B,T,P,C,H_pad,W_pad)
        # apply ASM kernel
        field_data = field_data * ASM_Kernel[None,:,None,:,:,:]
        # Convert 6D into 4D so that FFT2 can be applied to 4D tensor
        field_data = field_data.view(B*T*P,C,H_pad,W_pad) # B*T*P x C x H x W

        # convert back to spatial domain  
        field_data = ift2(field_data) # B*T*P x C x H x W

        # If not linear convolution, unpad the image to avoid wrap around effects
        if not self.linear_conv:
            center_crop = torchvision.transforms.CenterCrop([H,W])
            field_data = center_crop(field_data)

        # convert field back to 6D tensor
        field_data = field_data.view(B,T,P,C,H,W)
        
        field.spacing.set_spacing_center_wavelengths(field.spacing.data_tensor)

        Eout = ElectricField(
                data=field_data,
                wavelengths=wavelengths,
                spacing=field.spacing
                )

        return Eout