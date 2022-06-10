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
from holotorch.utils.Helper_Functions import ft2, ift2

class SLM_Upsampler(CGH_Component):
    
    def __init__(self,  
                        replicas        : int       = 3,
                        pixel_fill_ratio: float     = 1.0,
                        pixel_fill_ratio_opt: bool  = False
        ):
        """ 

        Apply the SLM sampling function as described in:

        M. Gopakumar, J. Kim, S. Choi, Y. Peng, and G. Wetzstein, Unfiltered holography: optimizing high diffraction orders without optical filtering for compact holographic displays, Opt. Lett. 46, 5822-5825 (2021)

        Args:
            slm_dimension (TensorDimension.HW): tensor dimensions H x W
            replicas (int): the number of sampling replicas to create in the frequency domain
            pixel_fill_ratio (float) the ratio of active pixel area to total pixel area 
        """

        super().__init__()

        # Add the attribute properties for all parameters
        # NOTE: This is complicated code in the background, but all this is doing is
        # to ensure that torch variables are moved from Buffer <-> Parameter
        # Whenever the "variablename_opt" flag is changed (via its setter method)
        self.add_attribute( attr_name="pixel_fill_ratio")

        self.replicas               = replicas
        self.pixel_fill_ratio_opt   = pixel_fill_ratio_opt
        self.pixel_fill_ratio       = pixel_fill_ratio

        # initialize the shape to a 6D size tensor
        self.shape = torch.Size([0,0,0,0,0,0])

    def create_frequency_grid(self, H, W):
        # precompute frequency grid for pixel transfer function (PTF)

        R   = self.replicas
        df  = torch.sqrt(self.pixel_fill_ratio) # linear_pixel_fill

        with torch.no_grad():
            # Creates the frequency coordinate grid in x and y direction
            kx          = R*torch.linspace(-1/2, 1/2, H*R)
            ky          = R*torch.linspace(-1/2, 1/2, W*R)        
            Kx, Ky      = torch.meshgrid(kx, ky)
            self.PTF    = torch.sinc(Kx*df) * torch.sinc(Ky*df)

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
    def PTF(self):
        return self._PTF
    
    @PTF.setter
    def PTF(self, PTF):
        self.register_buffer("_PTF", PTF)

    def forward(self,
            field : ElectricField,
            ) -> ElectricField:
        """ Takes an input field and passes it through the SLM Upsample model

        Upsample Algorithm Outline:
            a. Compute angular spectrum of slm
            b. Zero pad and create replicas of slm (from sampling)
            c. Add DC terms for each replica
            d. Apply pixel transfer function
            e. Convert back to spatial domain (same size as before (a), but now with larger resolution)         

        Args:
            field (ElectricField): Complex field 6D tensor object 

        Returns:
            ElectricField: The electric field after the hologram model
        """
        """
        
        
        
        Parameters
        ==========
        field           : torch.complex128
                           - batch x time x color x height x width.

        Output
        ==========
        Eout            : torch.complex128
                           Complex field 6D tensor with dims - batch x time x color x height x width.


        """

        # store the shape - the setter automatically creates a new grid if dimensions changed
        self.shape = field.shape

        # extract the data tensor from the field
        field_data  = field.data

        # add diffraction efficiency and DC/unmodulated term
        # NOTE: pixel_fill_ratio        = magnitude of diffracted field
        # NOTE: (1-pixel_fill_ratio)    = magnitude of DC/unmodulated 
        field_energy    = field_data.abs().pow(2).mean()
        DC_amp          = (1-self.pixel_fill_ratio) * torch.sqrt(field_energy)
        field_data      = self.pixel_fill_ratio * field_data + DC_amp 

        # Noop of there is only one replica
        if self.replicas > 1:

            # convert field to 4D tensor for batch processing
            B,T,P,C,H,W = field_data.shape
            field_data = field_data.view(B*T*P,C,H,W)

            # convert to angular spectrum
            field_data = ft2(field_data)        

            # create replicas in the frequency domain
            R = self.replicas
            field_data = field_data.view(B*T*P,C,1,H,1,W)       # B*T*P x C x H x W
            field_data = field_data.expand(B*T*P,C,R,H,R,W)     # B*T*P x C x H x R x W x R
            field_data = field_data.reshape(B*T*P,C,H*R,W*R)    # B*T*P x C x H*R x W*R

            # apply pixel transfer function (PTF)
            field_data = field_data * self.PTF 

            # convert back to spatial domain  
            field_data = ift2(field_data) # B*T*P x C x H x W

            # convert field back to 6D tensor
            field_data = field_data.view(B,T,P,C,H*R,W*R)

        Eout = ElectricField(
                data=field_data,
                wavelengths=field.wavelengths,
                spacing=field.spacing
                )

        return Eout       



