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
from holotorch.CGH_Datatypes.Light import Light
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.CGH_Datatypes.IntensityField import IntensityField 
from holotorch.utils import transformer_6d_4d
import holotorch.utils.Homography as Homography 

class Homography_Module(CGH_Component):
    
    def __init__(self,
        output_shape : torch.Size,
        homography : torch.Tensor = None,
        make_copy : bool = True,
        homography_opt : bool =  False
                 ) -> None:
        super(
            ).__init__()
        
        self.add_attribute( attr_name="homography")

        self.homography_opt = homography_opt
        
        if homography is None:
            homography = torch.eye(3,dtype = torch.float64)
            self.homography = homography
            self.reset_parameters()

        if make_copy:
            homography = homography.clone()
            
        self.output_shape = output_shape
            
        self.homography = homography
        self.output_shape = output_shape
    
    def __str__(self) -> str:
        mystr = str(self.homography.detach().cpu().numpy())
        return mystr
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def reset_parameters(self):
        torch.nn.init.eye_(self.homography)
        
    def set_homography(self, homography : torch.Tensor):
        
        with torch.no_grad():
            self.homography.copy_(homography)
    
    def warp_image(self,
                   image : torch.Tensor,
                   target_shape = None,
                   ) -> torch.Tensor:
        
        if image.ndim == 2:
            image = image[None,None]
        
        if target_shape is None:
            target_shape = [*image.shape[0:2], *self.output_shape[-2:] ]

        homography = torch.unsqueeze(self.homography, dim=0)  # 1x3x3  

        img_warped = Homography.warp_image(
        image = image,
        target_shape = target_shape,
        homography = homography
        )
        return img_warped
    
    def forward(self, field : Light) -> IntensityField:
        
    
        orig_shape = [*field.shape[0:4], *self.output_shape[-2:] ]
        img = transformer_6d_4d.transform_6D_to_4D(field.data)
        
        img_warped = self.warp_image(
            image = img,
            target_shape = orig_shape,
        )             
            
        img_warped = transformer_6d_4d.transform_4D_to_6D(img_warped, orig_shape)
    
        out_field = IntensityField(img_warped, wavelengths=field.wavelengths, spacing=field.spacing)
    
        return out_field
