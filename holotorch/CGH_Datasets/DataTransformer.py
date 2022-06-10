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
import kornia
import numpy as np
import torchvision

class DataTransformer(torch.nn.Module):

    def __init__(self,
        num_pixel_x,
        grayscale : bool = True,
        num_pixel_y  :int = None,
        border_x   : int = 0,
        border_y   : int = 0,
        center_x   : int = 0,
        center_y   : int = 0,
        sigma_gaussian_filter : float = 0,
                ) -> None:
        super().__init__()
        
        self.num_pixel_x = num_pixel_x
        self.grayscale = grayscale
        self.num_pixel_y = num_pixel_y
        self.border_x = border_x
        self.border_y = border_y
        self.sigma_gaussian_filter = sigma_gaussian_filter

        self.center_x = center_x
        self.center_y = center_y

        if num_pixel_y is None:
            num_pixel_y = num_pixel_x
        
    @staticmethod
    def normalize(input : torch.Tensor, method="mean"):
        """
        Args:
            method(str) : mean | max

        """
        if method == "mean":
            # compute the mean from only the valid portion
            normalization_factor = torch.mean(input.abs())
        elif method == "max":
            # compute the max from only the valid portion
            normalization_factor = torch.mean(input.abs())

        out = input / normalization_factor

        return out
    
    @staticmethod
    def compute_aspect_ratio(height,width):
        aspect = width / height
        return aspect

    @staticmethod
    def crop_to_aspect_ratio(ideal_aspect, img):

        height  = img.shape[-2]
        width = img.shape[-1]

        # Compute the aspect ratio (use simple function for consitency)
        aspect_ratio = DataTransformer.compute_aspect_ratio(
            height = height,
            width  = width
        )

        if aspect_ratio > ideal_aspect:
            # Then crop the left and right edges:
            new_width = int(height * ideal_aspect)
            offset = (width - new_width) / 2
            resize = (0, offset, height, width - offset)
        else:
            # ... crop the top and bottom:
            new_height = int(width / ideal_aspect)
            offset = (height - new_height) / 2
            resize = (offset, 0, height - offset, width)

        resize = np.array(resize,dtype=np.uint32)

        img_new = img[:,resize[0]:resize[2],resize[1]:resize[3]]

        
        return img_new

    def forward(self,
                input : torch.Tensor
                ):

        # Gray-Scale images should be expanded to third dimension
        if input.ndim == 2:
            input = input[None]
            
        if input.shape[0] == 4: # We ignore alpha channels for now. So remove
            input = input[:3]

        # Convert to default type
        dummy = torch.zeros(1)
        input = input.type_as(dummy)
        
        n_pixel_x_effective = self.num_pixel_x - 2*self.border_x
        n_pixel_y_effective = self.num_pixel_y - 2*self.border_y
        
        # Compute the aspect ratio (use simple function for consitency)
        aspect_ratio = DataTransformer.compute_aspect_ratio(
            height = n_pixel_x_effective,
            width  = n_pixel_y_effective
        )

        # First crop the image to the aspect ratio that we'll want to have in the end
        image = DataTransformer.crop_to_aspect_ratio(aspect_ratio,input)    

        if self.grayscale:
            if image.shape[0] != 1:
                image = kornia.color.rgb_to_grayscale(image)
            else:
                pass # Image is already grayscale
        
        # Resize the image
        op_resize = torchvision.transforms.Resize(
            size=[n_pixel_x_effective, n_pixel_y_effective]
            )
        image = op_resize(image)
        
        if self.border_x != 0 or self.border_y != 0:
            op_pad = torchvision.transforms.Pad(padding=[self.border_y,self.border_x, self.border_y,self.border_x])
            image = op_pad(image)

        image = DataTransformer.normalize(input = image, method =  "mean")
        
        # Image is not yet 5D ( for BPTCHW format)
        image = image[None,None]

        return image