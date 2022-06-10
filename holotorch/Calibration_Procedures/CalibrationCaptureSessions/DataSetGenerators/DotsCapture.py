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


import holotorch.utils.Homography as Homography
from holotorch.CGH_Datasets.Single_Image_Dataset import Single_Image_Dataset


def create_dot_dataset(
        sigma_smooth : int = 3,
    ):
    img_dots, _ = Homography.create_dots_image(
        N_dots_x= 5,
        N_dots_y= 8,
        border_x=200,
        border_y=200,
        num_pixel_x= 1080,
        num_pixel_y= 1920,
        kernel_size=int(10*sigma_smooth),
        sigma_smooth=sigma_smooth
    )

    dataset     = Single_Image_Dataset(
                width=1080,
                num_pixel_y= 1920,
                image=img_dots
                )
    
    return dataset
