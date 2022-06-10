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

from holotorch.CGH_Datasets.HoloDataModule import HoloDataModule
from holotorch.ComponentWrapper.PARAM_DATASET import PARAM_DATASET
from holotorch.CGH_Datasets.Directory_Image_Dataset import Directory_Image_Dataset
from holotorch.CGH_Datasets.Single_Image_Dataset import Single_Image_Dataset
from holotorch.CGH_Datasets.Captured_Image_Dataset import Captured_Image_Dataset
from holotorch.utils.Enumerators import *

try:
    from holotorch.CGH_Datasets.LightField_EXR_Dataset import LightField_EXR_Dataset
except:
    pass

def create_data_module(
    param : PARAM_DATASET
) -> HoloDataModule:
    
    
    module =  create_data_module_from_param(
        data_folder     =param.data_folder,
        data_sz         =param.data_sz,
        num_pixel_x     = param.num_pixel_x,
        num_pixel_y     = param.num_pixel_y,
        border_x        = param.border_x,
        border_y        = param.border_y,
        batch_size      = param.batch_size,
        TYPE_dataloader = param.TYPE_dataloader,
        color_flag      = param.color_flag,
        path            = param.path
    )
    
    assert module is not None, "Data Module cannot be none"
    
    return module

def create_data_module_from_param(
    data_folder     : str,
    data_sz         : int,
    path            : str,
    num_pixel_x     : int,
    num_pixel_y     : int,
    border_x        : int,
    border_y        : int,
    batch_size      : int,
    TYPE_dataloader : ENUM_DATASET,
    color_flag      : ENUM_SENSOR_TYPE,
        ) -> HoloDataModule:

    if color_flag is ENUM_SENSOR_TYPE.BAYER:
        flag_grayscale = False
    elif color_flag is ENUM_SENSOR_TYPE.MONOCHROMATIC:
        flag_grayscale = True

    if TYPE_dataloader == ENUM_DATASET.DIV2K_Dataset:
        dataset = Directory_Image_Dataset(
            num_pixel_x = num_pixel_x,
            num_pixel_y = num_pixel_y,
            border_x    = border_x,
            border_y    = border_y,
            img_dir=data_folder,
            data_sz = data_sz,
            batch_size = batch_size,
            grayscale=flag_grayscale
            )
    elif TYPE_dataloader == ENUM_DATASET.Single_Image_Loader:
        dataset = Single_Image_Dataset(
            num_pixel_x = num_pixel_x,
            num_pixel_y = num_pixel_y,
            border_x    = border_x,
            border_y    = border_y,
            path = path,
            grayscale=flag_grayscale
            )
    elif TYPE_dataloader == ENUM_DATASET.LightField_EXR_Loader:
        dataset = LightField_EXR_Dataset(
            num_pixel_x = num_pixel_x,
            num_pixel_y = num_pixel_y,
            lf_folder   = data_folder,
            data_sz     = data_sz,
            batch_size  = batch_size,
            grayscale   = flag_grayscale
            )
    elif TYPE_dataloader == ENUM_DATASET.Captured_Image_Dataset:
        dataset = Captured_Image_Dataset(
            num_pixel_x = num_pixel_x,
            num_pixel_y = num_pixel_y,
            img_dir     = data_folder,
            data_sz     = data_sz,
            grayscale   = flag_grayscale
            )


    data_module = HoloDataModule(
        dataset=dataset,
        batch_size=batch_size,
        shuffle                 = False,
        shuffle_for_init        = False,
    )
    return data_module