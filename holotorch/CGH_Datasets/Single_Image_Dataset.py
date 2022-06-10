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
import cv2
import torch
import torch
import torchvision
import pathlib

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from holotorch.CGH_Datasets.CGH_Dataset import CGH_Dataset
from holotorch.CGH_Datasets.DataTransformer import DataTransformer

class Single_Image_Dataset(CGH_Dataset):

    def __init__(self,
        num_pixel_x = None, # x is height
        image : torch.Tensor = None,
        path : pathlib.Path or str = None,
        grayscale = False,
        num_pixel_y  :int = None, # y is width
        border_x   : int = 0,
        border_y   : int = 0,
        data_sz    : int = 1,
        ):
        """
        
        Args:
        
        """
        super().__init__()
        
        if num_pixel_x is None:
            self._set_single_image(image)

        
        self.transform  = DataTransformer(
            num_pixel_x = num_pixel_x,
            num_pixel_y = num_pixel_y,
            border_x    = border_x,
            border_y    = border_y,
            grayscale   = grayscale,
        )
        
        self.data_sz = data_sz
        
        if path is not None:
            image = self._read_image(path)
            
        if image is not None:
            image = self._process_single_image(image)
            self._set_single_image(image)
    
    def set_image(self, image : torch.Tensor = None, path : torch.Tensor = None):
        assert (image is None and path is None) == False
        
        if path is not None:
            image = self._read_image(path)
            
        if image is not None:
            image = self._process_single_image(image)
        
        self._set_single_image(image)
    
    def _read_image(self, path):

        try:
            image = torchvision.io.read_image(path)
        except:
            image = cv2.imread(filename= path)
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = torch.tensor(image)
                image = image.permute(2,0,1)
            image = torch.tensor(image)
            image = image / image.max()

        return image
    
    def _process_single_image(self, image : torch.Tensor):
        """ Assume that image is 3D tensor
        
        Can be either grayscale or color as input

        Args:
            image (torch.Tensor): [description]
        """        
        image = self.transform(image)
        return image
    
    def _set_single_image(self, image : torch.Tensor):
        self._image = image

    def _get_single_image(self):
        return self._image

    def __getitem__(self, idx):
        """
        
        """
        
        if idx >= len(self): raise IndexError

        image = self._get_single_image()
                
        self.current_batch = image

        return image
                
    def __len__(self):
        return self.data_sz

    def pre_load_dataset(self):
        pass

    def set_test_dataset(self):
        pass
    
    def set_train_dataset(self):
        pass
        
    def show_sample(self):
        pass