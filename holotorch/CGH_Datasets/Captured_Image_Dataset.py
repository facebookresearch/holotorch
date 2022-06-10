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
import tifffile
import os
import torch
import pandas as pd
import numpy as np
import os, torch
import pandas as pd
import numpy as np
import glob
from torch.utils.data import Subset
import tifffile

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from holotorch.CGH_Datasets.CGH_Dataset import CGH_Dataset
import holotorch.utils.Homography as Homography

class Captured_Image_Dataset(CGH_Dataset):

    # Pre-initialize current batch to none
    current_batch = None

    def __init__(self,
            img_dir : str,
            homography_matrix : torch.Tensor = None,
            target_shape = None,
            data_sz = None,
            prefix = "",
            normalize_flag = True,
        ):
        """
        
        Args:
        
        """
        super().__init__()

        # Set the directory where images are stored                
        self.img_dir = img_dir
        
        self.homography_matrix = homography_matrix

        self.target_shape = target_shape



        # Get the Filnames
        
        files = []
        for ext in ('.gif', '.png', '.jpg', '.jpeg', '.tiff', '.tif'):
            ext = '*' + prefix + '*' + ext
            tmp_files = glob.glob(os.path.join(self.img_dir, ext))
            names = [os.path.basename(x) for x in tmp_files]

            files.extend(names)
        self.img_labels = pd.DataFrame(files)

        if data_sz is None:
            data_sz = len(self.img_labels)
        # Setting the datasize automatically sets the training subset (NOTE: property method)
        self.data_sz = data_sz


        self.pre_loaded = False   

        self.normalize_flag = normalize_flag      

    @property
    def data_sz(self):
        return len(self.train_dataset)
    
    @data_sz.setter
    def data_sz(self, data_sz : int):
        self.train_dataset = Subset(self, np.arange(data_sz))

    def get_preload_path(self,
        idx : int
        ):
        
        filename = self.img_labels.iloc[idx,0]
        base = os.path.splitext(filename)
        filename = base[0] + '.pt'
        
        path = os.path.join(self.img_dir, filename )
        return path

    def pre_load_dataset(self,
                         load_on_gpu : bool = False
                         ):
        '''
        save transformed data in tensor format
        '''
        
        self.gpu_preloaded_dataset = []

        if load_on_gpu:
            self.load_on_gpu = True
        else:
            self.load_on_gpu = False
        
        for idx in range(self.data_sz):
            image = self.read_and_process_image(idx)
            
            if load_on_gpu:
                self.gpu_preloaded_dataset.append(image)
            else:
                tensor_path = self.get_preload_path(idx)
                torch.save(image, tensor_path)
               

            
        self.pre_loaded = True
        
    def get_preloaded(self,
                      idx : int,
                      ):
        # print("loading pre-transformed tensor: ", str(idx), end = '')
        if self.load_on_gpu:
            images = self.gpu_preloaded_dataset[idx]
        else:
            tensor_path = self.get_preload_path(idx)
            images = torch.load(tensor_path)
        return images
    
    def save_as_tiff(self,
                     prefix = "homography",
                     save_dir = None):
        '''
        save transformed data in tensor format
        '''
        
        if save_dir == None:
            save_dir = self.img_dir

        for idx in range(self.data_sz):
            image = self.__getitem__(idx).squeeze().cpu().numpy()
            name = self.img_labels.iloc[idx,0]
                        
            tensor_path = os.path.join(save_dir, name)
            
            tifffile.imsave(tensor_path, image.astype(np.float32) , imagej=True)


    def get_image_path(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        return img_path
    
    def read_and_process_image(self, idx):
        img_path = self.get_image_path(idx)
        
        # assumes that path is .tiff-file
        image = tifffile.imread(img_path)

        image = torch.tensor(image)
        
        image = self.apply_homography(image)

        if self.normalize_flag:
            image = image / image.mean()
        
        image = image[None,None,None]
        
        return image

    def apply_homography(self, x : torch.Tensor):
        
        if self.homography_matrix is None:
            return x
        
        img_warped = Homography.warp_image(
            image = x,
            target_shape = self.target_shape,
            homography = self.homography_matrix.to(device=x.device)
        )
        return img_warped

    def __getitem__(self, idx):
        """
        
        """
        if idx >= len(self):
            raise IndexError
            
            
        if self.pre_loaded:
            image = self.get_preloaded(idx)
        else:
            image = self.read_and_process_image(idx)
                
        self.current_batch = image

        return image
                
    def __len__(self):
        return self.data_sz

    def set_test_dataset(self):
        pass
    
    def set_train_dataset(self):
        pass
        
    def show_sample(self):
        pass

    def imshow_current_batch(self):
        """
        
        
        """

        if self.current_batch is None:
            raise ValueError("Current Batch has not been loaded yet. Please load.")

        self.current_batch.visualize()