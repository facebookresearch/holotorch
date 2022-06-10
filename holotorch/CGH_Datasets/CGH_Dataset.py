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

from abc import abstractmethod, ABC, ABCMeta
import six
import torch
import os

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from holotorch.utils.string_processor import convert_integer_into_string
@six.add_metaclass(ABCMeta)
class CGH_Dataset(torch.utils.data.Dataset, ABC):

    def __init__(self):
        """
        Args:

        """

    def __len__(self):
        """[summary]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def pre_load_dataset(self, save_dir : str = None):
        '''
        save transformed data in tensor format
        '''
        
        if save_dir == None:
            save_dir = self.img_dir
        
        for idx in range(len(self)):
            image = self.__getitem__(idx)
            tensor_path = os.path.join(save_dir, convert_integer_into_string(idx,depth=4) + '.pt')
            torch.save(image, tensor_path)
        self.pre_loaded = True

    def get_preloaded(self,
                      idx : int
                      ):
        # print("loading pre-transformed tensor: ", str(idx), end = '')
        tensor_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0] + '.pt')
        images = torch.load(tensor_path)
        self.current_batch = images
        return images

    @property
    def num_pixel_x(self):
        
        item = self[0]
        
        if isinstance(item,tuple) or isinstance(item,list) :
            item = item[0]
        return item.shape[-2]
    
    @property
    def num_pixel_y(self):
        
        item = self[0]
        
        if isinstance(item,tuple) or isinstance(item,list) :
            item = item[0]
        return item.shape[-1]
    
    @abstractmethod
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def show_sample(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError


