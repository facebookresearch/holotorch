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

import pathlib
import torch
import matplotlib.pyplot as plt

from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.utils.Dimensions import TensorDimension
from holotorch.utils.Enumerators import *
from holotorch.utils.Visualization_Helper import add_colorbar

class ValueContainer(CGH_Component):
    
    def __init__(self,
                 tensor_dimension : TensorDimension,
                 init_type       : ENUM_SLM_INIT = None,
                 init_variance   : float = 0,
                 flag_complex    : bool = False,
                 ) -> None:
        super().__init__()
        
        self.tensor_dimension = tensor_dimension

        self.flag_complex   = flag_complex
        self.init_type      = init_type
        self.init_variance  = init_variance
        
        # Those values shouldn't be accessed, hence we make them private
        self.add_attribute(attr_name="data_tensor")
        self.data_tensor_opt = True


        self.add_attribute(attr_name="scale")
        self.scale_opt = True
        
        
        self.set_images_per_batch()

    @staticmethod
    def load_data_from_file(path : pathlib.Path):
        single_slm = torch.load(path)
        return single_slm['_data_tensor']


    @property
    def shape(self):
        return self.tensor_dimension.shape
    
    @property
    def images_per_batch(self) -> int:
        return self.tensor_dimension.batch

    def set_images_per_batch(self, number_images_per_batch : int = None):

        if number_images_per_batch == None:
            number_images_per_batch = self.images_per_batch
        
        self.tensor_dimension.batch = number_images_per_batch
    
        self.data_tensor = ValueContainer.compute_init_tensor(
            tensor_shape    = self.tensor_dimension.shape,
            init_type       = self.init_type,
            init_variance   = self.init_variance,
            flag_complex    = self.flag_complex
        )
        
        self.scale = torch.ones(self.tensor_dimension.shape[:-2])
        
    def __str__(self) -> str:
        my_str = str(self.data_tensor.shape)
        #my_str += "/n" + str(self.scale.detach().cpu())
        return my_str
    
    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def compute_init_tensor(
            init_type        : ENUM_SLM_INIT,
            tensor_shape     : TensorDimension,
            init_variance           = None,
            flag_complex      : bool = False
        ) -> torch.Tensor:
        """ Computes the data tensor for one value depending on the init-type

        Raises:
            NotImplementedError: If an init type is passed that's not implemented yet.

        Returns:
            torch.Tensor: [description]
        """        

        if init_type == ENUM_SLM_INIT.RANDOM:           
            my_tensor = init_variance*torch.rand(tensor_shape, requires_grad=True)
            
            if flag_complex:
                my_tensor = my_tensor + 1j * init_variance*torch.rand(tensor_shape, requires_grad=True)
                
        elif init_type == ENUM_SLM_INIT.ZEROS:
            my_tensor = torch.zeros(tensor_shape, requires_grad=True)
            if flag_complex:
                my_tensor += 0j
        elif init_type == ENUM_SLM_INIT.ONES:
            my_tensor = torch.ones(tensor_shape, requires_grad=True)
            if flag_complex:
                my_tensor += 0j
        else:
            raise NotImplementedError("NOT YET IMPLEMENTED")
        
 
        
        return my_tensor
     

    def visualize(self):
        
        add_colorbar(plt.imshow(self.data_tensor.detach().cpu().squeeze()))
