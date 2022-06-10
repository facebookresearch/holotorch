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
from holotorch.utils.Helper_Functions import tt

def helper_set_function(object : object, name : str, attr_opt_name : str, val):
    
    # print("Set Attr-Name: " + name)
    # print(object.__class__)
    # print("Set Attr-val: " + str(val))

    if hasattr(object, name):
        delattr(object,name)
    
    if not torch.is_tensor(val):
        val = tt(val)
    
    opt_flag =  getattr(object, attr_opt_name)

    if opt_flag == True:
        object.register_parameter(name, torch.nn.Parameter(val))
    elif opt_flag == False:
        object.register_buffer(name, val)
    else:
        raise ValueError("opt_flag has not been set yet")
    

def helper_get_function(object, name : str):
    #print("Get Attr-Name: " + name)
    if hasattr(object, name):
        return getattr(object, name)

def set_torch_attribute(name : str, attr_opt_name : str):
    #print("Add attribute with name: " + name)
    my_func = lambda object, val: helper_set_function(object, name, attr_opt_name, val)
    return my_func

def get_torch_attribute(name):
    my_func = lambda object: helper_get_function(object, name)
    return my_func


def helper_set_opt_function(object : object, attr_name : str, hidden_name_opt : str, val : bool):
    
    setattr(object,hidden_name_opt,val)

    if hasattr(object, attr_name):
        # Moving beta to a tensor first is important because if 
        # the variable is Parameter, PyTorch will do weird stuff
        foo = getattr(object,attr_name)
        # Check if the attribute is not none ==> Reset the buffer/parameter
        if foo is not None:
            setattr(object,attr_name, tt(foo))
    
def helper_get_opt_function(object, name : str) -> bool:
    #print("Get Attr-Name: " + name)
    if hasattr(object, name):
        return getattr(object, name)
    

def set_torch_opt_attribute(attr_name, hidden_name_opt):
    #print("Add attribute with name: " + name)
    my_func = lambda self, val: helper_set_opt_function(self, attr_name=attr_name, hidden_name_opt = hidden_name_opt, val = val)
    return my_func

def get_torch_opt_attribute(name):
    my_func = lambda self: helper_get_opt_function(self, name)
    return my_func

def add_attribute(object, attr_name = "test"):
    
    hidden_name = "_" + attr_name
    attr_opt_name = attr_name + "_opt"
    hidden_name_opt = "_" + attr_opt_name
    
    fget = get_torch_attribute(hidden_name)
    fset = set_torch_attribute(hidden_name, attr_opt_name)

    fget_opt = get_torch_opt_attribute(hidden_name_opt)
    fset_opt = set_torch_opt_attribute(attr_name, hidden_name_opt)    

    setattr(object, attr_name, property(fget=fget, fset=fset))
    setattr(object, attr_opt_name, property(fget=fget_opt, fset=fset_opt))

    
