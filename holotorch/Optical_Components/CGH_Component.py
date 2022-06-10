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
import holotorch.utils.torch_buffer_switcher as torch_buffer_switcher

class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.detach().cpu())
        
    def clear(self):
        self.outputs = []

class CGH_Component(torch.nn.Module):
    """[summary]

    Args:
        torch ([type]): [description]
    """    
    
    # Stores the handle for a possible forward hook
    handle = None
    save_output = None
    attribute_list = None

    def __init__(self) -> None:
        super().__init__()
        self.attribute_list = []

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)
    
    def __str__(self) -> str:
        
        tmp = self.__class__.__bases__[0].__name__
        mystr = "=======================================================\n"
        mystr += "CLASS Name: " + type(self).__name__ + " (extended from " + tmp + ")"
        #mystr += "\n"
        #mystr += "======================================================="

        return mystr

    

    def renitialize_attributes(self):
        """ When a pickled object is loaded non-static references are lost, so we need to rebuild them.
        """
        
        # First we need to reit
        if self.attribute_list is None:
            pass
        else:
            for name in self.attribute_list:
                #print(name)
                self.add_attribute(attr_name=name)
            
        for idx, name in enumerate(self._modules):
            #print(idx)
            #print(name)
            child = self._modules[name]
            #print(parent_module)
            try:
                child.renitialize_attributes()
            except AttributeError:
                pass
            

    def add_attribute(self, attr_name = "test"):
        object = self.__class__
        torch_buffer_switcher.add_attribute(object=object, attr_name=attr_name)
        
        # NOTE: It's absolutely essential to add the attribute list 
        # here in order to have pickling work properly
        # Note, that the attribute list is called in "reinitlaize attributes"
        if attr_name in self.attribute_list:
            pass
        else:
            self.attribute_list.append(attr_name)
        
    def __getattr__(self, item):
        return super().__getattr__(item)

    def __setattr__(self, key, value, write_to_dict = False):
        super().__setattr__(key, value)
        if write_to_dict == True:
            self.__dict__[key] = value
          
    def print_state_dict_nice(self):
        state_dict = self.state_dict()

        for var_name  in enumerate(state_dict):
            print(var_name )
            data = state_dict[var_name[1]]
            print(data.shape)
            print(data.device)
            
    def print_param_nice(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
            else:
                print("Requires Grad(false):", name, param.shape)

    def add_output_hook(self):
        """Adds the hook the particular forward function
        """
        
        if self.handle is None:
            # Create the Hook Class
            self.save_output = SaveOutput()
            self.handle = self.register_forward_hook(self.save_output)
    
    @property
    def outputs(self):
        return self.save_output.outputs

    def clear_outputs(self):
        if self.save_output is not None:
            self.save_output.clear()
    
    def remove_output_hook(self):
        """Gives the option to remove the handle again
        
        Also clears 
        """        
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            self.save_output = None
