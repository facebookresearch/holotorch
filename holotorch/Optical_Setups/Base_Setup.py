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

from __future__ import annotations
from abc import abstractmethod
import torch
import pathlib
import pickle

from holotorch.ComponentWrapper.PARAM_COMPONENT import PARAM_COMPONENT
from holotorch.ComponentWrapper.PARAM_DETECTOR import PARAM_DETECTOR
from holotorch.ComponentWrapper.PARAM_EXPANDER import PARAM_EXPANDER
from holotorch.ComponentWrapper.PARAM_SLM import PARAM_SLM
from holotorch.ComponentWrapper.PARAM_PUPIL import PARAM_PUPIL
from holotorch.ComponentWrapper.PARAM_SLM import PARAM_SLM
from holotorch.ComponentWrapper.PARAM_SLM import PARAM_SLM
from holotorch.ComponentWrapper.PARAM_SOURCE import PARAM_SOURCE
from holotorch.ComponentWrapper.PARAM_PROPAGATOR import PARAM_PROPAGATOR

from holotorch.ComponentWrapper.PARAM_SLM_UPSAMPLE import PARAM_SLM_UPSAMPLE
from holotorch.ComponentWrapper.SLM_Upsample_Factory import create_slm_upsample
from holotorch.ComponentWrapper.SourceFactory import create_source
from holotorch.ComponentWrapper.SLM_Factory import create_slm
from holotorch.ComponentWrapper.Expander_Factory import create_expander
from holotorch.ComponentWrapper.Propagator_Factory import create_propagator
from holotorch.ComponentWrapper.Detector_Factory import create_detector
from holotorch.ComponentWrapper.Pupil_Factory import create_pupil
from holotorch.Optical_Components.CGH_Component import CGH_Component
    
class Base_Setup(CGH_Component):
    
    # Create the empy component dict
    
    def __init__(self,
                 filename : str or pathlib.Path = None
        ) -> None:
        super().__init__()
        self.filename = filename
            
    @staticmethod
    def load_pickle_object(filename) -> Base_Setup:
        outfile = open(filename,'rb')
        setup = pickle.load(outfile)
        outfile.close()
        
        # We need to set every module manually again
        for idx, name in enumerate(setup._modules):
            obj = setup._modules[name]
            setup.add_component(name=name, component=obj)

        setup.renitialize_attributes()
        return setup

    def save_model(self, filename):
        """Saves the DPAC model as a pickled file to the path

        Args:
            filename (_type_): _description_
        """        

        path = pathlib.Path(filename)
        path.parent.mkdir(exist_ok=True,parents=True)

        outfile = open(filename,'wb')
        pickle.dump(self,outfile)
        outfile.close()    

    @abstractmethod
    def forward(self,
                input : any = None,
                channel_idxs : torch.Tensor = None,
                ):
        """[summary]

        Args:
            input (any): Possibly input to the forward method. E.g. this could be pattern to be displayed
            channel_idxs (torch.Tensor, optional): [description]. Defaults to None.

        Raises:
            NotImplementedError: [description]
        """        
        raise NotImplementedError
        
    @staticmethod
    def create_component_from_param( 
                    component : PARAM_COMPONENT
                    ) -> CGH_Component:
        
        if isinstance(component, PARAM_SOURCE):
            cgh_component = create_source(component)
        elif isinstance(component, PARAM_SLM):
            cgh_component = create_slm(component)
        elif isinstance(component, PARAM_EXPANDER):
            cgh_component = create_expander(component)
        elif isinstance(component, PARAM_PROPAGATOR):
            cgh_component = create_propagator(component)
        elif isinstance(component, PARAM_DETECTOR):
            cgh_component = create_detector(component)
        elif isinstance(component, PARAM_SLM_UPSAMPLE):
            cgh_component = create_slm_upsample(component)
        elif isinstance(component, PARAM_PUPIL):
            cgh_component = create_pupil(component)
            
        if cgh_component is None:
            raise ValueError("CGH Component should not be none. Check input.")
        
        return cgh_component

    def add_component(self,
            name : str,
            component : CGH_Component or PARAM_COMPONENT
                    ) -> None:
        
        if isinstance(component, PARAM_COMPONENT):
            component = Base_Setup.create_component_from_param(component)
        
        assert isinstance(component, CGH_Component), "Component needs to be a CGH Component."
        # Set the component as a member variable so that it appears in state and param dicts
        self.__setattr__(name, component, write_to_dict = True)
    
    def remove_component(self, component : PARAM_COMPONENT) -> None:
        pass
    
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
                print(name, param.shape, param.device)
        
        flag_false = False
        for name, param in self.named_parameters():
            if param.requires_grad == False:
                if flag_false == False:
                    print("Parameters where requires_grad == False:")
                    flag_false = True
                    
                print(name, param.shape)
            
    def clear_cache(self):
        with torch.no_grad():
            torch.cuda.empty_cache()
        torch.cuda.empty_cache() 
        import gc
        gc.collect() # Python thing
        torch.cuda.empty_cache()
        with torch.no_grad():
            torch.cuda.empty_cache()
            

        