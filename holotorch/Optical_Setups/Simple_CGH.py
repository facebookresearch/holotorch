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
from holotorch.Optical_Setups.Base_Setup import Base_Setup
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.utils.Enumerators import *
from holotorch.utils.units import *
from holotorch.ComponentWrapper import *
from holotorch.Sensors.Detector import Detector
from holotorch.LightSources.Source import Source
from holotorch.Optical_Propagators import Propagator
from holotorch.HolographicComponents import Modulator_Container

from holotorch.ComponentWrapper.PARAM_SOURCE import PARAM_SOURCE
from holotorch.ComponentWrapper.PARAM_SLM import PARAM_SLM
from holotorch.ComponentWrapper.PARAM_PUPIL import PARAM_PUPIL
from holotorch.ComponentWrapper.PARAM_PROPAGATOR import PARAM_PROPAGATOR
from holotorch.ComponentWrapper.PARAM_DETECTOR import PARAM_DETECTOR

class Simple_CGH(Base_Setup):
    
    def __init__(self,
            source          : Source or PARAM_SOURCE,
            slm             : Modulator_Container or PARAM_SLM,
            propagator      : Propagator or PARAM_PROPAGATOR,
            pupil           : CGH_Component or PARAM_PUPIL,
            detector        : Detector or PARAM_DETECTOR,
            normalize       : bool = True
                 ) -> None:
        super().__init__()
                
        self.add_component( name = 'source', component = source)
        
        if not isinstance(slm, list):
            slm = [slm]
        
        for idx_slm, tmp_slm in enumerate(slm):
            tmp_name = "slm_" + str(idx_slm)
            self.add_component( name = tmp_name, component = tmp_slm)
            del tmp_name, tmp_slm
            
        self.add_component(name = 'propagator', component = propagator)
        self.add_component(name = 'pupil', component = pupil)
        self.add_component(name = 'detector', component = detector)

        if normalize:
            self.__normalize_output_to_be_unit_mean__()

    @staticmethod
    def create_setup_from_param(
            source          : PARAM_SOURCE,
            slm             : PARAM_SLM,
            propagator      : PARAM_PROPAGATOR,
            pupil           : PARAM_PUPIL,
            detector        : PARAM_DETECTOR,
            ):
        
        source          = super().create_component_from_param(component=source)
        slm             = super().create_component_from_param(component=slm)
        propagator      = super().create_component_from_param(component=propagator)
        pupil           = super().create_component_from_param(component=pupil)
        detector        = super().create_component_from_param(component=detector)
        
        expansion_setup     = Simple_CGH(
            source          = source,
            slm             = slm,
            propagator      = propagator,
            pupil           = pupil,
            detector        = detector
        )
        
        return expansion_setup

    def forward(self,
                batch_idx       : int           = 0,
                flag_debug      : bool          = False,
                extract_output  : list or int   = None,
                bit_depth       : int or None   = None,
                modulation_input = None
                ):
        
        #
        # Get the components for better readibility of code
        #
        source          = self.source
        slm             = self.slm
        pupil           = self.pupil
        prop            = self.propagator
        detector        = self.detector
        
        #
        # Assemble the optical path
        #
        
        # STEP 1 
        field = source()
        # STEP 2   
        field = slm(field, batch_idx = batch_idx, bit_depth = bit_depth)
        # STEP 3
        field = pupil(field)
        # STEP 4
        field = prop(field)
        # STEP 5
        intensity_field = detector(field)

        # normalize output by mean intensity
        mean_intensity = intensity_field.data.mean(dim=(4,5))
        intensity_field.data.div_(mean_intensity[:,:,:,:,None,None])
        
        # apply a scaling factor
        intensity_field.data.multiply_(self.slm.values.scale[:,None,None,None,None,None]**2)

        return intensity_field
    
    @property
    def source(self) -> Source:
        return self.component_dict['source']
    
    @property
    def slm_0(self) -> Modulator_Container:
        return self.component_dict['slm_0']
    
    @property
    def slm(self) -> Modulator_Container:
        return self.slm_0
    
    @property
    def pupil(self) -> Propagator:
        return self.component_dict['pupil']
    
    @property
    def propagator(self) -> Propagator:
        return self.component_dict['propagator']

    @property
    def detector(self) -> Propagator:
        return self.component_dict['detector']
    
    def print_components(self):
        print(self.source)
        print(self.slm_0)
        print(self.pupil)
        print(self.propagator)     
        print(self.detector)       

    

    def __normalize_output_to_be_unit_mean__(self):

        with torch.no_grad():
            # TODO: normalize each slm separately
            # calculate the output of the SLM from first batch
            out = self.forward(batch_idx=0)
            light_budget = out.data.mean()

            # normalize input to have unit energy output
            self.source.scale = 1 / torch.sqrt(light_budget)

            # TODO: normalize each slm separately
            # calculate the output of the SLM from first
            test_out = self.forward(batch_idx=0)
            light_budget = test_out.data.mean()

            if light_budget - 1.0 > 1e-6:
                raise ValueError("Error normalizing output to unit mean")
