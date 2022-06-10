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
from holotorch.Sensors.Detector import Detector
from holotorch.utils.Enumerators import *
from holotorch.utils.units import *
from holotorch.LightSources.Source import Source
from holotorch.Optical_Propagators import Propagator
from holotorch.HolographicComponents import Modulator_Container
from holotorch.HolographicComponents import Hologram_Model
from holotorch.ComponentWrapper.PARAM_SOURCE import PARAM_SOURCE
from holotorch.ComponentWrapper.PARAM_SLM import PARAM_SLM
from holotorch.ComponentWrapper.PARAM_PUPIL import PARAM_PUPIL
from holotorch.ComponentWrapper.PARAM_EXPANDER import PARAM_EXPANDER
from holotorch.ComponentWrapper.PARAM_PROPAGATOR import PARAM_PROPAGATOR
from holotorch.ComponentWrapper.PARAM_DETECTOR import PARAM_DETECTOR

class Expansion_setup(Base_Setup):
    
    def __init__(self,
            source      : Source or PARAM_SOURCE,
            slm         : Modulator_Container or PARAM_SLM,
            expander    : Hologram_Model or PARAM_EXPANDER,
            propagator  : Propagator or PARAM_PROPAGATOR,
            detector    : Detector or PARAM_DETECTOR,
            pupil       : CGH_Component or PARAM_PUPIL = PARAM_PUPIL(),
                 ) -> None:
        super().__init__()
                
        self.add_component( name = 'source', component = source)
        self.add_component( name = 'slm', component = slm)            
        self.add_component(name = 'expander', component = expander)
        self.add_component(name = 'propagator', component = propagator)
        self.add_component(name = 'pupil', component = pupil)
        self.add_component(name = 'detector', component = detector)

        self.__normalize_output_to_be_unit_mean__()


    def forward(self,
                input = None,
                batch_idx       : int           = 0,
                channel_idxs    : torch.Tensor  = None,
                flag_debug      : bool          = False,
                extract_output  : list or int   = None,
                ):
        
        assert batch_idx < self.slm.n_slm_batches, "Batch index must be smaller than n_slm_batches "
        
        #
        # Get the components for better readibility of code
        #
        source   = self.source
        slm      = self.slm
        expander = self.expander
        pupil    = self.pupil
        prop     = self.propagator
        detector = self.detector
        
        #
        # Assemble the optical path
        #
        
        # STEP 1 
        field = source(channel_idxs=channel_idxs)
        # STEP 2   
        field = slm(field, batch_idx = batch_idx)
        # STEP 3
        field = expander(field)
        # STEP 4
        field = pupil(field)
        # STEP 5
        field = prop(field)
        # STEP 6
        intensity_field = detector(field)
        
        return intensity_field
    
    @property
    def source(self) -> Source:
        return self.component_dict['source']
    
    @property
    def slm(self) -> Modulator_Container:
        return self.component_dict['slm']
        
    @property
    def expander(self) -> Hologram_Model:
        return self.component_dict['expander']
    
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
        print(self.slm)
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
