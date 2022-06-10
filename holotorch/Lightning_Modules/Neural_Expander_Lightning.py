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
import numpy as np
from holotorch.Optical_Setups import Expansion_setup
from holotorch.CGH_Datatypes.IntensityField import IntensityField
from holotorch.CGH_Datasets.HoloDataModule import HoloDataModule
from holotorch.Lightning_Modules.SLM_Lightning import SLM_Lightning

class Neural_Expander_Lightning(SLM_Lightning):
    
    """ Class HoloLightning doc-string """


    def __init__(self,
                setup       : Expansion_setup,
                datamodule  : HoloDataModule,
                lr_slm      : float,
                lr_expander : float,
                verbose     : bool              = True,
                num_preinitialize : int         = 10,
                ):

        # intialize the lightning model
        super().__init__(
            setup = setup,
            datamodule = datamodule,
            verbose = verbose,
            lr_slm=lr_slm
        )
        
        self.num_preinitialize = num_preinitialize
        
        self.lr_expander = lr_expander
        
    key_slm_optimizer       = None
    key_expander_optimizer  = None

    _expander_optimizer       = None
    
    @property
    def expander_optimizer(self):
        return self._expander_optimizer
    
    @expander_optimizer.setter
    def expander_optimizer(self, optim):
        self._expander_optimizer = optim
        
    def get_expander_optimizer(self) -> torch.optim.Optimizer:
                    
        return self.expander_optimizer
        
    def configure_optimizers(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        optimizer_slm = super().configure_optimizers()
        
        optimizer_expander = self.configure_expander_optimizer()
        
        if type(optimizer_expander) is not tuple:
            optimizer_expander = tuple([optimizer_expander])
        
        optimizers =optimizer_slm + optimizer_expander
        
        self.key_slm_optimizer = np.arange(0,len(optimizer_slm))
        self.key_expander_optimizer = np.arange(0,len(optimizer_expander)) + self.key_slm_optimizer[-1] + 1
        
        return optimizers
    
    @property
    def model(self) -> Expansion_setup:
        return super().model        

    def configure_expander_optimizer(self):
        
        if self.expander_optimizer is not None:
            return self.expander_optimizer
        
        param = self.model.expander.parameters()
        # Create the optimizer (ADAM)
        tmp_optim = torch.optim.Adam(
                params=param,
                lr= self.lr_expander
                )
        
        self.expander_optimizer = tmp_optim
        
        return tmp_optim

    
            
    def training_step(self,
                      batch : torch.Tensor,
                      batch_idx : int,
                    
                     ) -> torch.tensor:
        """ Implements the optimization step routine for Simple_CGH 
        
        Args:
            batch (torch.Tensor): [description]
            batch_idx (int): [description]

        Returns:
            torch.tensor: [description]
        """

        if self.current_it_number < self.num_preinitialize:
            print("It " + str(self.current_it_number) + " ( Batch " + str(batch_idx) + "): Pre-initialize model with SLM-Optimization")
            out = super().training_step(batch = batch,
                                         batch_idx=batch_idx)
            return out
        else: 
            print("It " + str(self.current_it_number) + " ( Batch " + str(batch_idx) + "): Run Neural Optimization.")

        manual_backward   = self.manual_backward

        # Get the optimizers
        slm_optimizer = self.get_slm_optimizer(batch_idx=batch_idx)
        expander_optmizer = self.get_expander_optimizer()
    
        # Manually zero grad the optimizers
        slm_optimizer.zero_grad()
        expander_optmizer.zero_grad()

        # compute output for entire dataset
        output_batch : IntensityField = self.forward(batch_idx=batch_idx)
        
        # Transform the batch into an Intensity Field
        batch_target = IntensityField(
                data = batch,
                wavelengths = output_batch.wavelengths
                )

        # compute the loss
        batch_loss = self.compute_loss(
            output=output_batch,
            target=batch_target
            )

        psnr = self.compute_psnr(target=batch_target, input = output_batch)


        # Call Manual backward
        manual_backward(batch_loss)
        
        # Step the optimizers
        slm_optimizer.step()
        expander_optmizer.step()
        
        out_dict = {}
        
        out_dict['loss'] = batch_loss
        out_dict['output_batch'] = output_batch.detach()
        out_dict['psnr']    = psnr.detach().cpu()
        out_dict['validation_loss'] = batch_loss.detach()

        return out_dict
    
    def visualize_epoch(self, batch_idx=0):

        super().visualize_epoch(batch_idx=0)
        
        self.model.expander.view_expander(save_id = None, current_it_number = None)