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
from holotorch.Optical_Setups.Simple_CGH import Simple_CGH
from holotorch.CGH_Datatypes.IntensityField import IntensityField
import holotorch.CGH_Datatypes.ElectricField as ElectricField
from holotorch.utils.Enumerators import *
from holotorch.CGH_Datasets.HoloDataModule import HoloDataModule
from holotorch.Lightning_Modules.Base_Lightning import Base_Lightning

class SLM_Lightning(Base_Lightning):

    """ Class HoloLightning doc-string """


    def __init__(self,
                setup       : Simple_CGH,
                datamodule  : HoloDataModule,
                lr_slm      : float,
                lr_scale    : float             = 1e-1,
                verbose     : bool              = True,
                preload_target     : bool       = True,
                ):
        """
        
        Order of initialization methods:
        
        Please make sure that you keep this order how the different functions
        are called. Otherwise some objects might not be created and it might
        lead to unexpected errors.

        1.) Initialize class parameters
        2.) Create the forward model
        3.) Initialize the logger
        4.) Initialize the datloader
        5.) Initialize the optimizer (NOTE: optimizer required param, model, logger and dataloader)
        
        """
        # intialize the lightning model
        super().__init__(
            setup = setup,
            datamodule = datamodule,
            verbose = verbose,
            preload_target = preload_target
        )
        
        self.lr_slm     = lr_slm
        self.lr_scale   = lr_scale

    _slm_optimizer = None
    
    @property
    def slm_optimizer(self):
        return self._slm_optimizer
    
    @slm_optimizer.setter
    def slm_optimizer(self, slm_optimizer):
        self._slm_optimizer = slm_optimizer

    @property
    def slm_lightning_model(self) -> Simple_CGH:
        return self.model
    

    
    def configure_optimizers(self,
                        ):

        lr_slm = self.lr_slm
        
        model = self.slm_lightning_model
        slm = model.slm
                
        n_batches = slm.n_slm_batches
        
        # If the optimizer already exists don't set it again
        if self.slm_optimizer:
            return self.slm_optimizer
        
        slm_optimizer = []
        
        for batch_id in range(n_batches):

            # Extract the current SLM
            tmp_param = slm.load_single_slm(batch_idx=batch_id)
            
            # Create the optimizer (ADAM)
            tmp_optim = torch.optim.Adam(
                        [   
                            {"params" : tmp_param.data_tensor,     "lr" : lr_slm},
                            {"params" : tmp_param.scale,    "lr" : self.lr_scale}
                        ]
                    )

            # Add Optimizer to Dictionary
            slm_optimizer.append(tmp_optim)

                # Converting into list of tuple
        slm_optimizer = tuple(slm_optimizer)
        

        self.slm_optimizer = slm_optimizer
        
        return slm_optimizer
    
    def get_slm_optimizer(self, batch_idx : int) -> torch.optim.Optimizer:
        return self.slm_optimizer[batch_idx]
    
            
            
    def training_step(self,
                      batch : torch.Tensor,
                      batch_idx : int
                      ) -> torch.tensor:
        """ Implements the optimization step routine for Simple_CGH 
        
        Args:
            batch (torch.Tensor): [description]
            batch_idx (int): [description]

        Returns:
            torch.tensor: [description]
        """     
        
        # Get the SLM Optimizer for the specific batch
        tmp_slm_optimizer = self.get_slm_optimizer(batch_idx=batch_idx)
        # Zero grad the optimizer for this specific batch
        tmp_slm_optimizer.zero_grad()
        
        with torch.no_grad():
            if hasattr(self.model.slm, 'wrap_and_update_voltages'):
                self.model.slm.wrap_and_update_voltages(batch_idx)

        # compute output for entire dataset
        try:
            output_batch : ElectricField = self.forward(batch_idx=batch_idx)
        except TypeError:
            output_batch : ElectricField= self.forward()
        
        # Transform the batch into an Intensity Field
        batch_target = IntensityField(
                data = batch,
                wavelengths = output_batch.wavelengths)
        
        # compute the loss
        batch_loss = self.compute_loss(
            output=output_batch, target=batch_target)

        psnr = self.compute_psnr(target=batch_target, input = output_batch)

        #ssim = self.compute_ssim(target=batch_target, input = output_batch)

        # Call Manual backward
        self.manual_backward(batch_loss)
        
        tmp_slm_optimizer.step()

        out_dict = {}
        
        # out_dict is a dictionary that will be passed after training_step to lightning to further processing
        out_dict = {}

        # store the MSE loss for the model
        with torch.no_grad():
            out_dict['loss']    = batch_loss.detach().cpu()
            out_dict['psnr']    = psnr.detach().cpu()
            try:
                out_dict['ssim']    = ssim.detach().cpu()
            except NameError:
                pass

        # cleanup temporary vars on gpu used to sample pupils
        del batch_target, output_batch, batch_loss
        torch.cuda.empty_cache()

        return out_dict