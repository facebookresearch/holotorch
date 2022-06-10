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
import itertools
import matplotlib.pyplot as plt
import os
import kornia
import kornia.metrics
import numpy as np
from typing import Any

# Import Torch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.loggers.base import LoggerCollection

# Import of HoloTorch Objects
from holotorch.CGH_Datatypes.Light import Light
from holotorch.Optical_Setups.Base_Setup import Base_Setup
from holotorch.CGH_Datatypes.IntensityField import IntensityField
from holotorch.CGH_Datasets.HoloDataModule import HoloDataModule
from holotorch.utils.transformer_6d_4d import transform_6D_to_4D

class Base_Lightning(pl.LightningModule):

    """ Class HoloLightning doc-string """


    def __init__(self,
                setup       : Base_Setup,
                datamodule  : HoloDataModule,
                verbose     : bool              = True,
                preload_target     : bool       = True,
                log_model_every_n_steps : int   = None,
                log_model_output_every_n_steps : int = None,
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
        super().__init__()
        


        self.model = setup
                
        # ----------------------------------------------------------------------------
        # Member Variables of CGH Framework (not exhaustive)
        # ----------------------------------------------------------------------------
        self.verbose = verbose
        self.visualize_step      = None
        self.init_epochs         = 0
        self.current_it_number   = 0

        self.list_loss = torch.tensor([]).cpu()
        self.list_psnr = torch.tensor([]).cpu()
        self.list_ssim = torch.tensor([]).cpu()

        self._initial_state_dict_saved = False
        
        self.log_model_every_n_steps = log_model_every_n_steps
        self.log_model_output_every_n_steps = log_model_output_every_n_steps

        # ----------------------------------------------------------------------------
        # Set the datamodule
        # ----------------------------------------------------------------------------
        self.datamodule = datamodule
        
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        
        self.count_batches = 0


    @property
    def logger(self) -> LoggerCollection:
        
        logger = super().logger
    
        if logger is None:
            logger = LoggerCollection([])
        
        if not isinstance(logger, LoggerCollection):
            logger = LoggerCollection([logger])

        return logger
    
    def get_targets(self, batch_idx : int) -> IntensityField:
        """ Returns the targets for a given batch_idx

        Args:
            batch_idx ([type]): [description]

        Returns:
            IntensityField: The IntensityField (or even a subclassed version of it. E.g. LightField or ImageView)
        """
        
        targets = next(itertools.islice(
                self.datamodule.train_dataloader(), batch_idx, None))
        
        if isinstance(targets,list):
            targets = targets[2]
        
        targets = IntensityField(
            data=targets,
            wavelengths = None)
        
        return targets

    def visualize_now(self) -> bool:
        """[summary]

        Returns:
            [type]: [description]
        """        
        visualize_now = False
        
        it_epoch = self.current_epoch
        if self.visualize_step is not None and (it_epoch+1) % self.visualize_step == 0:
            visualize_now = True
            
        return visualize_now

    def forward(self, no_grad : bool = False, *args, **kwargs) -> IntensityField:
        """ A forward method wrapper that allows to execute the forward model
        in  torch.no_grad() environment 

        Args:
            no_grad (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """    
        if no_grad:
            with torch.no_grad():
                out = self.model(*args, **kwargs)
        else:
            out = self.model(*args, **kwargs)
        return out

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.datamodule.train_dataloader

        
    def training_step(self,
                      batch : torch.Tensor,
                      batch_idx : int,  
                      ) -> torch.tensor:
        """ The training step is implemented by each CGH

        Args:
            batch (torch.Tensor): [description]
            batch_idx (int): [description]

        Returns:
            torch.tensor: [description]
        """
        
        raise NotImplementedError("Please implement your own training step.")
    
    def on_train_epoch_start(self) -> None:
        """ Takes care of visualization 

        Returns:
            [type]: [description]
        """        
        if self.verbose:
            # Increase the global iteration number (this was set in constructor)
            self.current_it_number = self.current_it_number + 1

            if self.visualize_now():
                self.visualize_epoch()



    def get_optimizers(self):
        
        optimizers = self.optimizers()
        
        if not type(optimizers) == list:
            optimizers = [optimizers]
            
        return optimizers
    
    def zero_grad_all_optimizers(self):
        # Manually zero grad the optimizers
        for opt in self.get_optimizers():
            opt.zero_grad()
    
    def step_all_optimizer(self):
        """Performs a Gradient Step for each optimizer of the Calibration Parameters
        """
        optimizers = self.get_optimizers()
        for opt in optimizers:
            opt.step()
            

        

    def log_image_batch(self,
                  batch : torch.Tensor,
                  tag : str = "",
                  ) -> None:

        if self.logger is None:
            return
        
        loggers = self.logger
        if isinstance(loggers, LightningLoggerBase):
            loggers = [loggers]
        
        for logger in loggers:
            try:
                logger.log_image(
                    tag = tag,
                    img_tensor = batch,
                    global_step = self.current_it_number,
                    is_batched = True,
                )
            except NotImplementedError:
                print("Image Saving not implemented.")
            except Exception:
                raise Exception


    def log_image(self,
                  batch : torch.Tensor,
                  tag : str = "",
                  ) -> None:

        if self.logger is None:
            return
        
        loggers = self.logger
        if isinstance(loggers, LightningLoggerBase):
            loggers = [loggers]
        
        for logger in loggers:
            try:
                logger.log_image(
                    tag = tag,
                    img_tensor = batch,
                    global_step = self.current_it_number,
                    is_batched = False
                )
            except NotImplementedError:
                print("Image Saving not implemented.")

    def log_full_model(self,
                flag_first : bool = False
                ) -> None:
        """ Logs the parameters of the model

        Raises:
            Exception: [description]
        """    
        
        if self.check_if_log_model_now() == False and flag_first == False:
            return
        
        if self.logger is None:
            return
        
        loggers = self.logger

        for logger in loggers:
            try:
                if flag_first:
                    cur_it = None
                else: 
                    cur_it = self.current_it_number
                    
                logger.log_full_model(
                    model = self.model,
                    current_it_number = cur_it
                    )
            except AttributeError:
                pass


    def log_state_dict(self,
                flag_first : bool = False
                ) -> None:
        """ Logs the parameters of the model

        Raises:
            Exception: [description]
        """    
        if self.logger is None:
            return
        
        loggers = self.logger

        for logger in loggers:
            try:
                if flag_first:
                    cur_it = None
                else: 
                    cur_it = self.current_it_number
                    
                logger.log_state_dict(
                    model = self.model,
                    current_it_number = cur_it
                    )
            except AttributeError:
                pass  

    def on_fit_start(self) -> None:
        
        if self._initial_state_dict_saved == False:
            self.log_state_dict(flag_first=True)
            self.log_full_model(flag_first=True)
            self._initial_state_dict_saved = True
        
    
    def on_fit_end(self):
        self.model.cuda()

    def log_light_object(self,
            object : Light,
            tag : str,
            extension = "tiff"
            ):


        for logger in self.logger:
            try:
                logger.log_light_object(
                    object  = object,
                    tag = tag,
                    extension = extension,
                    global_step = self.current_it_number
                )
            except AttributeError:
                pass  
            except Exception:
                raise Exception
            

    def log_scalar(self,
                name : str,
                value : float or torch.Tensor
                ):

        for logger in self.logger:
            try:
                logger.log_scalar(
                    tag = name,
                    scalar_value = value,
                    global_step = self.count_batches,
                )
            except Exception:
                raise Exception
    
    def training_epoch_end(self, outputs) -> None:



        self.log_full_model(flag_first=False)
        
    
    def on_train_batch_start(self, batch: Any, batch_idx: int, unused = 0) -> None:
        
        path = pathlib.Path("./.optimize/optimize.txt")
        # print(path.resolve())
        # path.parent.mkdir(exist_ok=True,parents=True)
        # import os

        optimize = True
        try:
            if (os.path.getsize(path) > 0):
                optimize = False
        except FileNotFoundError:
            pass

        if optimize == False:
            print("Leaving programm because optimize.txt is not empty.")
            open(path.resolve(), 'w').close()     
            raise KeyboardInterrupt("TEST")
    
    def on_train_batch_end(self,
                outputs,
                batch,
                batch_idx,
     )               -> None:

        
        self.count_batches = self.count_batches + 1
        
        self.log_val(name = 'loss', outputs = outputs)
        self.log_val(name = 'psnr', outputs = outputs)
        self.log_val(name = 'ssim', outputs = outputs)



    def log_val(self,
            name : str,
            outputs
            ):

        try:
            value = outputs[name].cpu()
        except KeyError:
            return

        if value.ndim == 0:
            value = torch.tensor([value]).cpu()

        attr_name = "list_" + name
        old_value = getattr(self,attr_name).cpu()

        setattr(self, attr_name, torch.cat((old_value, value),0))

        self.log_scalar(name = name + '/train', value = value)


    def do_logging_or_not(self) -> bool:
        if self.current_it_number % self.trainer.log_every_n_steps == 0:
            return True,
        else:
            return False

    def check_if_log_model_now(self) -> bool:
        if self.log_model_every_n_steps != None:
            if self.current_it_number % self.log_model_every_n_steps == 0:
                return True,
        else:
            return False


    def check_if_log_image_now(self) -> bool:
        if self.log_model_output_every_n_steps != None:
            if self.current_it_number % self.log_model_output_every_n_steps == 0:
                return True,
        else:
            return False

 
    
    def __str__(self):
        """
        TODO
        """
        
        rep = "holotorch.HoloLightning"

        rep += ""
        return rep

    def __repr__(self):
        """
        TODO
        """
        return self.__str__()

    def compute_loss(self,
                     output : IntensityField,
                     target : IntensityField
                     ) -> torch.Tensor:
        """ Computes the l2-loss

        Args:
            output ([type]): [description]
            target ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Note this works because we've overloaded the "-" operator for IntensityFields
        img_err = output - target
        loss = img_err.data.abs().pow(2).mean()
        
        return loss



    def add_optimizer(self, object : torch.Tensor or torch.nn.Module, learning_rate, lr_scheduler = None, lr_gamma = None):
        """Adds an optimizer

        Args:
            parameter (_type_): _description_
            learning_rate (_type_): _description_
            lr_scheduler (_type_): _description_
            lr_gamma (_type_): _description_
        """        
        
        if torch.is_tensor(object):
            parameter = object
        else:
            parameter = object.parameters()
        
        if lr_scheduler is None:
            lr_scheduler = self.lr_step
            
        if lr_gamma is None:
            lr_gamma = self.lr_gamma
        
        # Get the parameters of the physical system to optimize
        if torch.is_tensor(parameter):
            parameter = [parameter]
        else:
            parameter = list(parameter)
        
        # Only add the optimizer if there's an actual parameter to optimize
        if len(parameter) == 0:
            return
    
        # Create the optimizer (ADAM)
        citl_slm_glass_optimizer = torch.optim.Adam(
        # citl_slm_glass_optimizer = torch.optim.SGD(
                params  = parameter,
                lr      = learning_rate
                )

        # Add the schedulerer
        citl_slm_glass_scheduler = torch.optim.lr_scheduler.StepLR(
                citl_slm_glass_optimizer, 
                step_size=self.lr_step,
                gamma=self.lr_gamma
                )        

        self.opt_list.append({"optimizer": citl_slm_glass_optimizer, 
                            "lr_scheduler": citl_slm_glass_scheduler})


    def visualize_epoch(self, batch_idx=0):

        self.visualize_image_view(batch_idx=batch_idx)
        
        self.visualize_loss()
        # if self.param.OPTIMIZE_HOLO == True:
        #     self.model.holo_model.view_expander(
        #         save_id             = None,
        #         current_it_number   = self.current_epoch
        #     )

    def visualize_loss(self):

        fontsize = 17
        # visualize loss
        fig, ax = plt.subplots()
    
        plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

        mse = np.array(self.list_loss.cpu())
        psnr = np.array(self.list_psnr.cpu())


        # plot MSE/PSNR
        ax.plot(mse,color='blue')
        ax.set_ylabel('MSE Loss', fontsize=fontsize)
        ax.set_xlabel('Iteration #', fontsize=fontsize)

        ax = plt.gca()

        ax.yaxis.label.set_color('blue')        #setting up X-axis label color to yellow
        ax.tick_params(axis='y', colors='blue')  #setting up Y-axis tick color to black
        
        # Generate a new Axes instance, on the twin-X axes (same position)
        ax2 = ax.twinx()
        ax2.plot(psnr, color='red')
        ax2.set_ylabel('PSNR (dB)', fontsize=fontsize, color='red')
        ax2.spines['right'].set_color('red')  
        ax2.spines['left'].set_color('blue')  

        ax2.xaxis.label.set_color('red')        #setting up X-axis label color to yellow
        ax2.tick_params(axis='y', colors='red')  #setting up Y-axis tick color to black

        plt.show()

    @staticmethod
    def compute_psnr(
        target : IntensityField,
        input : IntensityField,
        max_value = 255.0,
    ):

        target : torch.Tensor = target.data
        input : torch.Tensor = input.data

        amax = target.amax(dim=[-2,-1])[...,None,None] 
        target = ( target / amax ) * max_value # 255 represents the bitlevel
        input  = ( input / amax ) * max_value 

        psnr = kornia.metrics.psnr(input = input, target = target, max_val = max_value)

        return psnr

    @staticmethod
    def compute_ssim(
        target : IntensityField,
        input : IntensityField,
    ):

        target = transform_6D_to_4D(target.data)
        input = transform_6D_to_4D(input.data)


        ssim = kornia.metrics.ssim(
            img1 = target,
            img2 = input,
            window_size = 3
        )

        return ssim

    def visualize_target(self, batch_idx = 0):

        if batch_idx == 0:
            targets = self.targets_first_batch
        else:
            targets = self.get_targets(batch_idx=batch_idx)
            
        return

    def visualize_image_view(self, batch_idx = 0, figsize = (3, 3)):

        targets = self.get_targets(batch_idx=batch_idx)

        with torch.no_grad(): 
            outputs = self.model(batch_idx)

        outputs = ImageView(data=outputs.data, wavelengths=targets.wavelengths)

        ImageView.visualize_difference(
            outputs=outputs, targets=targets, figsize= figsize )

    def print_state_dict_nice(self):
        state_dict = self.state_dict()

        for var_name  in enumerate(state_dict):
            data = state_dict[var_name[1]]
            print(var_name, data.shape, data.device)
            
    def print_param_nice(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
