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

from pytorch_lightning.loggers import TensorBoardLogger

from typing import Optional, Union
from torch.utils.tensorboard import SummaryWriter
import torch


class CustomTensorBoardLogger(TensorBoardLogger):
    
    def __init__(self, save_dir: str, name: Optional[str] = "default", version: Optional[Union[int, str]] = None, log_graph: bool = False, default_hp_metric: bool = True, prefix: str = "", sub_dir: Optional[str] = None, **kwargs):
        
        
        super().__init__(save_dir, name=name, version=version, log_graph=log_graph, default_hp_metric=default_hp_metric, prefix=prefix, sub_dir=sub_dir, **kwargs)
        
        
        import shutil
        shutil.rmtree(self.log_dir, ignore_errors=True)
       
    def get_summary_writer(self) -> SummaryWriter:
        return self.experiment
    
    def log_scalar(self,
            tag,
            scalar_value,
            global_step : int = None,
        ):
        writer = self.get_summary_writer()
        
        writer.add_scalar(
            tag = tag,
            scalar_value=scalar_value,
            global_step=global_step
        )
    

    
    def log_image(self,
                tag : str,
                img_tensor : torch.Tensor,
                global_step = None,
                dataformats = 'HW',
                walltime = None,
                is_batched : bool = False
                ):
        
        if is_batched:
            self.log_image_batch(
                tag = tag,
                img_tensor = img_tensor,
                global_step = global_step,
                walltime=walltime,
                dataformats = dataformats
            )
            return
        if img_tensor.ndim == 2:
            dataformats = 'HW'
        
        img_tensor = img_tensor / img_tensor.max()
        
        summary_writer = self.get_summary_writer()
        
        summary_writer.add_image(
            tag = tag,
            img_tensor = img_tensor,
            global_step = global_step,
            walltime=walltime,
            dataformats = dataformats)

    def log_image_batch(self,
                tag : str,
                img_tensor : torch.Tensor,
                global_step = None,
                dataformats = 'HW',
                walltime = None,
                ):
        summary_writer = self.get_summary_writer()
        
        if img_tensor.ndim == 3:
            dataformats = 'NHW'
            img_tensor = img_tensor[:,None]

        dataformats = 'NCHW'
            
        summary_writer.add_images(
            tag = tag,
            img_tensor = img_tensor,
            global_step = global_step,
            walltime=walltime,
            dataformats = dataformats
            )
        
        #print("DONE")
        

        

        
