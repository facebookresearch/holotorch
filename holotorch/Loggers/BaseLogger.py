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

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.cloud_io import get_filesystem

from holotorch.Optical_Setups.Base_Setup import Base_Setup
from holotorch.CGH_Datatypes.Light import Light

from typing import Optional, Union
import os

import torch
import pathlib

import shutil

from holotorch.utils.ImageSave import imsave
from holotorch.utils.string_processor import convert_integer_into_string


class BaseLogger(LightningLoggerBase):
    
    def __init__(self,
        save_dir: str,
        name: Optional[str] = "default",
        version: Optional[Union[int, str]] = None,
        sub_dir: Optional[str] = None,
        **kwargs
        ):
        super().__init__()
        
        self._save_dir = save_dir
        self._name = name
        self._experiment = None
        self._version = version
        self._sub_dir = sub_dir

        self._fs = get_filesystem(save_dir)
        
        self._experiment = None
        self.hparams = {}
        self._kwargs = kwargs

        shutil.rmtree(self.log_dir, ignore_errors=True)

        # Create the directory
        self.create_dir()

    def create_dir(self):
        if self.root_dir:
            self._fs.makedirs(self.root_dir, exist_ok=True)
            
            self._fs.makedirs(self.log_dir, exist_ok=True)
            #self._fs = get_filesystem(self.log_dir).makedirs() exist_ok = True)

    @property
    def log_dir(self) -> str:
        """The directory for this run's tensorboard checkpoint.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.
        """
        # create a pseudo standard path ala test-tube
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir
    
    @property
    def sub_dir(self) -> Optional[str]:
        """Gets the sub directory where the TensorBoard experiments are saved.

        Returns:
            The local path to the sub directory where the TensorBoard experiments are saved.
        """
        return self._sub_dir
    
    @property
    def root_dir(self) -> str:
        """Parent directory for all tensorboard checkpoint subdirectories.

        If the experiment name parameter is ``None`` or the empty string, no experiment subdirectory is used and the
        checkpoint will be saved in "save_dir/version_dir"
        """
        if self.name is None or len(self.name) == 0:
            return self.save_dir
        return os.path.join(self.save_dir, self.name)


    @property
    def save_dir(self) -> Optional[str]:
        """Gets the save directory where the TensorBoard experiments are saved.

        Returns:
            The local path to the save directory where the TensorBoard experiments are saved.
        """
        return self._save_dir

    @property
    def name(self) -> str:
        """Get the name of the experiment.

        Returns:
            The name of the experiment.
        """
        return self._name
    


    @property
    def experiment(self) -> None:
        """Return the experiment object associated with this logger."""
        return None
    
    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here

        print(metrics)
                
    
    def log_light_object(self,
            object : Light,
            tag : str,
            extension = "tiff",
            global_step : int = None
                         ):
        
        folder = pathlib.Path(self.log_dir)
        folder = folder / "images"
        
        filename = tag 
        
        if global_step is not None:
            filename = filename + "_" + convert_integer_into_string(global_step, depth = 5)
        
        object.save(
            filename    = filename,
            folder      = folder,
            extension   = extension
        )
    
    def log_full_model(self,
            model : Base_Setup,
            current_it_number : int = None,
                ):
                        
        fp = pathlib.Path(self.log_dir)
        fp = fp / "model"
        fp.mkdir(exist_ok=True, parents=True)
        
        filename = "model"
        if current_it_number is not None:
             filename +=("_it_" + convert_integer_into_string(current_it_number, depth = 5))
        
        filename += ".pickle"
        
        filename = fp / filename
        
        model.save_model(filename=filename)

    def log(self):
        
        print("DO NOTHING")
    
    def log_state_dict(self,
            model : Base_Setup,
            current_it_number : int = None,
                ):
                        
        fp = pathlib.Path(self.log_dir)
        fp = fp / "model"
        fp = fp / "state_dict" 
        if current_it_number is not None:
            fp / ("it_" + convert_integer_into_string(current_it_number, depth = 5))
        
        fp.mkdir(exist_ok=True, parents=True)
        
        state_dict = model.state_dict()

        for var_name  in enumerate(state_dict):
            # print(var_name )
            data = state_dict[var_name[1]]

            path = fp / (var_name[1] + ".pt")
            torch.save(data, path)
    
    def log_param(self,
                  model : Base_Setup,
                  current_it_number : int,
            ):

        fp = pathlib.Path(self.log_dir)
        fp = fp / "model"
        fp = fp / "parameters" / ("it_" + convert_integer_into_string(current_it_number, depth = 5))
        
        fp.mkdir(exist_ok=True, parents=True)
        
        for name, param in model.named_parameters():
            path = fp / (name + ".pt")
            torch.save(param, path)
            
    def log_scalar(self,
            tag,
            scalar_value,
            global_step : int =None,
        ):
        pass

    def log_image(self,
                img_tensor : torch.Tensor,
                tag : str = None,
                global_step = None,
                dataformats = 'HW',
                walltime = None,
                is_batched : bool = False,
                extension = "tiff"
                  ) -> None:
        """Log image.

        Arguments are directly passed to the logger.
        """
        data = img_tensor.data.squeeze()
        if data.ndim == 3:
            data = data[0]
        
        # Create the folder where images will be stored
        folder = pathlib.Path(self.log_dir) / "images"
        
        if tag is None:
            tag = ""
        else:
            tag = tag + "_"
            
        filename_tiff = pathlib.Path(tag + convert_integer_into_string(global_step, depth = 5))
       
        imsave(filename=filename_tiff, data = data, folder=folder, extension=extension)

    @property
    def version(self) -> int:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = self.root_dir

        try:
            listdir_info = self._fs.listdir(root_dir)
        except OSError:
            print("Missing root dir")
            return 0

        existing_versions = []
        for listing in listdir_info:
            d = listing["name"]
            bn = os.path.basename(d)
            if self._fs.isdir(d) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace("/", "")
                existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
    
    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass