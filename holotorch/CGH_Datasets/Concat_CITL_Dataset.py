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
import pathlib
from torch.utils.data import ConcatDataset

import holotorch.utils.pjji as piji
from holotorch.CGH_Datasets.Captured_Image_Dataset import Captured_Image_Dataset
from holotorch.CGH_Datasets.CITL_Dataset import CITL_Dataset
from holotorch.CGH_Datasets.CGH_Dataset import CGH_Dataset
from holotorch.Optical_Setups.Base_Setup import Base_Setup

def create_Concat_Dataset_from_folders(
                folder_list = list,
                normalize_flag = True,
                data_sz = None,
                prefix = ""
                ):
    """Creates a concat dataset from a list of folders

    Args:
        folder_list (_type_, optional): _description_. Defaults to list.

    Returns:
        _type_: _description_
    """    
    list_dataset = []
    for folder in folder_list:
        #sub_folder = 'dpac_linearfringes'
        dataset = Captured_Image_Dataset(
            img_dir         = folder,
            normalize_flag  = normalize_flag,
            data_sz         = data_sz,
            prefix          = prefix
        )
        list_dataset.append(dataset)
        
    concat_dataset = ConcatDataset(list_dataset)
    return concat_dataset

def create_folder_list(
            folder_list : list,
            base_folder : pathlib.Path,
            suffix = "measured"
            ):
    """Creates a list of folders all living in the same base_folder

    Args:
        folder_list (list): _description_
        base_folder (pathlib.Path): _description_
        suffix (str, optional): _description_. Defaults to "measured".

    Returns:
        _type_: _description_
    """    
    new_folder_list = []
    for folder in folder_list:
        new_folder = base_folder / folder / suffix
        new_folder_list.append(new_folder)
    return new_folder_list

def create_concat_dataset(
            folder_list : list,
            base_folder : pathlib.Path,
            suffix = "measured",
            normalize_flag : bool = True,
            data_sz = None,
            prefix = ""
            ):
    """Creates a concatenated dataset from a list of folders all living in the base_folder

    Args:
        folder_list (list): _description_
        base_folder (pathlib.Path): _description_
        suffix (str, optional): _description_. Defaults to "measured".

    Returns:
        _type_: _description_
    """    
    
    if suffix is None:
        return None
    
    new_folder_list = create_folder_list(
        folder_list = folder_list,
        base_folder = base_folder,
        suffix      = suffix,
    )
    concat_dataset = create_Concat_Dataset_from_folders(
        folder_list=new_folder_list,
        normalize_flag = normalize_flag,
        data_sz=data_sz,
        prefix=prefix
    )
    
    return concat_dataset

def create_concat_citl_dataset(
        folder_list : list,
        base_folder : pathlib.Path,
        suffix_pattern  = 'pattern',
        suffix_target   = 'target',
        suffix_measured = 'measured',
        prefix = ""
    ):
    
    pattern_dataset = create_concat_dataset(
            folder_list=folder_list,
            base_folder=base_folder,
            suffix=suffix_pattern,
            normalize_flag = False,
            prefix=prefix
            )   
    
    target_dataset = create_concat_dataset(
            folder_list=folder_list,
            base_folder=base_folder,
            suffix=suffix_target,
            normalize_flag = True,
            prefix=prefix
            )   
    
    measured_dataset = create_concat_dataset(
            folder_list=folder_list,
            base_folder=base_folder,
            suffix=suffix_measured,
            normalize_flag = True,
            prefix=prefix
            )   
    
    concat_citl = CITL_Dataset(
            dataset_measured = measured_dataset,
            dataset_pattern  = pattern_dataset,
            dataset_target   = target_dataset
    )
    
    return concat_citl

class Concat_CITL_Dataset(CITL_Dataset):
    
    def __init__(self,
            dataset_measured : CGH_Dataset,
            dataset_pattern : CGH_Dataset,
            dataset_target : CGH_Dataset
            ):
        super().__init__(
            dataset_measured = dataset_measured,
            dataset_pattern = dataset_pattern,
            dataset_target = dataset_target
        )

    def show_piji(self,
            N_imgs : int = 1,
            model : Base_Setup = None
            ):
        target_list = []
        measured_list = []
        voltage_list = []
        outputs_list = []

        for batch_idx in range(N_imgs):
            measured, voltage, target = self[batch_idx]
            voltage = voltage / 255.0
            
            if model is not None:
                out = model.forward(voltages=voltage)
                outputs_list.append(out.data.detach().squeeze().cpu())
                
            target_list.append(target.data.detach().squeeze().cpu())
            voltage_list.append(voltage.data.detach().squeeze().cpu())
            measured_list.append(measured.data.detach().squeeze().cpu())

        targets = torch.stack(target_list)
        voltages = torch.stack(voltage_list)
        measureds = torch.stack(measured_list)
        
        piji.show(voltages,  title="Voltages")
        piji.show(targets,  title="Targets")
        piji.show(measureds,  title="Measured")

        if model is not None:
            outputs = torch.stack(outputs_list)
            piji.show(outputs, title="Model Output")

    
    def pre_load_dataset(self,
            load_on_gpu : bool =True
        ):
        
        self.dataset_measured.datasets[0].pre_load_dataset(load_on_gpu=True)
        self.dataset_pattern.datasets[0].pre_load_dataset(load_on_gpu=True)
        self.dataset_target.datasets[0].pre_load_dataset(load_on_gpu=True)
        
    
    @classmethod
    def create_concat_citl_dataset(cls,
            folder_list : list,
            base_folder : pathlib.Path,
            suffix_pattern  = 'pattern',
            suffix_target   = 'target',
            suffix_measured = 'measured',
            data_sz         = None,
            prefix = ""
        ):
        
        pattern_dataset = create_concat_dataset(
                folder_list=folder_list,
                base_folder=base_folder,
                suffix=suffix_pattern,
                normalize_flag=False,
                data_sz=data_sz,
                prefix=prefix
                )   
        
        target_dataset = create_concat_dataset(
                folder_list=folder_list,
                base_folder=base_folder,
                suffix=suffix_target,
                normalize_flag=True,
                data_sz=data_sz,
                prefix=prefix
                )   
        
        measured_dataset = create_concat_dataset(
                folder_list=folder_list,
                base_folder=base_folder,
                suffix=suffix_measured,
                normalize_flag=True,
                data_sz=data_sz,
                prefix=prefix
                )   
        
        concat_citl = Concat_CITL_Dataset(
                dataset_measured = measured_dataset,
                dataset_pattern  = pattern_dataset,
                dataset_target   = target_dataset
        )
        
        return concat_citl
    

    def __len__(self):
        return super().__len__()

