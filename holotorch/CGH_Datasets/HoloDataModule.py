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

from typing import Tuple
import torch
import torch.utils
import torch.utils.data
import numpy as np
import itertools
from torch.utils.data import DataLoader, Dataset

# Import Torch Lightning
import pytorch_lightning as pl

# Import of HoloTorch Objects
from holotorch.CGH_Datatypes.IntensityField import IntensityField
from holotorch.CGH_Datasets.GPU_Dataset import GPU_Dataset
from holotorch.CGH_Datasets.CGH_Dataset import CGH_Dataset
import holotorch.utils.pjji as piji


class HoloDataModule(pl.LightningDataModule):
    def __init__(self,
            loader          : DataLoader = None, 
            dataset         : Dataset = None,
            train_transforms        = None,
            val_transforms          = None,
            test_transforms         = None,
            dims                    = None,
            batch_size              = 1,
            number_batches          = None,
            shuffle_for_init        = False,
            index_list              = None,
            shuffle                 = True,
            num_workers             = 0,
            pin_memory              = False,
                ):
        
        
        super().__init__(
            )
        

        
        if number_batches is not None:
            
            if ( len(dataset) // batch_size ) < number_batches:
                number_batches = None
            else:         
                if index_list is None:
                    data_sz = batch_size * number_batches
                    if shuffle_for_init:
                        index_list = np.random.choice(len(dataset), data_sz)
                    else:
                        index_list = np.arange(data_sz)

                dataset = torch.utils.data.Subset(dataset, index_list)

        if loader == None:
            assert dataset != None
            
            if shuffle is True:
                _tmp = torch.zeros(1)
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
                generator = torch.Generator(device = _tmp.device)
                generator.manual_seed(seed)
            else:
                generator = None
            # create a dataloader with the correct batch size
            loader = torch.utils.data.DataLoader(dataset,
                                drop_last=True,
                                shuffle=shuffle,
                                batch_size=batch_size,
                                num_workers = num_workers,
                                pin_memory= pin_memory,
                                generator=generator
                                )
    
            self.dataset = dataset

        # Save these for later if loader needs to be reinitalized
        self.shuffle=shuffle
        self.num_workers = num_workers
        self.pin_memory= pin_memory
        self.generator=generator



        self.loader = loader

    def preload_dataset(self,
                device
                    ):
        
        self.old_dataset = self.dataset
        
        self.dataset = GPU_Dataset(dataset=self.dataset)

        self.dataset.pre_load_dataset(device=device)

        self.loader = torch.utils.data.DataLoader(self.dataset,
                                drop_last   = True,
                                shuffle     = self.shuffle,
                                batch_size  = self.batch_size,
                                num_workers = self.num_workers,
                                pin_memory  = self.pin_memory,
                                generator   = self.generator
                                )
    
    def __len__(self):
        return len(self.dataset)

    
    @property
    def dataset(self) -> CGH_Dataset:
        return self._dataset
    
    @dataset.setter
    def dataset(self, dataset : CGH_Dataset):
        self._dataset = dataset

    def __getitem__(self, idx) -> IntensityField:
        return self.get_batch(idx)


    def __str__(self) -> str:
        
        
        tmp = self.__class__.__bases__[0].__name__
        mystr = "=======================================================\n"
        mystr += "CLASS Name: " + type(self).__name__ + " (extended from " + tmp + ")"
        mystr += "\n-------------------------------------------------------------\n"

        
        mystr += "Number of batches = " + str(self.number_batches)
        mystr += "\n" + "Number of images / batch = " + str(self.batch_size)
        return mystr
    
    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def batch_size(self) -> int:
        return self.train_dataloader().batch_size
    
    @property
    def number_batches(self):
        """Returns the number of batches in the dataloader for training

        Returns:
            _type_: _description_
        """        
        return len(self.train_dataloader())
    
    def setup(self, stage = None) -> None:
        #self.loader.dataset.pre_load_dataset()
        #return super().setup(stage=stage)
        return

    def train_dataloader(self) -> DataLoader:
        return self.loader
    
    def set_image(self, image : torch.Tensor = None, path : torch.Tensor = None):
        assert (image is None and path is None) == False
        self.loader.dataset.set_image(image = image, path = path)

    def get_batch(self, batch_idx) -> Tuple:
        batch = next(itertools.islice(
            self.train_dataloader(), batch_idx, None))
        return batch

    def get_batch_IF_single(self, batch_idx,
                     sub_index = None,
                     ) -> IntensityField:
        batch = self.get_batch(batch_idx)
        targets = batch
        
        targets = IntensityField(targets)
        return targets


    def get_batch_IF(self, batch_idx,
                     sub_index = None,
                     ) -> IntensityField:
        batch = self.get_batch(batch_idx)

        if torch.is_tensor(batch):
            targets = batch
            measured = None
            pattern = None
        elif len(batch) == 3:
            measured, pattern, targets = batch
        elif len(batch) == 2:
            measured, pattern = batch
            targets = None
        else:
            raise ValueError("This combination cannot exist.")
        
        outs = []


        if measured is not None:
            measured = IntensityField(measured)
            if sub_index is not None:
                measured = measured[sub_index]
        
            outs.append(measured)

        if pattern is not None:
            pattern = IntensityField(pattern)
            if sub_index is not None:
                pattern = pattern[sub_index]

            outs.append(pattern)

        if targets is not None:

            targets = IntensityField(targets)
            if sub_index is not None:
                targets = targets[sub_index]
            
            outs.append(targets)
        
        if len(outs) == 1:
            outs = outs[0]
            
        return outs #measured, pattern, targets

    def show_piji(self,
                    batch_idxs : int or list = [0],
                    model = None
    ):
        if type(batch_idxs) != list:
            batch_idxs = [batch_idxs]
        n_batch = len(batch_idxs)

        outputs         = []
        targets         = [] 
        measureds       = []
        voltages        = [] 
        for b_idx in batch_idxs:
            # first assume there are targets
            try:
                measured, voltage, target = self.get_batch(batch_idx=b_idx)
                targets.append(target.squeeze().cpu())
            except ValueError:
                measured, voltage = self.get_batch(batch_idx=b_idx)
            if model != None:
                voltage = voltage[:,:,0,:,:,:] / 255.0 # remove pupil dimension and normalize to [0,1]
                out = model.forward(voltages=voltage)
                outputs.append(out.data.detach().squeeze().cpu())
            measureds.append(measured.squeeze().cpu())
            voltages.append(voltage.squeeze().cpu())


        B,_,_,_,H,W = measured.shape
        measureds   = torch.stack(measureds).reshape(n_batch*B,H,W)
        if model != None:
            outputs     = torch.stack(outputs).reshape(n_batch*B,H,W)
            errors      = (outputs - measureds).abs()
            piji.show(outputs,  title="Model")
            piji.show(errors,  title="Model - Measured")

        H,W = voltage.shape[-2:]
        voltages    = torch.stack(voltages).reshape(n_batch*B,H,W)      

        piji.show(voltages,  title="Voltages")
        piji.show(measureds,  title="Measured")

        if len(targets) > 0:
            B,_,_,_,H,W = measured.shape
            targets     = torch.stack(targets).reshape(n_batch*B,H,W)
            piji.show(targets,  title="Target")

        
