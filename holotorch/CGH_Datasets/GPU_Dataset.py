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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from holotorch.CGH_Datasets.CGH_Dataset import CGH_Dataset

class GPU_Dataset(CGH_Dataset):

    def __init__(self,
            dataset : CGH_Dataset
                 ):
        super().__init__()
        
        self.dataset = dataset
        
   
    def pre_load_dataset(self,
                device = None
                         ):
        '''
        save transformed data in tensor format
        '''
        
        self.gpu_preloaded_dataset = []
        
        for _, tmp in enumerate(self.dataset):

            if not ( isinstance(tmp,tuple) or  isinstance(tmp,list)) :
                tmp = [tmp]
            
            tmp2 = []
            # Sometimes datasets returns tuples, so we need to move every element in this tuple to the GPU            
            for foo in tmp:
                foo = foo.to(device)
                tmp2.append(foo)
            
            # Let's remove the tuple option if we have only one element
            if len(tmp2) == 1:
                tmp2 = tmp2[0]

            self.gpu_preloaded_dataset.append(tmp2)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        if idx >= len(self):
            raise IndexError
        
        try: 
            tmp = self.gpu_preloaded_dataset[idx]
        except IndexError:
            raise  # do not handle this error
        except:
            pass
        
        return tmp
