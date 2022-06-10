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
import numpy as np

def tensor_list_to_tiff(
    list_data: list, 
    filename : str
    ):
    tiff_data = torch.cat([img[None,:,:] for img in list_data], dim=0)
    tiff_data = tiff_data.cpu().squeeze().numpy()
    imsave(tiff_data, filename)

def imsave(
        data: np.ndarray or torch.Tensor,
        filename: str or pathlib.Path,
        folder : str = None,
        extension : str = None
        ):
    """Custom implementation of imsave to avoid skimage dependency.

    Parameters
    ----------
    filename : string
        The path to write the file to.
    data : np.ndarray
        The image data.
    """
    import os
    
    filename, ext = os.path.splitext(filename)
    if not ext:
        filename = filename + '.' + extension
    else:
        filename = filename + ext

    #print(path)
    
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    if isinstance(filename,str):
        filename = pathlib.Path(filename)
        
    if folder is not None:
        filename = folder / filename
    
    filename.parent.mkdir(exist_ok=True, parents = True)
    
   
    ext = os.path.splitext(filename)[1]
    if ext in [".tif", ".tiff"]:
        import tifffile

        tifffile.imsave(filename.resolve(), data.astype(np.float32) , imagej=True)
    else:
        import imageio
        imageio.imsave(filename.resolve(), (data*255).astype(dtype=np.uint8)) 