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

import numpy as np
import torch

# Check if imageJ is already initialized
try:
    ij
except NameError: ij = None
    
if ij is None:
    try:
        import imagej
    except:
        pass
    #ij = imagej.init('sc.fiji:fiji',headless=False)
    #ij = imagej.init(headless=False)
else:
    print('ImageJ already initiated')

def initialize_ij():
    global ij
    if ij is None:
        ij = imagej.init('sc.fiji:fiji', mode='interactive')
        ij.ui().showUI()

def show(img,title='ImageJ'):
    
    if ij is None:
        initialize_ij()
    
    if torch.is_tensor(img):
        img = img.detach().squeeze().cpu().numpy()
    
    # be careful not to check for np.array but for np.ndarray!
    if type(img) is np.ndarray:
        img = ij.py.to_java(img);
    ij.ui().show(title, img);

__macro_close_all_windows = """
// "Close All Windows"
// This macro closes all image windows.
// Add it to the StartupMacros file to create
// a "Close All Windows" command, or drop it
// in the ImageJ/plugins/Macros folder.
// Note that some ImageJ 1.37 has a bug that
// causes this macro to run very slowly.

  macro "Close All Windows" { 
      while (nImages>0) {
          selectImage(nImages);
          close(); 
      } 
  } 
"""
def close_all_windows():
    ij.py.run_macro(__macro_close_all_windows);


