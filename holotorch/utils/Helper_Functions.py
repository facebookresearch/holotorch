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



from __future__ import print_function
import torch
import warnings
from typing import Union

import holotorch.utils.transformer_6d_4d as transformer_6d_4d

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

def replace_bkwd(fwd: torch.Tensor, bkwd: torch.Tensor):
    new         = bkwd.clone()  # contains backwardFn from bkwd
    new.data    = fwd.data      # copies data from fwd  
    return new

def replace_fwd(fwd : torch.Tensor, bkwd : torch.Tensor): 
    bkwd.data = fwd; 
    return bkwd

def set_default_device(device: Union[str, torch.device]):
    if not isinstance(device, torch.device):
        device = torch.device(device)
        
    print(device)

    if device.type == 'cuda':
        print("TEST")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.set_device(device.index)
        print("CUDA1")
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        print("CUDA2")

def total_variation(input: torch.tensor):
    '''
    compute centerered finite difference derivative for input along dimensions dim
    zero pad the boundary

    input: 4D torch tensor with dimension 2,3 being the spatial difference
    returns: scalar value |dx|_1 + |dy|_1  
    '''
    # reshape if its a 6D tensor
    if input.ndim == 6:
        B,T,P,C,H,W = input.shape
        input = input.view(B*T*P,C,H,W)

    dx, dy = center_difference(input)
    return dx.abs().mean() + dy.abs().mean()

def center_difference(input: torch.tensor):
    '''
    compute centerered finite difference derivative for input along dimensions dim
    zero pad the boundary

    input: 4D torch tensor with dimension 2,3 being the spatial difference
    returns: dx, dy - 4D tensors same size as input 
    '''
    # create a new tensor of zeros for zeropadding
    dx = torch.zeros_like(input)
    dy = torch.zeros_like(input)
    _, _, H, W = input.shape
    dx[:,:,:,1:-1] = W/4*(-input[:,:,:,0:-2] + 2*input[:,:,:,1:-1] - input[:,:,:,2:])
    dy[:,:,1:-1,:] = H/4*(-input[:,:,0:-2,:] + 2*input[:,:,1:-1,:] - input[:,:,2:,:])
    return dx, dy

def tt(x):
    return torch.tensor(x)

def regular_grid4D(M,N,H,W, range=tt([[-1,1],[-1,1],[-1,1],[-1,1]]), device=torch.device("cpu")):
    '''
    Create a regular grid 4D tensor with dims M x N x H x W specified within a range 
    '''
    #Coordinates                 
    x = torch.linspace(range[0,0], range[0,1], M, device=device)  
    y = torch.linspace(range[1,0], range[1,1], N, device=device)  
    u = torch.linspace(range[2,0], range[2,1], H, device=device)  
    v = torch.linspace(range[3,0], range[3,1], W, device=device)  

    #Generate Coordinate Mesh and store it in the model
    return torch.meshgrid(x,y,u,v)    

def regular_grid2D(H,W, range=tt([[-1,1],[-1,1]]), device=torch.device("cpu")):
    '''
    Create a regular grid 2D tensor with dims H x W specified within a range 
    '''
    #XCoordinates                 
    x_c = torch.linspace(range[0,0], range[0,1], W, device=device)  
    #YCoordinates 
    y_c = torch.linspace(range[1,0], range[1,1], H, device=device)  
    #Generate Coordinate Mesh and store it in the model
    return torch.meshgrid(x_c,y_c)    

def ft2(input, delta=1, norm = 'ortho', pad = False):
    """
    Helper function computes a shifted fourier transform with optional scaling
    """
    return perform_ft(
        input=input,
        delta = delta,
        norm = norm,
        pad = pad,
        flag_ifft=False
    )

def ift2(input, delta=1, norm = 'ortho', pad = False):
    
    return perform_ft(
        input=input,
        delta = delta,
        norm = norm,
        pad = pad,
        flag_ifft=True
    )

def perform_ft(input, delta=1, norm = 'ortho', pad = False, flag_ifft : bool = False):
    
    # Get the initial shape (used later for transforming 6D to 4D)
    tmp_shape = input.shape

    # Save Size for later crop
    Nx_old = int(input.shape[-2])
    Ny_old = int(input.shape[-1])
        
    # Pad the image for avoiding convolution artifacts
    if pad == True:
        
        pad_scale = 1
        
        pad_nx = int(pad_scale * Nx_old / 2)
        pad_ny = int(pad_scale * Ny_old / 2)
        
        input = torch.nn.functional.pad(input, (pad_nx,pad_nx,pad_ny,pad_ny), mode='constant', value=0)
    
    if flag_ifft == False:
        myfft = torch.fft.fft2
        my_fftshift = torch.fft.fftshift
    else:
        myfft = torch.fft.ifft2
        my_fftshift = torch.fft.ifftshift


    
    # Compute the Fourier Transform
    out = (delta**2)* my_fftshift( myfft (  my_fftshift (input, dim=(-2,-1))  , dim=(-2,-1), norm=norm)  , dim=(-2,-1))
    
    if pad == True:
        input_size = [Nx_old, Ny_old]
        pool = torch.nn.AdaptiveAvgPool2d(input_size)
        out = transformer_6d_4d.transform_6D_to_4D(tensor_in=out)
        
        if out.is_complex():
            out = pool(out.real) + 1j * pool(out.imag)
        else:
            out = pool(out)
        new_shape = (*tmp_shape[:4],*out.shape[-2:])

        out = transformer_6d_4d.transform_4D_to_6D(tensor_in=out, newshape=new_shape)

    return out
        




