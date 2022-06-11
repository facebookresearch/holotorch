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


import matplotlib.pyplot as plt
import numpy as np
import pylab as plt

import torch

import holotorch.utils.pytorch_colors.pytorch_colors as colors

from IPython.display import display, Image
import imageio as io
from holotorch.utils.Helper_Functions import tt
import torch.nn.functional as F

from holotorch.utils.units import *

def float_to_unit_identifier(val):
    """
    Takes a float value (e.g. 5*mm) and identifies which range it is
    e.g. mm , m, um etc.

    We always round up to the next 1000er decimal

    e.g.
    - 55mm will return mm
    - 100*m will return m
    - 0.1*mm will return um
    """
    exponent = np.floor(np.log10( val) / 3)
    unit_val = 10**(3*exponent)

    if unit_val == m:
        unit = "m"
    elif unit_val == mm:
        unit = "mm"
    elif unit_val == um:
        unit = "um"
    elif unit_val == nm:
        unit = "nm"
    return unit_val, unit

def view_output_as_gif(out, vmax=1, gamma=1, size=(512,512)):
    '''
    View an image tensor with 3 non-empty dims as a gif
    tensor can be real or complex
    '''
    # save a gif of the image stack and display
    name = 'tmp.gif'
    # rf = RF(size=size, device = out.device, mode="bicubic")
    # out = rf(out)

    # force 4D output
    if out.ndim == 2:
        H,W = out.shape
        N = 1
        C = 1
        out = out.view(1,1,H,W)
    if out.ndim == 3:
        N,H,W = out.shape
        C = 1
        out = out.view(N,1,H,W)
    elif out.ndim == 4:
        N,C,H,W = out.shape
    elif out.ndim == 5:
        B,N,C,H,W = out.shape
        out = out[0,:,:,:,:]
    elif out.ndim == 6:
        B,T,N,C,H,W = out.shape
        out = out[0,0,:,:,:,:]

    if out.is_complex():
        out = F.interpolate(out.real, size=size, mode="bicubic") + \
                1j*F.interpolate(out.imag, size=size, mode="bicubic") 
    else:
        out = F.interpolate(out, size=size, mode="bicubic")
    _,_,H,W = out.shape

    out = (out/out.abs().max()).pow(gamma)
    out = out.cpu().detach()
    
    if out.is_complex():
        C = 3
        new_out = torch.zeros((N,H,W,3))
        for n in range(N):
            new_out[n,:,:,:] = tt(colorize(out[n,0,:,:])) # just plot first color channel
        out = new_out
    else:
        # need to permute color channel dimensions
        out = out.permute(0,2,3,1) # N x H x W x C

    # normalize and convert to 8 bit
    out = (255/vmax*out)
    out = out.type(torch.uint8).cpu().detach()

    # need to create at least two frames
    if N == 1:
        out = out.squeeze().view(1,H,W,C).expand(2,H,W,C)
    else:
        out = out.squeeze().reshape(N,H,W,C)

    # NOTE: 'GIF-FI' format requires imagefreelib
    # can be downloaded in python with: imageio.plugins.freeimage.download()
    # or just replace the format from "GIF-FI" to default (GIF-PIL)
    # io.mimsave(name, out, fps=2, )

    io.mimsave(name, out, 'GIF-FI', fps=2, )
    with open(name,'rb') as file:
        display(Image(file.read()))

def view_output(out, vmax=1, gamma=1, size=(512,512)):
    '''
    View an image tensor with 3 non-empty dims as a grid of images
    tensor can be real or complex
    '''
    # rf = RF(size=size, device = out.device, mode="bicubic")
    # out = rf(out)

    # if out.is_complex():
    #     out = F.interpolate(out.real, size=size, mode="bicubic") + \
    #             1j*F.interpolate(out.imag, size=size, mode="bicubic") 
    # else:
    #     out = F.interpolate(out, size=size, mode="bicubic")

    # force 4D output
    if out.ndim == 2:
        H,W = out.shape
        N = 1
        out.view(1,1,H,W)
    if out.ndim == 3:
        N,H,W = out.shape
    elif out.ndim == 4:
        N,C,H,W = out.shape
    elif out.ndim == 5:
        B,N,C,H,W = out.shape
        out = out[0,:,:,:,:]
    elif out.ndim == 6:
        B,T,N,C,H,W = out.shape
        out = out[0,0,:,:,:,:]

    if out.is_complex():
        out = F.interpolate(out.real, size=size, mode="bicubic") + \
                1j*F.interpolate(out.imag, size=size, mode="bicubic") 
    else:
        out = F.interpolate(out, size=size, mode="bicubic")
    _,_,H,W = out.shape

    # need to permute color channel dimensions
    out = out.permute(0,2,3,1) # N x H x W x C

    out = out.detach().cpu()
    out = (out/out.abs().max()).pow(gamma)
    out = (out/vmax)
    ND = int(np.ceil(np.sqrt(N)))
    plt.figure(figsize=(20,20))
    for i in range(N):
        ax = plt.subplot2grid((ND, ND), (i//ND, i%ND))
        tmp_im = out[i].cpu().detach()
        if tmp_im.is_complex():
            tmp_im = colorize(tmp_im[0,:,:]) # shouldn't be a color channel now
        p = plt.imshow(tmp_im.squeeze(), cmap='gray', vmax=1)
        add_colorbar(p)

# Better colorbar for subplots
# https://stackoverflow.com/questions/23876588/matplotlib-colorbar-in-each-subplot

def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


# def add_colorbar(mappable):
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
#     import matplotlib.pyplot as plt
#     last_axes = plt.gca()
#     ax = mappable.axes
#     fig = ax.figure
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = fig.colorbar(mappable, cax=cax)
#     plt.sca(last_axes)
#     return cbar

def imshow(img):

    if img.ndim == 3:
        img = img.permute(1, 2, 0)

    plt.imshow(img.cpu().detach())



def colorize(z : torch.Tensor):
    """
   x
    """

    r = torch.abs(z)
    arg = torch.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1- 1.0/(1.0 + r**0.3)
    s = 0.8

    hv = h
    V = l + s * torch.min(l,1-l)
    Sv = 2 * (1 - l/V)
    Sv[V==0] = 0

    img = torch.zeros((*(h.shape),3),device=z.device)
    img[:,:,0] = hv
    img[:,:,1] = V
    img[:,:,2] = Sv

    c = colors.hsv2rgb(img.detach().cpu().numpy())
    
    return c
