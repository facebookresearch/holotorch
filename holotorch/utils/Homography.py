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
import cv2
import numpy as np
import scipy.optimize as opt
from scipy.ndimage import gaussian_filter
import matplotlib.pylab as plt
import kornia
import tifffile

import kornia
import kornia.geometry as KG
from holotorch.CGH_Datatypes.Light import Light

from holotorch.utils.Visualization_Helper import add_colorbar

def create_disk_kernel(
        kernel_size = 11,
        sigma_smooth = 1
        ):
    x = torch.linspace(-1,1,kernel_size )
    X,Y = torch.meshgrid(x,x)

    R = torch.sqrt(X**2 + Y**2)
    kernel = torch.zeros(R.shape)
    kernel[R < 0.4] = 1

    smooth_kernel_size= 2*int(2*sigma_smooth)+1
    smooth_kernel_size = max(3,smooth_kernel_size)
    

    if sigma_smooth != 0:
        kernel = kornia.filters.gaussian_blur2d(
            kernel[None,None],
            kernel_size = [smooth_kernel_size, smooth_kernel_size],
            sigma = [sigma_smooth, sigma_smooth], border_type='reflect', separable=True
            ).squeeze()
    
    return kernel

def create_dots(
        N_dots_x : int = 10,
        N_dots_y : int = 10,
        border_x :int = 300,
        border_y :int = 300,
        num_pixel_x : int = 1000,
        num_pixel_y : int = 1000,   
    ):
    range_x = num_pixel_x - 2*border_x
    range_y = num_pixel_y - 2*border_y

    x_dots = border_x +  range_x*torch.linspace(0,1, N_dots_x)
    y_dots = border_y +  range_y*torch.linspace(0,1, N_dots_y)

    X_dots, Y_dots = torch.meshgrid(x_dots.to(torch.long),y_dots.to(torch.long))
    return X_dots, Y_dots

def create_dots_image(
        N_dots_x : int = 10,
        N_dots_y : int = 10,
        border_x :int = 300,
        border_y :int = 300,
        num_pixel_x : int = 1000,
        num_pixel_y : int = 1000,   
        kernel_size = 1,
        sigma_smooth = 1
    ):

    X_dots, Y_dots = create_dots(
        N_dots_x=N_dots_x,
        N_dots_y=N_dots_y,
        border_x=border_x,
        border_y=border_y,
        num_pixel_x=num_pixel_x,
        num_pixel_y=num_pixel_y
    )
    
    img = torch.zeros([num_pixel_x,num_pixel_y]) 

    img[X_dots.flatten(), Y_dots.flatten()] = 1

    if kernel_size > 1:
        kernel = create_disk_kernel(
            kernel_size = kernel_size,
            sigma_smooth = sigma_smooth
            )
        img = kornia.filters.filter2d(img[None,None], kernel[None], border_type='reflect', normalized=False, padding='same')[0,0]
        
    img = img / img.max()
    # center_dots = np.stack([Y_dots,X_dots])
    center_dots = torch.stack([Y_dots,X_dots])
    return img, center_dots

def gaussianFilt(img, sigma):
    temp = np.zeros(img.shape)
    if len(img.shape) > 2:
        for c in range(img.shape[2]):
            temp[:,:,c] = gaussian_filter(img[:,:,c], sigma)
    else:
        temp = gaussian_filter(img,sigma)
    return temp

# fit gaussian
def twoD_Gaussian(var, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = var
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def fitGaussian(img):
    
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    xr = np.arange(img.shape[1])#-Nx/2
    yr = np.arange(img.shape[0])#-Ny/2
    xx, yy = np.meshgrid(xr, yr)
    imgc = img if len(img.shape)==2 else np.mean(img, axis=2)    
    imgn = np.ravel(imgc)/np.max(img)
    ym, xm = np.where(imgc == np.max(imgc))
    ym = ym[0]
    xm = xm[0]
    initial_guess = (1,float(xm),float(ym),1,1,0,0)
    popt, pcov = opt.curve_fit(twoD_Gaussian, (xx, yy), imgn, p0=initial_guess)
    xc = popt[1]
    yc = popt[2]
    return xc, yc, popt
    
def refinecenters(img_exp,
                  centers,
                  sqsize=64
                  ):
    """Refines center position by fitting a Gaussian to the image

    Args:
        img_exp ([type]): [description]
        centers ([type]): [description]
        sqsize (int, optional): [description]. Defaults to 64.

    Returns:
        [type]: [description]
    """
    
    img_exp = img_exp.cpu()
    
    refinedcenters = []

    N_centers = centers.shape[0]
    
    for idx in range(N_centers):

        i = centers[idx,0]
        j = centers[idx,1]

        Ny, Nx = img_exp.shape[:2]
        x1,x2 = max(i-sqsize//2, 0), min(i+sqsize//2, Nx-1)
        y1,y2 = max(j-sqsize//2, 0), min(j+sqsize//2, Ny-1)
        crop = img_exp[y1:y2, x1:x2]

        xc, yc, popt = fitGaussian(crop)
        #pluto.plotFittedGaussian(crop, popt)
        refinedcenters.append([x1+xc, y1+yc])

    centers_refined = np.asarray(refinedcenters)
    
    return centers_refined 

def preprocess_and_crop_everthing_but_circle(
            img,
            th = 0.25,
            kernel_size = 45
        ):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if img.max() == 1:
        
        img = (img * 255).astype(np.uint8)
    img[img<(th*255)] = 0

    kernel_size = 45
    img_blur = cv2.GaussianBlur(img,(kernel_size,kernel_size),0)
    img_blur = (img_blur/img_blur.max() * 255).astype(np.uint8)

    img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,131,0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size//2, kernel_size//2))
    img_th = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)
    
    img_th = img_th*1.0 / img_th.max()
    img = img * img_th
    
    kernel_size = 11
    img = cv2.GaussianBlur(img,(kernel_size,kernel_size),0)
    img = (img/img.max() * 255).astype(np.uint8)
    
    img = img / img.max()

    kernel_size = 51
    img_blur = cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

    mask = np.where(img_blur != 0)
    img = img.copy()
    img[mask] = img[mask] / img_blur[mask]
    img = img / img.max()

    return img


def compute_rough_centers(
    img, 
            threshold = 50,
        gaussian_blur_size = 1,
        min_size = 1,
        max_size = 50
):
    """Computes a rough estimate of all dots seen in the image

    Args:
        img ([type]): [description]
        threshold (int, optional): [description]. Defaults to 50.
        gaussian_blur_size (int, optional): [description]. Defaults to 1.
        min_size (int, optional): [description]. Defaults to 1.
        max_size (int, optional): [description]. Defaults to 50.
    """    
    if img.max() <= 1:
        img = img * 255
    gray = img.astype(np.uint8)
    #gray = cv2.GaussianBlur(gray, (gaussian_blur_size, gaussian_blur_size), 0)
# threshold
    th, threshed = cv2.threshold(gray, threshold, 255,
        cv2.THRESH_BINARY)
    # findcontours
    cnts = cv2.findContours(threshed, cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE)[-2]

    # filter by area
    s1 = min_size
    s2 = max_size
    xcnts = []
    x_centers = []
    y_centers = []
    centers = []
    for cnt in cnts:
        if s1<cv2.contourArea(cnt) <s2:
            xcnts.append(cnt)

            x_mean = int(np.round(cnt[:,0,0].mean()))
            y_mean = int(np.round(cnt[:,0,1].mean()))
            
            x_centers.append(x_mean)
            y_centers.append(y_mean)
            
            centers.append([x_mean, y_mean])
    x_centers = np.array(x_centers)
    y_centers = np.array(y_centers)

    return np.array(centers)

def compute_aligned_centers(
    img : torch.Tensor or np.ndarray,
    threshold=40,
    gaussian_blur_size=2,
    min_size=5,
    max_size= 200,
    sqsize = 128,
    N_vertical=5,
    N_horizontal=8,
    debug = False,
    flip_point_symetric : bool = False
        ):
    
    img_crop = preprocess_and_crop_everthing_but_circle(img)

    centers = compute_rough_centers(
        img = img_crop,
        threshold=threshold,
        gaussian_blur_size=gaussian_blur_size,
        min_size=min_size,
        max_size=max_size
    )

    if debug == True:
        print(centers.shape)

    centers_refined = refinecenters(
        img,
        centers=centers,
        sqsize=sqsize
    )

    if debug == True:
        print(centers.shape)

    centers_aligned = align_dots_to_grid(
        centers_refined,
        N_vertical=N_vertical,
        N_horizontal=N_horizontal,
        flip_point_symetric = flip_point_symetric
    )
    
    return centers_aligned, centers


def computer_centers(img,
        threshold = 50,
        gaussian_blur_size = 1,
        min_size = 1,
        max_size = 50
                     ):
    

    centers = compute_rough_centers(
        img = img,
        threshold=threshold,
        gaussian_blur_size=gaussian_blur_size,
        min_size=min_size,
        max_size=max_size
    )
    #centers = np.stack([x_centers, y_centers])

    # print("\nDots number: {}".format(len(xcnts)))

    centers_refined = refinecenters(img, centers)
    
    return centers_refined

# make image for affine transform
def getDots1D(dot_spacing, N, boarder):
    dots = np.arange(0, N//2-boarder, dot_spacing)
    dots = np.concatenate((-np.flip(dots[1:]), dots), axis=0) + N//2
    return dots


def align_dots_to_grid(
            centers,
            N_vertical,
            N_horizontal,
            flip_point_symetric : bool = False
):
    """Takes 2D list of x,y points and aligns to a rectangualr grid system

    Args:
        centers ([type]): [description]
        N_vertical ([type]): [description]
        N_horizontal ([type]): [description]

    Returns:
        [type]: [description]
    """    
    print("num centers found: ", centers.shape[0])
    
    x = centers[:,0]
    y = centers[:,1]

    N_horizontal = 8
    N_vertical = 5


    centers_ordered = np.zeros([2, N_vertical, N_horizontal])

    ind_x = np.argsort(x)
    x = x[ind_x]
    y = y[ind_x]

    for k in range(N_horizontal):
        sub_y = y[ ( k ) * N_vertical : (k +1) * N_vertical ]
        sub_x = x[ ( k ) * N_vertical : (k +1) * N_vertical ]

        ind_y = np.argsort(sub_y)
        
        sub_y = sub_y[ind_y]
        sub_x = sub_x[ind_y]
        centers_ordered[0, :, k] = sub_x
        centers_ordered[1, :, k] = sub_y

    if flip_point_symetric:
        my_x = centers_ordered[0]
        my_y = centers_ordered[1]
        
        my_x = np.fliplr(np.flipud(my_x))
        my_y = np.fliplr(np.flipud(my_y))

        centers_ordered[0] = my_x
        centers_ordered[1] = my_y


    return centers_ordered

def visualize_aligned_dots(
    centers_ordered,  
    figsize=(12,4)
):
    
    N_horizontal = centers_ordered.shape[2]
    N_vertical = centers_ordered.shape[1]

    
    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.scatter(centers_ordered[0], centers_ordered[1])

    plt.subplot(132)
    for k in range(N_horizontal):
        plt.scatter(centers_ordered[0,:,k], centers_ordered[1,:,k])

    plt.subplot(133)

    for k in range(N_vertical):
        plt.scatter(centers_ordered[0,k,:], centers_ordered[1,k,:])
        
    plt.tight_layout()

def sort_measured_dots_to_target(centers,
            N_dots_vertical,
            N_dots_horizontal
            ):
    """
    NOTE: This assumes that the coordinate system of measured and target are already aligned
    and that the detected dots are already somewhat ordered
    I.e. we only need to flip the ordering and we're done
    """
      
    new_centers = centers.copy()

    newshape=[2,N_dots_vertical, N_dots_horizontal]

    new_centers = np.reshape(new_centers, newshape = newshape)

    new_centers[0,:] = np.flipud(new_centers[0,:,:])
    new_centers[1,:] = np.flipud(new_centers[1,:,:])

    print(new_centers.shape)

    new_centers = np.reshape(new_centers, newshape=centers.shape)
    return new_centers

import operator
def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def load_image_from_captured_dataset(
    base_folder : str,
    folder_name : str,
    extension : str = "tiff",
    idx = 0,
):
    
    image_folder = base_folder / folder_name
    
    name_measured = "measured_" + str(idx) + "." + extension
    name_target = "target_" + str(idx) + "." + extension

    path_measured = image_folder / name_measured
    path_target = image_folder / name_target

    img_measured = tifffile.imread(path_measured)
    img_target = tifffile.imread(path_target)
    # img_exp = img_exp * 1.0 / img_exp.max()
    return img_measured, img_target

def visualize_image_pair(
            img1 : torch.Tensor,
            img2 : torch.Tensor,
            figsize=(10,10),
            x0 = None,
            y0 = None,
            width = None,
            height = None,
            vmax = None,
            vmin = None,
            title1 = "",
            title2 = ""
                ):
    plt.figure(figsize=figsize)
    
    if height == None:
        x1 = None
    else:
        x1 = x0 + height

    if width == None:
        y1 = None
    else:
        y1 = y0 + width
    

    
    if isinstance(img1,Light):
        img1 = img1.data
    
    if isinstance(img2,Light):
        img2 = img2.data
    
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().squeeze()
        
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().squeeze()

    img1 = img1[x0:x1,y0:y1]
    img2 = img2[x0:x1,y0:y1]

    plt.subplot(121)
    plt.grid(alpha=0.3)
    _im = plt.imshow(img1, vmax = vmax, vmin=vmin, cmap = 'gray')
    add_colorbar(_im)
    plt.title(title1)
    
    plt.subplot(122)
    plt.grid(alpha=0.3)
    _im = plt.imshow(img2, vmax = vmax, vmin=vmin, cmap = 'gray')
    add_colorbar(_im)
    plt.title(title2)

    plt.tight_layout()


def warp_image(
    image : torch.Tensor,
    target_shape,
    homography,    
):
    
    if torch.is_tensor(homography) == False:
        homography = torch.Tensor(homography)
        
    if homography.ndim == 2:
        homography = homography[None,:,:]

    
    if torch.is_tensor(image) == False:
        image = torch.Tensor(image)
    
    if image.ndim == 2:
        image = image[None,None]

    warped = KG.homography_warp(
    patch_src = image,
    src_homo_dst = homography.to(image.dtype),
    dsize = target_shape[-2:],
    #mode='bicubic',
    normalized_coordinates = True,
    normalized_homography = False
            )
    
    warped = warped.squeeze()
    return warped