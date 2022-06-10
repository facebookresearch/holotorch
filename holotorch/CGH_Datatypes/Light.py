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

from __future__ import annotations
from torch.types import _device, _dtype, _size
import torch
import numpy as np
import pathlib
import matplotlib
import matplotlib.pyplot as plt
from copy import copy

from holotorch.Spectra import Spectrum
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.Spectra.SpacingContainer import SpacingContainer
from holotorch.utils.ImageSave import imsave
from holotorch.utils.Helper_Functions import ft2
import holotorch.utils.Visualization_Helper as VH
import holotorch.utils.pjji as piji

from holotorch.utils.Enumerators import *
class Light():

    _wavelengths = None
    _data = None
    
    _dimension_state = None
    
    _spacing = None

    def __init__(self, 
                data : torch.Tensor,
                wavelengths : WavelengthContainer = None,
                spacing : SpacingContainer = None,
                requires_grad = None,
                identifier : FIELD_IDENTIFIER = FIELD_IDENTIFIER.NOTHING
                ):
        
        if np.isscalar(wavelengths):
            wavelengths = WavelengthContainer(wavelengths=wavelengths)

        if np.isscalar(spacing):
            spacing = SpacingContainer(spacing=spacing)
        
        self.spacing : SpacingContainer = spacing
        #assert isinstance(spectrum, type(Spectrum)), "Spectrum must be of instance Spectrum"          
        self.wavelengths = wavelengths
        assert torch.is_tensor(data), "Data must be a torch tensor"
        self.data : torch.Tensor = data      

        self.identifier = identifier

    def like(self, data : torch.Tensor) -> Light:

        return Light(
            data        = data,
            wavelengths = self.wavelengths,
            spacing     = self.spacing,
            identifier  = self.identifier
        )
    

    @property
    def spacing(self) -> torch.Tensor:
        """ Spacing of each of the datatensors stored in here.
        
        NOTE: Most of the time this should be constant, but
        it can happen that this is different for time and wavelength dimension

        Returns:
            torch.Tensor: [description]
        """        
        return self._spacing
    
    @spacing.setter
    def spacing(self, spacing : torch.Tensor) -> None:
        self._spacing = spacing
                  
    @property
    def wavelengths(self):
        return self.wavelengths

    @wavelengths.setter
    def wavelengths(self, spectrum):
        self.wavelengths = spectrum

    @property
    def requires_grad(self):
        return self.data.requires_grad
          
    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    
    def abs(self) -> Light:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data = self.data.abs()
        wavelengths = self.wavelengths   
        spacing = self.spacing
             
        return Light(
            data = data,
            wavelengths=wavelengths,
            spacing = spacing,
            identifier=self.identifier
                     )    
    

    def angle(self) -> Light:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data = self.data.angle()
        wavelengths = self.wavelengths   
        spacing = self.spacing
             
        return Light(
            data = data,
            wavelengths=wavelengths,
            spacing = spacing,
            identifier=self.identifier
                     )    
    
    def detach(self) -> Light:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data = self.data.detach()
        wavelengths = self.wavelengths.detach()
        spacing = self.spacing.detach()

        
        return Light(
            data = data,
            wavelengths=wavelengths,
            spacing = spacing,
            identifier=self.identifier
                     )    
        
    def cpu(self) -> Light:
        """ Detaches the data and wavelength tensor from the computational graph
        """        
        
        data = self.data.cpu()
        wavelengths = self.wavelengths.detach()
        spacing = self.spacing.detach()

        return Light(
            data = data,
            wavelengths=wavelengths,
            spacing = spacing,
            identifier=self.identifier
                     )    
        
    def __sub__(self, other) -> Light:
        
        new_data = self.data - other.data
        
        return Light(
            data = new_data,
            wavelengths = self.wavelengths,
            spacing = self.spacing,
            identifier = self.identifier
        )
        
    def __add__(self, other : Light) -> Light:
        
        new_data = self.data + other.data
        
        return Light(
            data = new_data,
            wavelengths = self.wavelengths,
            spacing = self.spacing,
            identifier = self.identifier
        )
        
    def __mul__(self, other : Light) -> Light:
        
        new_data = self.data * other.data

        return Light(
            data = new_data,
            wavelengths = self.wavelengths,
            spacing = self.spacing,
            identifier = self.identifier
        )

    def __getitem__(self, keys):
        
        if np.isscalar(keys):
            keys = [keys]
        
        if isinstance(keys[0], int):
            if keys[0] > self.num_batches - 1:
                raise ValueError("Index can't be larger than number of batches.")
        # Change the keys
        keys = list(keys)
        new_keys = copy(keys)
        for idx, key in enumerate(keys):
            new_key = key
            if np.isscalar(key):
                new_key = slice(key,key+1,1)
            new_keys[idx] = new_key
            
        # subslices data
        data = self.data[new_keys]
         
        new_spectrum = self.wavelengths
        new_spacing = self.spacing
        
        new_field = Light(
                    data =  data,
                    wavelengths=    new_spectrum,
                    spacing =   new_spacing,
                    identifier = self.identifier
                    )
        
        return new_field
    
    def __str__(self) -> str:
        mystr = "======================================================\n"
        mystr += "Type of field: " + type(self).__module__ 
        try:
            mystr += "\nShape of Data Tensor: " + str(self.data.shape) + " ( n_wave = " + str(len(self.wavelengths)) + ")"
        except TypeError:
            pass
        # mystr += "\nNumber of wavelengths: " + str(len(self.wavelengths))

        return mystr
    
    def __repr__(self):
        
        mystr = type(self).__module__ 
        mystr += "\nShape of Data Tensor: " + str(self.data.shape)

        return mystr
    
    @staticmethod
    def zeros(
        size : _size,
        spectrum : Spectrum,
        device : _device = None ,
        dtype : _dtype = None,
        **param) -> Light:
        """[summary]

        Args:
            size (_size): [description]
            wavelengths ([type]): [description]
            device (_device, optional): [description]. Defaults to None.
            dtype (_dtype, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """        
        assert spectrum is not None, "Wavelenghts cannot be None"
        tmp = torch.zeros(size = size,dtype=dtype, device=device, **param)
        return Light(data = tmp, wavelengths=spectrum, **param)
    
    @staticmethod
    def ones(
            size : _size,
             spectrum : Spectrum,
             device : _device = None ,
             dtype : _dtype = None,
             **param) -> Light:
        """[summary]

        Args:
            size (_size): [description]
            wavelengths ([type]): [description]
            device (_device, optional): [description]. Defaults to None.
            dtype (_dtype, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """        
        assert spectrum is not None, "Wavelenghts cannot be None"

        tmp = torch.ones(size = size,dtype=dtype, device=device, **param)
        return Light(data = tmp, wavelengths=spectrum, **param)
   
    _BATCH = 0
    _TIMES = 1
    _PUPIL = 2
    _CHANNELS = 3
    _HEIGHT = 4
    _WIDTH = 5
    
    @property
    def BATCH(self):
        return self._BATCH  

    @property
    def PUPIL(self):
        return self._PUPIL  
    
    @property
    def TIMES(self):
        return self._TIMES  
    
    @property
    def CHANNELS(self):
        return self._CHANNELS  
    
    @property
    def num_batches(self):
        return self.shape[self._BATCH]   

    @property
    def wavelengths(self) -> WavelengthContainer:
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, wavelengths : WavelengthContainer):
        self._wavelengths = wavelengths

    @property
    def ndim(self):
        return self.data.ndim    

    @property
    def shape(self):
        return self.data.shape    

    @property
    def num_times(self):
        return self.shape[self._TIMES]


    @property
    def num_pupils(self):
        return self.shape[self._PUPIL]
    
    
    @property
    def num_channels(self):
        return self.shape[self._CHANNELS]
    
    @property
    def height(self):
        return self.shape[self._HEIGHT]
    
    @property
    def width(self):
        return self.shape[self._WIDTH]
    
    class ENUM_TensorDIM():
        BPC_HW = "BPC"
        BTPC_HW = "BTPC"
        HW = "HW"
    
    class ENUM_OUTPUT_TYPE():
        GIF = "GIF"
        ROW = "ROW"
        COL = "COL"
        GRID = "GRID"
    
    
    """
    B x T x P 
    5 x 1 x 1         == > Plot 5 different images
    1 x 1 x 5         == > Plot 5 different images 
    1 x 5 x 1         == > Plot 5 different images

    """
    
    @staticmethod
    def to_matplotlib_view(data : torch.Tensor) -> torch.Tensor:
        
        if data.ndim == 2:
            data = data
        elif data.ndim == 3:
            data = data.permute(1,2,0)
        else:
            raise NotImplementedError("This is not yet implemented")
        
        return data

    def log(self):
        
        data = self.data.log()
        
        new_light = Light(
            data=data,
            wavelengths=self.wavelengths,
            spacing = self.spacing
        )
        
        return new_light

    def visualize_fft(self,
        flag_abs : bool = True,
        flag_log : bool = True
        ):
        
        
        fft2 = ft2(self)

        fft2 = Light(
                data=fft2,
                )

        fft2.abs()



    def visualize(self,
            title : str             =   "",
            flag_colorbar : bool    = True,
            flag_axis : str         = False,
            cmap                    ='gray',
            index                   = None,
            open_new_figure         = False,
            figsize                 = None,
            vmax                    = None,
            vmin                    = None,
            adjust_aspect : bool    = False,
            rescale_factor : float  = 1.0,
            ):
        

        
        if index is None:
            for k in range(self.num_batches):
                if open_new_figure:
                    plt.figure()
                self[k].visualize_image(
                    vmax = vmax,
                    vmin = vmin,
                    figsize=figsize,
                    flag_colorbar = flag_colorbar,
                    title = title,
                    flag_axis = flag_axis,
                    adjust_aspect=adjust_aspect,
                    rescale_factor = rescale_factor
                     )
        else:
            out = self[index]
            out.visualize(  
                vmax = vmax,
                vmin = vmin,
                figsize=figsize,
                flag_colorbar = flag_colorbar,
                title = title,
                flag_axis = flag_axis,
                adjust_aspect = adjust_aspect,
                rescale_factor = rescale_factor,
                    )
    
    def visualize_image(self,
            title : str             =   "",
            flag_colorbar : bool    = True,
            flag_axis : str         = False,
            cmap                    ='gray',
            vmax                    = None,
            vmin                    = None,
            figsize                 = None,
            adjust_aspect : bool    = False,
            rescale_factor  : float = 1.0,
            ) -> int:
        """[summary]

        Args:
            data (torch.Tensor): [description]
            vis_type (Type, optional): [description]. Defaults to None.

        Returns:
            int: [description]
        """
        
        if figsize is not None:
            plt.figure(figsize=figsize)
        
        object = self

        if rescale_factor != 1.0:
            object = object.rescale(rescale_factor)
        data = object.data.cpu().detach().squeeze()
        
        if data.is_complex():
            data = data.abs()

        data = Light.to_matplotlib_view(data)

        if flag_axis == True:
            if self.spacing == None:
                size_x = self.height
                size_y = self.width
                unit = ""
                unit_val = 1
                extent = [0, size_y, 0, size_x]

            else:
                size_x = float(self.spacing.data_tensor[...,0]  / 2.0 * self.height ) 
                size_y = float(self.spacing.data_tensor[...,1]  / 2.0 * self.width ) 

                unit_val, unit = VH.float_to_unit_identifier(max(size_x,size_y))

                size_x = size_x / unit_val
                size_y = size_y / unit_val

                extent = [-size_y, size_y, -size_x, size_x]
        else:
            extent = None
            size_x = self.height
            size_y = self.width

        if adjust_aspect:
            aspect = size_x / size_y
        else:
            if np.isclose(size_x, size_y):
                aspect = self.height / self.width
                #aspect = "auto"

            else:
                aspect = 1

        if data.ndim == 2:
            _im = plt.imshow(data,
             cmap=cmap, 
             vmax = vmax,
              vmin = vmin,
              extent = extent,
              aspect = aspect
              )
        else:
            if vmax == None:
                vmax = 1
            # If it's an RGB image we need to normalize
            if data.ndim == 3 and data.shape[2] == 3:
                data = data/data.max()  * vmax
                
            _im = plt.imshow(
                data,
                extent = extent,
                aspect = aspect
                )
        
        if flag_axis:
            plt.axis("on")
        else:
            plt.axis("off")

        if flag_axis:
            if unit != "":
                plt.xlabel("Position (" + unit + ")")
                plt.ylabel("Position (" + unit + ")")

        plt.title(title)

        if flag_colorbar:
            VH.add_colorbar(_im)

        if figsize is not None:
            plt.tight_layout()
        
        

        

    
    def _identify_non_singular_dimensions(self):
        """Identifies the dimensions in the object that are not 1
        
        By definition HEIGHT and WEIGHT are never singular
        Channel Dimension can be 1, but is also not a singular one
        """        
        
        non_singular_list = []
        
        if self.num_batches != 1:
            non_singular_list.append(self.BATCH)

        if self.num_times != 1:
            non_singular_list.append(self.TIMES)

        if self.num_pupils != 1:
            non_singular_list.append(self.PUPIL)
        
        if non_singular_list == []:
            non_singular_list.append(self.BATCH)
            
        return non_singular_list

    def rescale(self, scale = 0.5):
        from holotorch.Optical_Components.Resize_Field import Resize_Field

        resizer = Resize_Field(
            scale_factor=scale
        )

        out = resizer(self)
        return out

    def animate_time_series(self, t,
                            data,
                            flag_axis = "off",
                            title = None,
                            cmap='gray'):
        plt.cla()   
        
        #print(str(t))
        
        im_data = data[t]
        
        # Normalize if it's a color image
        if im_data.shape[2] == 3:
            im_data = im_data / im_data.max()

        #plt.plot(range(100))
        plt.imshow(im_data, vmax=1)
        plt.title(t)
        plt.axis(flag_axis)
        plt.tight_layout()
    
    def save(self,
        filename    : str or pathlib.Path,
        folder      : str = None,
        extension   : str = "tiff"
            ):
        
        img = self.data.detach().squeeze().cpu()
        
        imsave(
            filename    = filename,
            folder      = folder,
            extension   = extension,
            data        = img
        )

    def show_piji(self, title : str = ""):
        img = self.data.detach().cpu().squeeze()

        piji.show(img, title=title)


    def visualize_time_series_gif(self,
                    figsize=(5,3)
                    ):
        
        # Prepare data for plotting
        data = self.data.detach().cpu()
        
        non_singular = self._identify_non_singular_dimensions()

        assert len(non_singular) == 1, "Singular Values can only be one"
        
        new_shape = [
            self.shape[non_singular[0]],
            self.num_channels,
            self.height,
            self.width
        ]
        
        data = data.view( new_shape )
        
        data = data.permute(0,2,3,1)        
        
        data = data

        plt.ioff()
        fig,axis = plt.subplots(figsize=figsize)
        
        flag_axis   = "off"
        title       = "Test"

        anim = matplotlib.animation.FuncAnimation(fig,
                                        self.animate_time_series,
                                        fargs = (data, flag_axis,title, ),
                                        frames = data.shape[0],
                                        )
        #plt.close()

        #f = r"./examples/Example_Notebooks/animation2.gif" 
        #anim.save(f, writer='pillow', fps=2)
        #plt.show()
        plt.ion()
        return anim
    


    def visualize_grid(self,
            flag_axis = "off",
            time_idx = 0,
            suptitle = "Collection",
            flag_colorbar=True,
            title = None,
            max_images = 9,
            figsize=(12,7),
            num_col = None,
            num_row = None,
            cmap='gray',
            vmin = None,
            vmax = None
                                   ) -> None:
        #plt.rcParams["animation.html"] = "jshtml"
        #plt.rcParams['figure.dpi'] = 150  
        
        plt.ion()
        plt.show()
        plt.close()
        
        # Prepare data for plotting
        data = self.data.detach().cpu()
        
        non_singular = self._identify_non_singular_dimensions()

        assert len(non_singular) == 1, "Singular Values can only be one"
        
        new_shape = [
            self.shape[non_singular[0]],
            self.num_channels,
            self.height,
            self.width
        ]
        
        data = data.view( new_shape )
        data = data.permute(0,2,3,1)     
        
        num_images = new_shape[0]
        if num_images > max_images:
            num_images = max_images
        
        plt.figure(figsize=figsize)

        if num_col == None and num_row == None:
            num_col = int(np.ceil(np.sqrt(num_images)))
            num_row = int(np.floor(np.sqrt(num_images)))
        else:
            pass
            
        Light.imshow_grid(
            data = data,
            num_col=num_col,
            num_row=num_row,
            flag_colorbar=flag_colorbar,
            flag_axis = flag_axis,
            cmap = cmap,
            title = title,
            vmin = vmin,
            vmax = vmax,
        )
        
        plt.tight_layout()
        
        plt.show()


    @staticmethod
    def imshow_grid(
            data : torch.Tensor,
            num_col : int,
            num_row : int,
            flag_colorbar : bool = True,
            flag_axis : str = "off",
            cmap = 'gray',
            flag_tight_layout = bool,
            title = None,
            vmin = None,
            vmax = None
        ):
        
        num_images = data.shape[0]
        
        for k in range(num_col * num_row):
            try:
                tmp = data[k]
            except IndexError:
                continue
            plt.subplot(num_row,num_col,k+1)

            # Normalize if it's a color image
            if tmp.shape[2] == 3:
                tmp = tmp / tmp.max()

            _im = plt.imshow(tmp.squeeze(),cmap=cmap, vmin = vmin, vmax = vmax)
            if flag_colorbar:
                VH.add_colorbar(_im)

            plt.axis(flag_axis)
            if title is not None:
                plt.title(str(k))
            
        #plt.tight_layout()
        

    def animate_grid_gif(self, 
            t : int,
            data : torch.Tensor,
            flag_axis = "off",
            title : str = None, 
            cmap='gray',
            flag_colorbar : bool = True
                        ) -> None:
        
        # Get the sub image
        data = data[t]
        
        num_images = data.shape[0]
        num_col = int(np.ceil(np.sqrt(num_images)))
        num_row = int(np.floor(np.sqrt(num_images)))
        
        if (num_col * num_row) < num_images:
            num_row = num_row + 1
            
        Light.imshow_grid(
            data = data,
            num_col=num_col,
            num_row=num_row,
            flag_colorbar=False,
            flag_axis = flag_axis,
            cmap = cmap
        )

    def visualize_grid_gif(self,
            flag_axis = "off",
            time_idx = 0,
            suptitle = "Collection",
            flag_colorbar=True,
            title = None,
            max_images = 9,
            figsize=(6,7),
            cmap='gray'
                            ) -> None:
        #plt.rcParams["animation.html"] = "jshtml"
        #plt.rcParams['figure.dpi'] = 150  
        
        
        # Prepare data for plotting
        data = self.data.detach().cpu()
        
        non_singular = self._identify_non_singular_dimensions()

        assert len(non_singular) == 2, "Input needs to be 2D"
        
        new_shape = [
            self.shape[non_singular[0]],
            self.shape[non_singular[1]],
            self.num_channels,
            self.height,
            self.width
        ]
        
        data = data.view( new_shape )
        data = data.permute(0,1,3,4,2)        
        
        data = data
        
        flag_axis   = "off"
        title       = "Test"
        
        #plt.ioff()
        fig = plt.figure(figsize=figsize)

        anim = matplotlib.animation.FuncAnimation(fig,
                                        self.animate_grid_gif,
                                        fargs = (data, flag_axis,title, ),
                                        frames = data.shape[0]
                                        )
        

        f = r"./examples/Example_Notebooks/animation.gif" 
        #anim.save(f, writer='pillow', fps=5)

        return anim
    
