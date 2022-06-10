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

from hardwaretorch.Cameras.Camera import Camera
from holotorch.utils.tictoc import *

try:
    from ximea import xiapi
except:
    ModuleNotFoundError

import numpy as np



from holotorch.utils.units import *

class XimeaCamera(Camera):


        
    def __init__(self,
                 exposure = 0,
                 white_balance =0,
                 auto_focus = 0,
                 fps = 0,
                 resolution = 0,
                 grayscale = 0
                 ):
        super().__init__(exposure,white_balance,auto_focus=auto_focus,fps=fps,resolution=resolution, grayscale=grayscale)

        #create instance for first connected camera
        self.cam = xiapi.Camera()

        self.open_device()
        
        #create instance of Image to store image data and metadata
        self.img = xiapi.Image()
        
        # By Default we bin the Ximea to 2,2
        self.set_binning(2,2)

    def __del__(self):
        print("Close Camera")
        self.close_device()
        
    def getImage(self) -> np.ndarray:
        return self.capture_image()

    def capture_image(self) -> np.ndarray:
        cam = self.cam
        try:
            cam.start_acquisition()
        except xiapi.Xi_error:
            pass

        #get data and pass them from camera to img
        cam.get_image(self.img)
        data = self.img.get_image_data_numpy()

        print('Stopping acquisition...')
        cam.stop_acquisition()
        
        return data

    @property
    def height(self) -> int:
        return self.cam.get_height()
    
    @property
    def width(self) -> int:
        return self.cam.get_width()
    

    def getResolution(self):
        """ Returns the current resolution
        """
        return [self.width, self.height]

    def setFullResolution(self):
        self.setResolution(
            width=self.cam.get_width_maximum(),
            height=self.cam.get_height_maximum(),
            offset_x=0,
            offset_y=0
        )    
        

    def setResolution(self, width, height, offset_x = 0, offset_y = 0):
        """Sets the Area-of-Interest to be captured

        Args:
            width ([type]): Width must me divisble by 32
            height ([type]): Width must me divisble by 32
            offset_x (int, optional): [description]. Defaults to 0.
            offset_y (int, optional): [description]. Defaults to 0.
        """
        # Set the area of interest (AOI) to the middle half
        cam = self.cam
        cam.OffsetX = offset_x
        cam.OffsetY = offset_y
        cam.Width = width
        cam.Height = height


    @property
    def gain(self) -> float:
        """Returns gain in dBXI_PRM_GAIN

        Returns:
            float: _description_
        """        
        return self.cam.get_gain()
    
    @gain.setter
    def gain(self, val : float):
        """ Gain in dBXI_PRM_GAIN

        Args:
            val (float): _description_
        """        
        self.cam.set_gain(val)

    @property
    def gain_maximum(self) -> float:
        return self.cam.get_gain_maximum()

    @property
    def exposure_time(self) -> float:
        """Returns exposure time in seconds (10**-6 s)
        """        
        return self.cam.get_exposure() * us
    
    @exposure_time.setter
    def exposure_time(self, exposure : float):
        """ Set the exposure time in seconds
        """
        # Ximea sets in micro seconds so transform
        exposure = exposure / us
        self.cam.set_exposure(exposure)
        
    def open_device(self):
        print('Opening first camera...')
        self.cam.open_device()
    
    def close_device(self):
        #stop communication
        self.cam.close_device()
        
    def set_binning(self, binning_vertical : int = None, binning_horizontal :int = None):
        if binning_vertical is not None:
            self.cam.set_binning_vertical(binning_vertical=binning_vertical)
        if binning_horizontal is not None:
            self.cam.set_binning_horizontal(binning_horizontal=binning_horizontal)

    def print_ximea(self):
        img = self.img
        cam = self.cam
        
        print("Height", img.height)
        print("Width", img.width)
        print(img.AbsoluteOffsetX)
        print("Downsampling", img.DownsamplingX)
        print("Black Level", img.black_level)
        print("Expsoure Time", cam.get_exposure())
        print("Binning Horizontal", cam.get_binning_horizontal())
        print("Binning vertical", cam.get_binning_vertical())
        print("Width Maximum", cam.get_width_maximum())
        print("Height Maximum", cam.get_height_maximum())
        print("Width Minimum", cam.get_width_minimum())
        print("Height Minimum", cam.get_height_minimum())
        print("")
        

