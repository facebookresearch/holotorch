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

# from hardwaretorch.Cameras import Camera
from hardwaretorch.Cameras.Camera import Camera
from holotorch.utils.tictoc import *

try:
    import simple_pyspin
except:
    ModuleNotFoundError

import numpy as np

from enum import Enum

from terminaltables import AsciiTable


class FlirCamera(Camera):

    class AQUISITION_MODE(Enum):
        MultiFrame ='MultiFrame'
        Continuous = 'Continuous'
        SingleFrame = 'SingleFrame'
        
    class PIXEL_FORMAT(Enum):
        """A selective list of available image pixel formats

        Args:
            Enum ([type]): [description]
        """
        BayerGB8 = "BayerGB8" # Returns the Monochromatic Raw image with bayer pattern
        Mono8 = "Mono8" # Returns a monochromatic image demosaiced from raw bayer pattern
        RGB8 = "RGB8" # Returs demosaiced color image

    class AUTO_GAIN(Enum):
        """A selective list of gain options

        Args:
            Enum ([type]): [description]
        """
        OFF = "Off" # Returns the Monochromatic Raw image with bayer pattern
        ONCE = "Once" # Returns a monochromatic image demosaiced from raw bayer pattern
        CONTINUOUS = "Continuous" # Returs demosaiced color image
        
    class EXPOSURE_MODE(Enum):
        """A selective list of AUTO_EXPOSURE options

        Args:
            Enum ([type]): [description]
        """
        OFF = "Off" # Returns the Monochromatic Raw image with bayer pattern
        ONCE = "Once" # Returns a monochromatic image demosaiced from raw bayer pattern
        CONTINUOUS = "Continuous" # Returs demosaiced color image
        
    def __init__(self, exposure = 0, white_balance =0, auto_focus = 0, fps = 0, resolution = 0, grayscale = 0):
        super().__init__(exposure,white_balance,auto_focus=auto_focus,fps=fps,resolution=resolution, grayscale=grayscale)
        
        cam = simple_pyspin.Camera() # Acquire Camera
        cam.init() # Initialize camera
        self.cam = cam


        self.PixelFormat = FlirCamera.PIXEL_FORMAT.Mono8

        # To control the exposure settings, we need to turn off auto
        self.GainAuto = FlirCamera.AUTO_GAIN.OFF
        # Set the gain to 0 dB or the minimum of the camera.
        self.Gain = cam.get_info('Gain')['min']
        
        self.ExposureAuto = FlirCamera.EXPOSURE_MODE.OFF
        self.ExposureTime = 10000 # microseconds

        # If we want an easily viewable image, turn on gamma correction.
        self.GammaEnabled = True
        self.Gamma = 1.0
        
        self.BlackLevel = 0

            

    def __del__(self):
        print("Close Camera")
        self.cam.close() # You should explicitly clean up


    def getImage(self):
        self.cam.start() # Start recording
        img = self.cam.get_array() # Get 10 frames
        self.cam.stop() # Stop recording

        return img

    def capture_sequence(self, AcquisitionFrameCount = 2, debug = False):
        cam = self.cam
        cam.AcquisitionFrameCount = AcquisitionFrameCount

        cam.start() # Start recording

        tic()
        imgs = {}
        for k in range(cam.AcquisitionFrameCount):
            imgs[k] = cam.get_array() # Get 10 frames
        cam.stop() # Stop recording
        time_pass = toc()
        
        imgs = np.array(list(imgs.values()))
        if imgs.ndim == 3:
            imgs = np.transpose(imgs,[1,2,0])
        elif imgs.ndim == 4:
            imgs = np.transpose(imgs,[1,2,3,0])
        else:
            pass

        if debug:
            print("Time Passed", time_pass)
            print("Nr. Frames", cam.AcquisitionFrameCount)
            print("Framerate", cam.AcquisitionFrameRate)
            print("Expected time", 1/cam.AcquisitionFrameRate * cam.AcquisitionFrameCount)
            
        return imgs

    @property
    def AcquisitionMode(self) -> str:
        return self.cam.AcquisitionMode
    
    @AcquisitionMode.setter
    def AcquisitionMode(self, mode : AQUISITION_MODE or str):
        if type(mode) is not str:
            mode = mode.value
        self.cam.AcquisitionMode = mode
        
    @property
    def AcquisitionFrameRate(self) -> str:
        """Returns AcquisitionFrameRate in Hz

        Returns:
            str: [description]
        """        
        return self.cam.AcquisitionFrameRate
    
    @AcquisitionFrameRate.setter
    def AcquisitionFrameRate(self, val : float):
        """Sets the AcquisitionFrameRate in Hz

        Args:
            val (float): [description]
        """        
        self.cam.AcquisitionFrameRate = val

    @property
    def GammaEnabled(self) -> bool:
        return self.cam.GammaEnabled
    
    @GammaEnabled.setter
    def GammaEnabled(self, val : bool = False):
        try:
            self.cam.GammaEnabled = val
        except:
            print("Failed to change Gamma correction (not avaiable on some cameras).")
        
    @property
    def Gamma(self) -> float:
        return self.cam.Gamma
    
    @Gamma.setter
    def Gamma(self, val : float):
        try:
            self.cam.Gamma = val
        except:
            print("Failed to change Gamma correction (not avaiable on some cameras).")
    @property
    def ExposureMode(self) -> EXPOSURE_MODE:
        return self.cam.ExposureAuto
    
    @ExposureMode.setter
    def ExposureMode(self, val : EXPOSURE_MODE or str):
        if val is not str:
            val = val.value   
        self.cam.ExposureAuto = val
        
    @property
    def ExposureTime(self) -> float:
        """Returns exposure time in microseconds (10**-6 s)

        Returns:
            float: [description]
        """        
        return self.cam.ExposureTime
    
    @ExposureTime.setter
    def ExposureTime(self, exposure : float):
        """ Set the exposure time in microseconds

        Args:
            exposure (float): [description]
        """
        self.cam.ExposureTime = exposure



    def getFPS(self):
        # To override
        pass

    def setFPS(self):
        # To override
        pass

    @property
    def BlackLevel(self) -> float:
        return self.cam.BlackLevel
    
    @BlackLevel.setter
    def BlackLevel(self, val : float):
        """Sets the Black Leven in Percentage

        Args:
            val (float): [description]
        """    
        if val < 0:
            val = 0
        if val > 29:
            val = 29
        self.cam.BlackLevel = val
        
    @property
    def GainAuto(self) -> AUTO_GAIN:
        return self.cam.GainAuto
    
    @GainAuto.setter
    def GainAuto(self, val : AUTO_GAIN or str):
        if val is not str:
            val = val.value   
        self.cam.GainAuto = val
        
    def setAutoGain(self, val : AUTO_GAIN or str = AUTO_GAIN.OFF):
        self.GainAuto = val.value   
 
    def getGainInfo(self):
        """Returns information about the gain
        """
        return self.cam.get_info('Gain')
        
    @property
    def Gain(self) -> float:
        return self.cam.Gain
    
    @Gain.setter
    def Gain(self, val : float):
        self.cam.Gain = val
    
    def getGain(self) -> float:
        return self.Gain

    def setGain(self, val : float):
        # To override
        self.Gain = val
        
    def getMaxGain(self) -> float:
        return self.cam.get_info('Gain')['max']

    @property
    def SensorWidth(self) -> int:
        """ Returns Total Pixel in Width Direction

        Returns:
            int: [description]
        """        
        return self.cam.SensorWidth
    
    @property
    def SensorHeight(self) -> int:
        """Return total amount of pixel for height dimension

        Returns:
            int: [description]
        """
        return self.cam.SensorHeight
    
    @property
    def Width(self) -> int:
        return self.cam.Height
    
    @property
    def Height(self) -> int:
        return self.cam.Width

    def getResolution(self):
        """ Returns the current resolution
        """
        return [self.Width, self.Height]

    def setFullResolution(self):
        self.setResolution(
            width=self.SensorWidth,
            height=self.SensorHeight,
            offset_x=0,
            offset_y=0
        )

    def set_pixel_format(self, pixel_format : PIXEL_FORMAT):
        
        self.cam.PixelFormat = pixel_format.value

    def setBinning(self, binning : int):
        """We can bin the camera up to a factor of 2
        """        
        self.cam.BinningVertical = binning
    
    @property
    def Binning(self):
        return self.cam.BinningVertical
        

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



    def viewCameraStream(self):
        # To override
        pass

    def quit_and_close(self):
        # To override
        pass

    def quit_and_open(self):
        # To override
        pass

    def getStatus(self):
        # To override
        pass

    def setCalibration(self, calibration):
        # Set the calibration object
        self.calibration = calibration


    def print_camera(self):

        """Prints information on the camera
        """        
        table_data = [
            ['Name', 'Value'],
            ['Height', self.Height],
            ['Width', self.Width],
            ['SensorHeight', self.SensorHeight],
            ['SensorWidth', self.SensorWidth],
            ['OffsetX', self.cam.OffsetX],
            ['OffsetY', self.cam.OffsetY],
            ['Binning', self.Binning],
            ['--------------', '--------------'],
            ['AcquisitionMode', self.AcquisitionMode],
            ['AcquisitionFrameRate', self.AcquisitionFrameRate],
            ['ExposureMode', self.ExposureMode],
            ['ExposureTime', self.ExposureTime],
            ['--------------', '--------------'],
            ['GammaEnabled', self.GammaEnabled],
            ['Gamma', self.Gamma],
            ['GainAuto', self.GainAuto],
            ['Gain', self.Gain],
            ['BlackLevel', self.BlackLevel],
        ]
        table = AsciiTable(table_data)
        print(table.table)




