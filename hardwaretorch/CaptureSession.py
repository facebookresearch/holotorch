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

# from hardwaretorch.Cameras import Camera
# from hardwaretorch.Displays import Display
try:
    from hardwaretorch.Cameras.FlirCamera import FlirCamera
except:
    pass
try:
    from hardwaretorch.Cameras.XimeaCamera import XimeaCamera
except:
    pass
from hardwaretorch.Cameras.DummyCamera import DummyCamera
from hardwaretorch.Cameras.Camera import Camera
from hardwaretorch.Displays.Display import Display
from hardwaretorch.Displays.SLM_Display import SLM_Display
from hardwaretorch.Displays.DummyDisplay import DummyDisplay
from holotorch.utils.tictoc import *
import time

from holotorch.utils.Enumerators import *


class CaptureSession():
    

    
    def __init__(self,
                 display : Display,
                 camera : Camera
                 ) -> None:
        self.display = display
        self.camera = camera
    
    @classmethod
    def dummy_session(
        cls
    ):
        return CaptureSession.default_session(
                camera_type = CAMERA_TYPE.DUMMY,
                display_type = DISPLAY_TYPE.DUMMY
        )
    
    # a class method to create a Person object by birth year.
    @classmethod
    def default_session(cls,
                        camera_type : CAMERA_TYPE = CAMERA_TYPE.DUMMY,
                        display_type : DISPLAY_TYPE = DISPLAY_TYPE.SLM
                        ):
        
        if display_type is DISPLAY_TYPE.SLM:
            # my_display = hardwaretorch.Displays.SLM_Display( isImageLock = True, alwaysTop = False)
            my_display = SLM_Display( isImageLock = True, alwaysTop = False)
        elif display_type is type(display_type).DUMMY:
            my_display = DummyDisplay()
        else:
            raise NotImplementedError()


        if camera_type is CAMERA_TYPE.FLIR:
            # my_cam = hardwaretorch.Cameras.FlirCamera(
            my_cam = FlirCamera()
        elif camera_type is type(camera_type).XIMEA:
            my_cam = XimeaCamera()
        elif camera_type is CAMERA_TYPE.DUMMY:
            my_cam = DummyCamera()
        else:
            raise NotImplementedError()

        return cls(display = my_display, camera = my_cam)
      
      
    def capture_pattern(self,
                        x : np.ndarray,
                        sleep = 0.25,
                        # max_phase = 2*np.pi,
                        bit_depth = 8
                        ) -> np.ndarray:
        """ Displays a pattern and captures an image

        Args:
            x (np.ndarray): [description]
            sleep (float, optional): [description]. Defaults to 0.25.

        Returns:
            np.ndarray: [description]
        """ 
        # Displat the pattern       
        # display_pattern = self.display.display_pattern(x, sleep = sleep)
        display_pattern = self.display.display_pattern(x, sleep = 0, bit_depth=bit_depth)
        time.sleep(sleep)
        
        # Capture the image
        # tic()

        captured_img = self.camera.getImage()
        # time.sleep(.5)
        # elapsed = toc()
        # print('image capture took: ', elapsed, ' sec')
        return captured_img, display_pattern