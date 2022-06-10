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

from abc import abstractmethod


class Camera:
    def __init__(self, exposure, white_balance, auto_focus, fps, resolution, grayscale):
        # Exposure passed as float in seconds
        self.exposure = exposure
        # White balanced passed as a float
        self.white_balance = white_balance
        # Auto_focus passed as boolean
        self.auto_focus = auto_focus
        # FPS in float
        self.fps = fps
        # Resolution as tuple (Width, Height)
        self.resolution = resolution
        # Grayscale in boolean
        self.grayscale = grayscale
        # Capture object may be in cv2.capture, pypylon, PySpin etc.
        self.capture = None
        # Calibration object
        self.calibration = None

    @property
    @abstractmethod
    def capture_image(self):
        # To override: Capture image, return frame and save in corresponding folder in specified file format
        raise NotImplementedError()

    @property
    @abstractmethod
    def ExposureTime(self, exposure):
        # To override
        pass

    # @abstractmethod
    # def getExposure(self):
    #     # To override
    #     pass

    # @abstractmethod
    # def getFPS(self):
    #     # To override
    #     pass

    # @abstractmethod
    # def setFPS(self):
    #     # To override
    #     pass

    # @abstractmethod
    # def setAutoGain(self):
    #     # To override
    #     pass

    # @abstractmethod
    # def getGain(self):
    #     # To override
    #     pass

    # @abstractmethod
    # def setGain(self):
    #     # To override
    #     pass

    # def getResolution(self):
    #     return self.resolution

    # @abstractmethod
    # def setResolution(self):
    #     # To override
    #     pass

    # @abstractmethod
    # def viewCameraStream(self):
    #     # To override
    #     pass

    # @abstractmethod
    # def quit_and_close(self):
    #     # To override
    #     pass

    # @abstractmethod
    # def quit_and_open(self):
    #     # To override
    #     pass

    # @abstractmethod
    # def getStatus(self):
    #     # To override
    #     pass

    # @abstractmethod
    # def setCalibration(self, calibration):
    #     # Set the calibration object
    #     self.calibration = calibration




