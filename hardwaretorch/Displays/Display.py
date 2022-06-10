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


class Display:
    def __init__(self):
        pass

    @abstractmethod
    def display_pattern(self, pattern):
        # To override
        pass

    @abstractmethod
    def get_resolution(self):
        # To override
        pass

    @abstractmethod
    def set_resolution(self, resolution):
        # To override
        pass

    @abstractmethod
    def quit_and_close(self):
        # To override
        pass

    @abstractmethod
    def getStatus(self):
        # To override
        pass