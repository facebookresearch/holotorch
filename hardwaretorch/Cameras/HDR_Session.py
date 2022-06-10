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


class HDR_Session():
    
    def __init__(self) -> None:
        pass
    
    def compute_exposure_times(self):
        pass
    

    def capture(self):
        # For radiometric calibration a series of differently exposed images of the same object is required
        for exp in self.exposures:
            self.camera.setExposure(exp)
            self.camera.getImage(name='Radiometric/'+str(exp), calibration=True)
