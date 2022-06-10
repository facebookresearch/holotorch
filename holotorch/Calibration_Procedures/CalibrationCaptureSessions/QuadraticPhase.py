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

from holotorch.Calibration_Procedures.CalibrationCaptureSessions.DataSetGenerators import QuadraticPhaseGenerator
import holotorch.Calibration_Procedures.CalibrationCaptureSessions.DataSetGenerators.Generator as Generator



def capture_quadratic_phase(self,
    num_pixel_x=1080,
    num_pixel_y= 1920,
    scale = torch.arange(1,10),
    amplitude_scale = torch.ones(1),
    dpac_encode_intensity = False,
    flag_blazed_or_not = True,
    ):
    """_summary_

    Args:
        num_pixel_x (int, optional): _description_. Defaults to 1080.
        num_pixel_y (int, optional): _description_. Defaults to 1920.
        freq_low (int, optional): _description_. Defaults to 2.
        freq_high (_type_, optional): _description_. Defaults to None.
        freq_step (int, optional): _description_. Defaults to 1.
        angle_rad (int, optional): _description_. Defaults to 0.
        dpac_encode_intensity (bool, optional): _description_. Defaults to False.
    """    

    if flag_blazed_or_not:
        experiment_name = "blazedquadraticphase"
    else:
        experiment_name = "sinequadraticphase"

    R = Generator.compute_radial_map(
            num_pixel_x=num_pixel_x,
            num_pixel_y= num_pixel_y,
        )

    img = QuadraticPhaseGenerator.compute_quadratic_phase_targets(
        R = R,
        scale=scale,
        amplitude_scale=amplitude_scale,
        flag_blazed_grating = flag_blazed_or_not,
    )


    self.capture_ramps(
            experiment_name = experiment_name,
            dpac_encode_intensity = dpac_encode_intensity,
            images = img
    )
    

