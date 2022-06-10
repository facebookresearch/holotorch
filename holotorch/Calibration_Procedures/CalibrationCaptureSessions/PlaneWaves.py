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

from holotorch.CGH_Datasets.Single_Image_Dataset import Single_Image_Dataset
from holotorch.CGH_Datasets.HoloDataModule import HoloDataModule
from holotorch.Optical_Setups.DPAC_CITL import DPAC_CITL

from holotorch.utils.string_processor import convert_integer_into_string
from holotorch.utils.ImageSave import imsave

from holotorch.Calibration_Procedures.CalibrationCaptureSessions.DataSetGenerators import BlazedGratingGenerator

@staticmethod
def capture_ramps(self,
            experiment_name : str,
            dpac_encode_intensity = False,
            images : torch.Tensor = None,
            sleep = .1,
            dpac_citl : DPAC_CITL = None,
            ):
    """_summary_

    Args:
        experiment_name (str): _description_
        dpac_encode_intensity (bool, optional): _description_. Defaults to False.
        images (torch.Tensor, optional): _description_. Defaults to None.
    """        

    subfolder = ""
    if dpac_encode_intensity:
        subfolder = 'dpac_' + experiment_name
    else:
        subfolder = experiment_name + '_on_slm'

    dpac_citl.slm.set_images_per_batch(number_images_per_batch=1, number_slm_batches=1)
    
    N_angles = images.shape[0]
    N_freqs  = images.shape[1]
    
    for idx_angle in range(N_angles):
        for idx_freq in range(N_freqs):
            
            pattern = images[idx_angle, idx_freq]
            
            #print(pattern.shape)
            if dpac_encode_intensity:                
                # intialize the slms with dpac encoding from the speckle pattern
                dataset     = Single_Image_Dataset(width=1080, num_pixel_y= 1920, image=pattern)
                datamodule  = HoloDataModule(dataset = dataset, batch_size=1)
                dpac_citl.init_with_dpac_from_targets(datamodule = datamodule)
                # grab the dpac encoding
                pattern = dpac_citl.slm.values._data_tensor.squeeze().detach()


            measured, displayed = self.capture_session.capture_pattern(
                x = pattern,
                sleep=sleep,
            )

            model_output = dpac_citl.forward(voltages=pattern[None,None,None,:,:]).data.squeeze()

            idx_angle_str = convert_integer_into_string(idx_angle)
            idx_freq_str = convert_integer_into_string(idx_freq)

            # Save the displayed, measured, and target image
            prefix = 'angle' + idx_angle_str + "_freq_" + idx_freq_str
            measured_path   = self.get_save_path(
                                    subfolder = subfolder , 
                                    prefix=prefix+'measured',
                                    prefix_folder='measured', 
                                    stack_index=None)
            
            pattern_path    = self.get_save_path(
                                    subfolder = subfolder, 
                                    prefix=prefix+'pattern', 
                                    prefix_folder='pattern', 
                                    stack_index=None)

            model_path    = self.get_save_path(
                                    subfolder = subfolder, 
                                    prefix=prefix+'target', 
                                    prefix_folder='target', 
                                    stack_index=None)

            # NO TARGET FOR SPECKLE ONLY
            # target_path     = self.get_save_path(prefix=prefix+'target', stack_index=i)
            # imsave(img_dots, target_path)
            imsave(pattern, pattern_path)
            imsave(measured, measured_path)
            imsave(model_output, model_path)
            print(prefix)

            torch.cuda.empty_cache()
 


def capture_blazed_gratings(self,
    num_pixel_x=1080,
    num_pixel_y= 1920,
    freq_low  = 2,
    freq_high = None,
    freq_step = 1,
    angle_rad = 0,
    dpac_encode_intensity = False,
    flag_capture = True,
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
    
    experiment_name = "blazedgratings"
    
    blazed_gratings = BlazedGratingGenerator.compute_blazed_gratings(
            resX = num_pixel_x,
            resY = num_pixel_y,
            freq_low  = freq_low,
            freq_high = freq_high,
            freq_step = freq_step,
            angle_rad = angle_rad,
            flag_blazed_or_sin = True
            )

    if flag_capture:

        self.capture_ramps(
                experiment_name = experiment_name,
                dpac_encode_intensity = dpac_encode_intensity,
                images = blazed_gratings
        )
    
    return blazed_gratings



def capture_linear_fringes(self,
        num_pixel_x=1080,
        num_pixel_y= 1920,
        freq_low  = 2,
        freq_high = None,
        freq_step = 1,
        angle_rad = 0,
        dpac_encode_intensity = False,
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

    experiment_name = "linearfringes"

    linear_fringes = BlazedGratingGenerator.compute_blazed_gratings(
            resX = num_pixel_x,
            resY = num_pixel_y,
            freq_low  = freq_low,
            freq_high = freq_high,
            freq_step = freq_step,
            angle_rad = angle_rad,
            flag_blazed_or_sin = False
            )

    self.capture_ramps(
            experiment_name = experiment_name,
            dpac_encode_intensity = dpac_encode_intensity,
            images = linear_fringes
    )
        

