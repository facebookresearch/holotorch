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
import kornia

from holotorch.CGH_Datasets.Single_Image_Dataset import Single_Image_Dataset
from holotorch.CGH_Datasets.HoloDataModule import HoloDataModule

from holotorch.utils.string_processor import convert_integer_into_string
from holotorch.utils.ImageSave import imsave

from holotorch.Calibration_Procedures.CalibrationCaptureSessions.DataSetGenerators import SpeckleGenerator


import holotorch.Calibration_Procedures.CalibrationCaptureSessions.DataSetGenerators.Generator as Generator
                
def capture_speckle_patterns(self,
            speckle_min = 0.01,
            speckle_max = 1,
            gamma = 3,
            sleep = 0.1,
            num_images = 10,
            n_speckl_sizes = 5,
            dpac_encode_intensity = False,
            flag_histogram_equalized = False
            ):

    subfolder = ""
    if dpac_encode_intensity:
        subfolder = 'dpac_speckle'
    else:
        subfolder = 'speckle_on_slm'

    # create the dpac_citl object
    dpac_citl = self.create_dpac_citl()
    dpac_citl.slm.set_images_per_batch(number_images_per_batch=1, number_slm_batches=1)

    R = Generator.compute_radial_map()

    speckle_range = speckle_min + speckle_max * torch.linspace(0,1,n_speckl_sizes) ** gamma
    for idx, speckle_size in enumerate(speckle_range):
        speckle = SpeckleGenerator.compute_speckle(
                R,
                speckle_radius=speckle_size,
                N_images= num_images,
                flag_histogram_equalized = flag_histogram_equalized
                )
        
        for idx_img, pattern in enumerate(speckle):

            # equalize the phase pattern to have a flat histogram
            pattern = kornia.enhance.equalize(pattern)

            #print(pattern.shape)
            if dpac_encode_intensity:                
                # intialize the slms with dpac encoding from the speckle pattern
                dataset     = Single_Image_Dataset(width=1080, num_pixel_y= 1920, image=pattern)
                datamodule  = HoloDataModule(dataset = dataset, batch_size=1)
                dpac_citl.init_with_dpac_from_targets(datamodule = datamodule)
                # grab the dpac encoding
                pattern = dpac_citl.slm.values._data_tensor.squeeze()


            measured, displayed = self.capture_session.capture_pattern(
                x = pattern,
                sleep=sleep,
            )

            model_output = dpac_citl.forward(voltages=pattern[None,None,None,:,:]).data.squeeze()

            speckle_size_str = convert_integer_into_string(idx)
            idx_img_str = convert_integer_into_string(idx_img)

            # Save the displayed, measured, and target image
            prefix = 'speckle_' + speckle_size_str + "_img_" + idx_img_str
            measured_path   = self.get_save_path(
                                    subfolder = subfolder , 
                                    prefix=prefix+'measured',
                                    prefix_folder='measured', 
                                    stack_index=idx_img)
            pattern_path    = self.get_save_path(
                                    subfolder = subfolder, 
                                    prefix=prefix+'pattern', 
                                    prefix_folder='pattern', 
                                    stack_index=idx_img)

            model_path    = self.get_save_path(
                                    subfolder = subfolder, 
                                    prefix=prefix+'target', 
                                    prefix_folder='target', 
                                    stack_index=idx_img)

            # NO TARGET FOR SPECKLE ONLY
            # target_path     = self.get_save_path(prefix=prefix+'target', stack_index=i)
            # imsave(img_dots, target_path)
            imsave(displayed, pattern_path)
            imsave(measured, measured_path)
            imsave(model_output, model_path)
            print(prefix)

            torch.cuda.empty_cache()

