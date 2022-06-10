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

import pathlib
import torch, time
from hardwaretorch.CaptureSession import CaptureSession
from holotorch.CGH_Datasets.CGH_Dataset import CGH_Dataset

from holotorch.CGH_Datasets.HoloDataModule import HoloDataModule
from holotorch.Optical_Setups.DPAC_FocalStackGenerator import DPAC_FocalStackGenerator

from holotorch.utils.string_processor import convert_integer_into_string
from holotorch.utils.ImageSave import imsave



import numpy as np
class DataSetCaptureSession():
    
    def __init__(self,
            base_folder : pathlib.Path,   
            capture_session : CaptureSession,
            distances_range = None,
            zmin        = 0,
            zmax        = 0,
            n_stack     = 1,
            slm_spacing : float = None,
            wavelength  : float = None
                 ) -> None:
        self.base_folder = base_folder
        self.capture_session = capture_session
        
        #self.dpac = DPAC()
        
        self.dpac_focal_stack_generator = DPAC_FocalStackGenerator(
            slm_spacing = slm_spacing,
            wavelength = wavelength
        )

        if distances_range is None:
            self.dpac_focal_stack_generator.create_focal_range(
                z_min = zmin,
                z_max = zmax,
                N_focal_stack_images = n_stack
            )
        else:
            self.dpac_focal_stack_generator.set_distances_range(distances_range)

    @property
    def distances_range(self):
        return self.dpac_focal_stack_generator.distances_range

    @staticmethod
    def _assemble_name(index : int, name : str, integer_depth : int  = 4) -> str:
        index_str = convert_integer_into_string(index, depth = integer_depth) 
        
        mystr = name + "_" + index_str
        return mystr

    @staticmethod
    def get_save_path(
            name_dataset : str, # E.g. Divk2k
            name_image_type : str, # E.g. Measured
            index_list = [0,0,0],
            name_list  = ["First", "Second", "Third"],
            integer_depth = [4,4,4],
            file_extension = "tiff" ,
            base_folder = None,
            ):
        """_summary_

        """

        assert len(index_list) == len(name_list)
        assert len(index_list) == len(integer_depth)
        
        filename = ""
        for index in range(len(index_list)):
            
            tmp_name = DataSetCaptureSession._assemble_name(
                index = index_list[index],
                name  = name_list[index],
                integer_depth=integer_depth[index]
            )
            if index is not 0:
                tmp_name = "_" + tmp_name
            filename += tmp_name
        
        filename = filename + "." + file_extension
        
        # Create the save folder name       
        save_folder = base_folder / name_dataset / name_image_type
        # create the folder on disk
        save_folder.mkdir(exist_ok=True, parents = True)
        
        # Assemble the save path
        save_path = save_folder / filename        
        
        return save_path
    
    def capture_focal_stack(self,
            optimized_focal_stack_pattern : torch.Tensor,
            name_dataset : str,
            image_indx : int,
            camera_position_ind : int,
            target_image : torch.Tensor,
            sleep : float       = .1,
            file_extension : str = "tiff"
                            ):
        

        for stack_idx, image in enumerate(optimized_focal_stack_pattern):
            
            # Get the DPAC encoding (these are voltage maps)
            measured, displayed = self.capture_session.capture_pattern(
                x = image,
                sleep=sleep,
            )            

            name_list = ["measured", "pattern"]
            object_list = [measured, image]
            
            if target_image is not None:
                name_list.append("target")
                object_list.append(target_image)             
            
            for idx in range(len(name_list)):
                name = name_list[idx]
                object = object_list[idx]
                
                if torch.is_tensor(object):
                    object = object.detach().cpu()
                object = object.squeeze()

                # save target image
                target_path     = DataSetCaptureSession.get_save_path(
                    name_dataset        = name_dataset,
                    name_image_type     = name,
                    index_list = [image_indx, stack_idx, camera_position_ind],
                    name_list  = ["img", "focus", "CamPosZ"],
                    integer_depth = [4,2,2],
                    file_extension      = file_extension,
                    base_folder = self.base_folder
                                            )
                imsave( object , target_path)
    
            
            # save target image
            target_path     = DataSetCaptureSession.get_save_path(
                name_dataset        = "dots",
                name_image_type     = "pattern",
                index_list = [image_indx, idx,camera_position_ind],
                name_list  = ["img", "focus", "CamPosZ"],
                integer_depth = [4,2,2],
                file_extension      = "tiff",
                base_folder = self.base_folder
                                        )
            
            imsave(
                data= image.detach().cpu().squeeze(),
                filename = target_path,
            )

    def capture_pattern_dataset_on_slm(self,
            dataset : CGH_Dataset,
            name_dataset        = 'speckle',                         
            sleep : float       = .1,
            motor_positions     = [0], 
            motor               = None,
            motor_sleep         = 2
            ):
        
        datamodule = HoloDataModule(
                    dataset=dataset,
                    batch_size = 1,
                    shuffle=False
                    )


        # enumerate through dataset
        for idx, voltage_modulation in enumerate(datamodule):
            # enumerate through motor positions
            for m_idx, pos in enumerate(motor_positions):
                if motor != None:
                    moving = True
                    while moving:
                        try:
                            motor.move_absolute(float(pos))
                            moving = False
                        except Exception as e:
                            moving = True
                    time.sleep(motor_sleep)

                # Get the voltage maps
                measured, displayed = self.capture_session.capture_pattern(
                    x = voltage_modulation,
                    sleep=sleep,
                )

                self.save_citl_image_pair(
                    measured = measured,
                    pattern  = displayed,
                    target   = None,
                    img_index = idx,
                    stack_index = m_idx,
                    offset_index  = 0,
                    name_dataset = name_dataset,
                    file_extension = "tiff" ,
                    )

    def capture_dpac_encoded_dataset(self,
            dataset : CGH_Dataset,
            name_dataset        = 'div2k',                         
            sleep : float       = .1,
            scale               = 2.0,
            off                 = 2*np.pi,
            max_phase           = 4*np.pi,
            num_voltage_offset  = 0,
            num_phase_noise     = 1,
            min_phase_sigma     = 0,
            max_phase_sigma     = 1,
            outside_of_boundary = 0.0,
            camera_position_ind : int = 0,
            bit_depth : int     = 8
            ):
        """ Captures a full dataset

        Args:
            name_dataset (str, optional): _description_. Defaults to 'div2k'.
            sleep (float, optional): _description_. Defaults to .1.
            crop (_type_, optional): _description_. Defaults to None.
            voltage_offsets (_type_, optional): _description_. Defaults to None.
            dataset (_type_, optional): _description_. Defaults to None.
        """
        datamodule = HoloDataModule(
                    dataset=dataset,
                    batch_size = 1,
                    shuffle=False
                    )

        
        sigma_range = torch.linspace(min_phase_sigma,max_phase_sigma, num_phase_noise)
        
        #
        # Iterates through the whole datamodule
        #
        for idx, intensity_image in enumerate(datamodule):
            
            # DPAC should encode amplitude and not intensities
            amplitude_image = torch.sqrt(intensity_image)
            
            # 
            # ITERATE THROUGH PHASE NOISE
            #
            for idx_phase_noise in range(num_phase_noise):
            
                tmp_sigma = sigma_range[idx_phase_noise]    
                if tmp_sigma == 0:
                    field = amplitude_image
                else:       
                    random_phase = tmp_sigma * torch.randn(amplitude_image.shape)
                    field = amplitude_image * torch.exp(1j * random_phase)
                
                # ########################################
                # COMPUTE the DPAC ENCODED 
                # ########################################
                
                dpac_modulation = self.dpac_focal_stack_generator.compute_dpac_focal_stack(
                    target      = field,
                    scale       = scale,
                    off         = off,
                    max_phase   = max_phase,
                )
                
                # # Get the DPAC encoding (these are voltage maps)
                # voltage_modulation = dpac_citl.compute_dpac_phase_from_taget(
                #                     target_field = field,
                #                     scale       = scale,
                #                     off         = off,
                #                     max_phase   = max_phase,
                #                 )
                
                for idx_focal_stack, voltage_modulation in enumerate(dpac_modulation):

                    # shift voltage around 
                    vmin = voltage_modulation.min()
                    vmax = 1.0 - voltage_modulation.max()
                    # vshift = vmin + *torch.rand(1)
                    # voltage_modulation = voltage_modulation + vshift
                                
                    if num_voltage_offset > 1:
                        voltage_offsets = - vmin + (vmax+vmin)*torch.linspace(-outside_of_boundary,1+outside_of_boundary, num_voltage_offset)
                    else:
                        voltage_offsets = [.1]

                    # ####################################
                    # ITERATE THROUGH VOLTAGE OFFSETS
                    # ####################################
                    for idx_voltage in range(len(voltage_offsets)):
                        voltage_offset = voltage_offsets[idx_voltage]
                        
                        tmp_voltage_modulation = voltage_modulation + voltage_offset
                        tmp_voltage_modulation = tmp_voltage_modulation % 1            
                        
                        measured, displayed = self.capture_session.capture_pattern(
                            x = tmp_voltage_modulation,
                            sleep=sleep,
                            bit_depth=bit_depth
                        )
                        
                        self.save_citl_image_pair(
                            measured = measured,
                            pattern  = displayed,
                            target   = intensity_image.abs().cpu().squeeze(),
                            img_index = idx,
                            offset_index  = idx_voltage,
                            noise_index  = idx_phase_noise,
                            stack_index  = idx_focal_stack,
                            name_dataset = name_dataset,
                            camera_position_ind = camera_position_ind,
                            file_extension = "tiff" ,
                            )

    def save_citl_image_pair(self,
        measured : torch.Tensor,
        pattern  : torch.Tensor,
        target   : torch.Tensor,
        name_dataset : str, # E.g. Divk2k
        img_index : int,
        offset_index : int = 0,
        stack_index : int = 0,
        noise_index : int = 0,
        camera_position_ind : int = 0,
        file_extension = "tiff" ,
            ):
        
        name_list = ["measured", "pattern", "target"]
        object_list = [measured, pattern, target]
        
        
        if target == None:
            name_list = ["measured", "pattern"]
            object_list = [measured, pattern]
            
        
        for idx in range(len(name_list)):
            name = name_list[idx]
            object = object_list[idx]
            
            if torch.is_tensor(object):
                object = object.detach().cpu()
            object = object.squeeze()

            # save target image
            target_path     = self.get_save_path(
                name_dataset        = name_dataset,
                name_image_type     = name,
                index_list = [img_index, noise_index, offset_index, stack_index, camera_position_ind],
                name_list  = ["img", "noise", "offset", "focus", "CamPosZ"],
                integer_depth = [4,2,2,2,2],
                file_extension      = file_extension,
                base_folder = self.base_folder
                                        )
            imsave( object , target_path)
    
        
