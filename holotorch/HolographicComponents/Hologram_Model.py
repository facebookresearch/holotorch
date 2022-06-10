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
import os
import matplotlib.pyplot as plt
import numpy as np

from holotorch.utils.Dimensions import HW
from holotorch.utils.tictoc import *
from holotorch.utils.Enumerators import *
from holotorch.Optical_Components.DiffractiveOpticalElement import DiffractiveOpticalElement
from holotorch.Optical_Components.Resize_Field import Resize_Field
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Material.CGH_Material import CGH_Material
import holotorch.utils.Visualization_Helper as VH
import holotorch.utils.Helper_Functions as HF
from holotorch.utils.units import *

class Hologram_Model(CGH_Component):
    
    def __init__(self,  
                        center_wavelength : float, # Source is needed for initialization
                        hologram_dimension : HW,
                        feature_size : float,
                        material : CGH_Material,
                        init_type : ENUM_HOLO_INIT, 
                        holo_type : ENUM_HOLO_TYPE, 
                        FLAG_OPTIMIZE_EXPANDER : bool = True
        ):
        """ Create a static hologram for optimizing.

        Args:
            holo_sz ([type]): 4D array of tensor dimensions D x C x H x W
            material (CGH_Material): the material object for the hologram 
            init_type (ENUM_HOLO_INIT): initialize method for hologram (e.g. random values or ones)
            holo_type ([type], optional): phase only / amplitude / complex. Defaults to ENUM_SLM_TYPE.phase_only.
            FLAG_OPTIMIZE_EXPANDER (bool, optional): FLAG if the expander should be optimized. Defaults to False.
        """

        super().__init__()
        
        self.center_wavelength = center_wavelength # just needed for initialization

        self.hologram_dimension = hologram_dimension

        self.FLAG_OPTIMIZE_EXPANDER = FLAG_OPTIMIZE_EXPANDER
        self.holo_type = holo_type
        self.init_type = init_type
        self.material = material

        self.thickness_scale = 1
        
        self.feature_size = feature_size

        self.__init_model__()

    def _init_dimensions(self):
       
        pass


    @staticmethod
    def dimensions():
        mystr = ""
        mystr += "\n" + "=====================================" 
        mystr += "\n" + "Hologram Object (4D): D x C x H x W"   
        mystr += "\n" + "=====================================" 
        mystr += "\n" + "Input Field Dimensions (5D): B x T x C x H x W."  
        mystr += "\n" + "=====================================" 
        mystr += "\n" + "Output Field Dimensions: B x T x C x H x W." 
        mystr += "\n" + "=====================================" 
        mystr += "\n" +  "B: batch" 
        mystr += "\n" +  "T: time"
        mystr += "\n" +  "D: depth / thickness of physical layer" 
        mystr += "\n" +  "C: color / wavelength" 
        mystr += "\n" +  "H: Hologram height (in pixel)" 
        mystr += "\n" +  "W: Hologram width (in pixe)"

        print(mystr)

    @property
    def FLAG_OPTIMIZE_EXPANDER(self) -> bool:
        return self._FLAG_OPTIMIZE_EXPANDER
    
    @FLAG_OPTIMIZE_EXPANDER.setter
    def FLAG_OPTIMIZE_EXPANDER(self, FLAG_OPTIMIZE_EXPANDER : bool ) -> None:
        self._FLAG_OPTIMIZE_EXPANDER = FLAG_OPTIMIZE_EXPANDER
        
        try:
            self.hoe1.fixed_pattern = not FLAG_OPTIMIZE_EXPANDER
        except AttributeError:
            pass

        try:
            self.A1.fixed_pattern = not FLAG_OPTIMIZE_EXPANDER
        except AttributeError:
            pass

    def __init_model__(self):
        '''
        Initializes the SLM model

        Outline:
            1. Low-Resolution SLM
                a. Initialization (phase-only,taransmission,complex)
                b. Assembly as nn.sequential for torch-graph        
        '''

        # ----------------------------------------------------------------------------
        # upsample field to hologram/expander resolution
        # ----------------------------------------------------------------------------      
        
        upsample_size = [ self.hologram_dimension.height, self.hologram_dimension.width]

        self.Up0 =  Resize_Field(
            size = upsample_size,
            mode="bicubic"
            )

        # ----------------------------------------------------------------------------
        # create the hologram/expander model
        # ----------------------------------------------------------------------------      
        self.__init__hologram()

    
    @staticmethod
    def compute_thickness_scale(
                center_wavelength : float,
                ) -> float:
        
        # The hologram thickness needs to be scaled so that I corresponds roughly to only a small phase-shift
        # thickness = thickness * 5 *um

        # scale so that center wavelength produces 2*pi phase shift
        n = 1.5 # approximate index of refraction
        thickness_scale = 2* center_wavelength / (n)
        
        return thickness_scale
    
    def __init__hologram(self):
        """
        
        Initializes the hologram
        
        """
        if self.init_type == ENUM_HOLO_INIT.RANDOM:
            thickness = torch.rand(self.hologram_dimension.shape)
        elif self.init_type == ENUM_HOLO_INIT.ONES:
            thickness = torch.ones(self.hologram_dimension.shape)
        elif self.init_type == ENUM_HOLO_INIT.ZEROS:     
            thickness = torch.zeros(self.hologram_dimension.shape)
        else:
            raise NotImplementedError("Not yet implemented")

        thickness_scale = Hologram_Model.compute_thickness_scale(self.center_wavelength)

        # Create the optical hologram
        #
        #     
        
        if self.holo_type is not ENUM_HOLO_TYPE.without:
            
            hoe1 = DiffractiveOpticalElement(   
                                doe_dimension       =   self.hologram_dimension, # NOTE THIS NEEDS BE REPLACED
                                thickness           =   thickness,
                                material            =   self.material,
                                fixed_pattern       =   False,
                                scale_thickness     =   thickness_scale
                                )

            self.hoe1 = hoe1
        
        else: 
            # If we don't have an expander just have the idendity operator in here.
            self.hoe1 = torch.nn.Identity()
        
        # ----------------------------------------------------------------------------
        # add an absorption mask to the HOE for a complex expander
        # ----------------------------------------------------------------------------

        if self.holo_type == ENUM_HOLO_TYPE.complex: # add an absorption mask
                transmission_mask = torch.rand(self.hologram_dimension.shape).double()
                # self.A1 = Absorption_Mask(
                #     transmission = transmission_mask
                #     )
        


    def forward(self,
            field : ElectricField,
            ) -> ElectricField:
        """ Takes an input field and passes it through the hologram

        Args:
            field (ElectricField): Complex field 6D tensor object 

        Returns:
            ElectricField: The electric field after the hologram model
        """
        """
        
        
        
        Parameters
        ==========
        field           : torch.complex128
                           - batch x time x color x height x width.

        Output
        ==========
        Eout            : torch.complex128
                           Complex field 6D tensor with dims - batch x time x color x height x width.


        """
        height = max(self.hologram_dimension.height, field.height)
        width = max(self.hologram_dimension.width, field.width)
        upsample_size = [height,width]

        Up0 =  Resize_Field(
            size = upsample_size,
            mode="bicubic"
            )
        
        E_out = Up0(field)
             
        E_out = self.hoe1(E_out)
        
        # Apply the absorption mask
        if self.holo_type == ENUM_HOLO_TYPE.complex: # add an absorption mask
            E_out = self.A1(E_out)

        return E_out       

    def save_expander(self, path):
        torch.save(self.state_dict(), path)

    
    def load_expander(self, path):
        self.load_state_dict(torch.load(path))

    def __save_expander_Logging(self, save_id = None, current_it_number : int = 0):
        """
        
        """

        expander_filename = "expander_" + str(current_it_number).zfill(4)

        if save_id is not None:
            expander_filename += "_saveid_" + str(save_id) 
        
        expander_filename += ".pt"

        expander_path = self.models_dir / expander_filename

        tic()
        #torch.save(self.holo_model.state_dict(), expander_path)
        time_passed = toc()

        time_passed =  "( " + str(format(time_passed, '.3f')) + " s)"

        self.write_log('Save expander at It: ' + str(self.current_it_number).zfill(4) + "      " + time_passed )

    def __load_expander_logging(self, PATH):
        """
        
        """
        expander_state_dict = torch.load(PATH)
        #self.holo_model.load_state_dict(expander_state_dict)


    def view_expander(self, save_id = None, current_it_number = None):
        """
        
        """

        if self.holo_type is ENUM_HOLO_TYPE.without:
            print("Ths model does not have an Expander-like object since ENUM_HOLO_TYPE was set to without")
            return

        tic()
        amp = torch.ones_like(self.hoe1.thickness)
        phase = torch.zeros(amp.shape)

        if self.holo_type is ENUM_HOLO_TYPE.phase_only or self.holo_type is ENUM_HOLO_TYPE.complex: # phase or complex only
            wavelengths = torch.tensor([530*nm], device = self.hoe1.thickness.device)
            phase = self.hoe1.calc_phase_shift(wavelengths=wavelengths)
            
        if self.holo_type is ENUM_HOLO_TYPE.tranmissive or self.holo_type is ENUM_HOLO_TYPE.complex: # amplitude or complex only
            amp = self.A1.transmission.clamp(0,1)
        
        assert isinstance(self.holo_type,ENUM_HOLO_TYPE), 'SELF.HOLOTYPE needs to be contained in ENUM_HOLO_TYPE. Please check'

        
        holo = torch.squeeze(amp*torch.exp(1j*phase))
        
        holo = holo / holo.abs().mean()

        if holo.ndim == 3:
            holo = holo[0]
    
        
        virt_freq = self.compute_virtual_frequency(holo)

        img_holo_rgb = VH.colorize(holo) # Note this will already by a numpy array after this function

        if holo.ndim == 3:
            holo = holo.permute(1,2,0)
            virt_freq = virt_freq.permute(1,2,0)

        if virt_freq.ndim == 3:
            virt_freq = virt_freq.permute(1,2,0)

        virt_freq = virt_freq.cpu().detach()

        # Create the figure and axes grid
        plt.figure(figsize=(20,4))

        idx_plots = 1
        col_plots = 3
        row_plots = 1

        # PLOT: Expander
        plt.subplot(row_plots,col_plots,idx_plots); idx_plots += 1
        im_mappable = plt.imshow(img_holo_rgb); 
        VH.add_colorbar(im_mappable)
        plt.title('Expander (HSV space')
        
        # # PLOT: Expander phase
        # plt.subplot(row_plots,col_plots,idx_plots); idx_plots += 1
        # im_mappable = plt.imshow(holo.angle())
        # VH.add_colorbar(im_mappable)
        # plt.title('expander phase')

        # # PLOT: Expander Amplitude
        # plt.subplot(row_plots,col_plots,idx_plots); idx_plots += 1
        # im_mappable = plt.imshow(holo.abs(), cmap="gray", vmin=0, vmax=1) 
        # VH.add_colorbar(im_mappable)
        # plt.title('expander amplitude')

        # PLOT: Virtual Frequency
        plt.subplot(row_plots,col_plots,idx_plots); idx_plots += 1
        im_mappable = plt.imshow(virt_freq, cmap="gray"); # D.C. is much greater than 0!!!

        # im_mappable = plt.imshow(virt_freq, vmin=-2, vmax=4, cmap="hot"); # D.C. is much greater than 0!!!
        VH.add_colorbar(im_mappable)
        plt.title('virtual frequency (log)')

        # PLOT: Virtual Frequency Center slice
        plt.subplot(row_plots,col_plots,idx_plots); idx_plots += 1
        im_mappable = plt.plot(virt_freq[self.hologram_dimension.height//2,:]); 
        plt.title('virtual frequency (center slice)')

        plt.tight_layout()



    def write_stuff(self, current_it_number = None, save_id = None):

        filename = "Expander_IT_" + str(current_it_number).zfill(4)

        if save_id is not None:
            filename += "_saveid_" + str(save_id).zfill(4)

        self.write_log("Save Expander image")

        path = self.figure_dir / filename  
        path = path.with_suffix('.png')

        if os.path.isfile(path):
            os.remove(path)   # Opt.: os.system("rm "+strFile)
        plt.savefig(path, format="png")

        if self.FLAG_HEADLESS == False:
            plt.show()
        
        plt.close()


        time_passed = toc()

        self.write_log("Visualize Expander" + "    ( " + str(format(time_passed, '.3f')) + "s)")


    @staticmethod
    def compute_virtual_frequency(holo : torch.Tensor) -> torch.Tensor:
        """Computes virtual frequency as defined in Neural Expansion paper

        Args:
            holo ([torch.Tensor]): 2D tensor (complex or real) 

        Returns:
            [torch.Tensor]: Virtual Frequency
        """        
       
        holo = holo / holo.abs().mean()
        #holo_squared = HF.ft2(holo).abs().pow(2)
        #virt_freq =  HF.ft2(holo_squared).abs() # note log scale

    
        tmp = torch.abs(HF.ft2(holo)) ** 2
        tmp = torch.abs(HF.ft2(tmp))
        tmp = tmp / tmp.max()
        
        quant = torch.quantile(tmp,0.9999)
        
        tmp[tmp > quant] = quant
        
        tmp2 = tmp.flatten()
        tmp2 = tmp2[tmp2 < quant] 
        
        n_bins = 100
        x_range = np.linspace(0, quant.cpu().detach() * 1000, n_bins)
        out = torch.histc(tmp2, bins=n_bins)
        out = out / out.sum()

        plt.figure()
        plt.plot(x_range, out.cpu().detach())
        plt.xlabel("X-range [promille of max]")
        plt.ylabel("Histogram")

        plt.title("Histogram of Virtual Frequency")
        plt.show()
        
        #
        
        
        return tmp



    def __load_expander_logging(self,expander):
        self.holo_model.hoe


