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
import warnings

from holotorch.Material.CGH_Material import CGH_Material
from holotorch.Optical_Components.CGH_Component import CGH_Component
from holotorch.utils.Dimensions import HW
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Components.Resize_Field import Resize_Field

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

class DiffractiveOpticalElement(CGH_Component):
    """
    Hologram is an abstract class that acts as an holographic element that can interact
    with a complex wavefront.

    This Hologram class is wavelength dependent, i.e. a multi-channel tensor can be used as input
    to calculate wavelength dependent output (e.g. if the phase-delays are different for
    different wavelengths)

    
    """
    
    def __init__(self,
                doe_dimension : HW, # NOTE THIS NEEDS BE REPLACED
                thickness      : torch.Tensor,
                material       : CGH_Material,
                fixed_pattern  : bool = True,
                scale_thickness: int = 1,
                ):
        """Initializes the Hologram class

        Args:
            dx (floatortorch.Tensor): input feature size
            thickness (torch.Tensor): the thickness of the hologram at each pixel which will e.g. define phase-delay
            material (CGH_Material): A material that can be passed here
            device (_device, optional): [description]. Defaults to torch.device("cpu").
            fixed_pattern (bool, optional): If True the phase delay will not be set to an nn.parameter to be optimized for . Defaults to True.
            scale_phase (bool, optional): factor to scale phase by before applying phase to input field
            dtype (dtype, optional): [description]. Defaults to torch.double.
        """        
    
        super().__init__()

        self.doe_dimension       = doe_dimension
        
        #Set internal variables
        self.fixed_pattern  = fixed_pattern # Flag (boolean) if thickness is optimized
        self.material       = material
        self.scale_thickness= scale_thickness
               
        self.thickness = thickness                   
    
    
    @property
    def thickness(self) -> torch.Tensor:
        try:
            return self._thickness
        except AttributeError:
            return None
    
    @thickness.setter
    def thickness(self,
                  thickness : torch.Tensor
                  ) -> None:
        """ Add thickness parameter to buffer
        
        If it is parameter make it a paramter, otherwise just add to buffer/statedict

        Args:
            thickness (torch.Tensor): [description]
        """
        # The None is handling is just a check which is needed to change flags
        if thickness is None:
            thickness = self.thickness
            if thickness is None:
                return
        
        if self.thickness is not None:
            del self._thickness

        if self.fixed_pattern == True:
            self.register_buffer("_thickness", thickness)
        elif self.fixed_pattern == False:
            self.register_parameter("_thickness", torch.nn.Parameter(thickness))

    @property
    def fixed_pattern(self) ->bool:
        return self._fixed_pattern
    
    @fixed_pattern.setter
    def fixed_pattern(self, fixed_pattern) -> None:
        self._fixed_pattern = fixed_pattern
        self.thickness = None          
                  
   
    def calc_phase_shift(self, wavelengths : torch.Tensor) -> torch.Tensor:
        """Helper method to write smaller code outside of this class.
        """
        
        
        thickness = self.scale_thickness * self.thickness
        
        assert wavelengths.device == thickness.device, "WAVELENGTHS: " + str(wavelengths.device) + ", THICKNESS: " + str(thickness.device)

        # return self.material.calc_phase_shift(thickness = self.thickness, wavelengths=self.wavelengths)
        phase_shift = self.material.calc_phase_shift(
                thickness   = thickness,
                wavelengths = wavelengths
                )       
        
        return phase_shift 
        
   
    def forward(self,
            field : ElectricField,
                ) -> ElectricField:
        """  Takes in a field and applies the DOE to it


        Args:
            field (ElectricField): [description]

        Returns:
            ElectricField: [description]
        """

        wavelengths = field.wavelengths

        assert wavelengths.data_tensor.device == field.data.device, "WAVELENGTHS: " + str(wavelengths.data_tensor.device) + ", FIELD: " + str(field.data.device)

        phase_shift = self.calc_phase_shift(wavelengths=wavelengths.data_tensor)

        height = max(phase_shift.shape[-1], field.height)
        width = max(phase_shift.shape[-2], field.width)
        upsample_size = [height,width]

        # View to the correct output shape
        view_shape = [*field.shape[0:-2],*phase_shift.shape[-2:]]
        phase_shift = phase_shift.expand(view_shape)

        phase_shift = ElectricField(data = phase_shift, spacing=None, wavelengths=None)

        Up0 =  Resize_Field(
            size = upsample_size,
            mode="bicubic"
            )
        
        phase_shift = Up0(phase_shift)

        data = field.data * torch.exp(1j * phase_shift.data)        

        E_out = ElectricField(
            data = data,
            wavelengths=field.wavelengths,
            spacing = field.spacing
        )

        return E_out        

    def __str__(self, ):
        """
        Creates an output for the hologram class.
        """

        mystr = ""

        mystr += str(self.thickness.shape)
        mystr += "\n==========\n"

        mystr += "Material used:"
        mystr += "\n==========\n"

        mystr += str(self.material)

        return mystr
