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
import numpy as np
import warnings
import matplotlib.pyplot as plt

from holotorch.utils.Dimensions import TC
from holotorch.utils.Helper_Functions import *
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Spectra.WavelengthContainer import WavelengthContainer
from holotorch.Spectra.Spectrum import Spectrum
from holotorch.LightSources.Source import Source
import holotorch.Spectra.SpacingContainer as SpacingContainer
import holotorch.utils.Dimensions as Dimensions
from holotorch.utils.Enumerators import *
from holotorch.utils.units import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class PartialCoherentSource(Source):

    
    def __init__(self,
            tensor_dimension    : Dimensions.HW,
            spectrum            : Spectrum,
            num_modes_per_center_wavelength : int,
            grid_spacing        : SpacingContainer,
            source_diameter     : float,
            focal_length_collimating_lens : float,
            spatial_coherence_sampling_type : ENUM_SPATIAL_COHERENCE_SAMPLER,
            temporal_coherence_sampling_type : ENUM_TEMPORAL_COHERENCE_SAMPLER = ENUM_TEMPORAL_COHERENCE_SAMPLER.LINEAR,
                ) -> None:
        """[summary]

        Args:
            tensor_dimension (TensorDimension): [description]
            data (torch.Tensor): [description]
            spectrum (Spectrum): [description]
            wavelengths (torch.Tensor): All wavelengths
            intensities (torch.Tensor): Intensities of the spectrum
            center_wavelengths (torch.Tensor): [description]
            sigma_wavelengths (torch.Tensor): [description]
            num_modes_per_center_wavelength (int): [description]
        """    
    
        # Get the tensor dimension for this partial coherent source
        # It is a 4D tensor since we also have height and width
        tensor_dimension = Dimensions.TCHW(
            n_time      = len(spectrum.center_wavelengths),
            n_channel   = 1,
            height      = tensor_dimension.height,
            width       = tensor_dimension.width,
        )        

        super().__init__(
            tensor_dimension    = tensor_dimension,
            wavelengths         = None,
            grid_spacing        = grid_spacing,
        )
        
        assert spectrum is not None, "Spectrum cannot be None for Partial Coherent Source"
        self.spectrum = spectrum
        
        # Store sampling types as member variables
        self.spatial_coherence_sampling_type = spatial_coherence_sampling_type
        self.temporal_coherence_sampling_type = temporal_coherence_sampling_type

        # Parameter for TEMPORAL coherence
        self.spectrum = spectrum
        self.num_modes_per_center_wavelength = num_modes_per_center_wavelength
        
        # Parameter for SPATIAL coherence
        self.source_diameter = source_diameter
        self.focal_length_collimating_lens = focal_length_collimating_lens
        

        
        # Resample the partial coherent source
        # NOTE: This does the following
        # 1. Resample wavelengths (and get the spectral intensities from the Spektrum)
        # 2. Resample 
        # 3. Recomputes the complete source from scratch
        self.resample_partial_coherence_source()


    @property
    def spectrum(self) -> Spectrum:
        return self._spectrum
    
    @spectrum.setter
    def spectrum(self, spectrum : Spectrum) -> None:
        self._spectrum = spectrum


    def __str__(self) -> str:
        
        mystr = ""
        mystr += "\n-------------------------------------------------------------"
        mystr += "\n" + "PARTIAL COHERENT SOURCE"
        mystr += "\n" + "CENE"
        mystr += "\n" + "PARTIAL COHERENT SOURCE"


    
        return mystr
    

    def __rep__(self) -> str:
        
        mystr = self.__str__()
        mystr += "\n-------------------------------------------------------------\n"
    
        return mystr

    def resample_partial_coherence_source(self,
                temporal_sampling_type : ENUM_TEMPORAL_COHERENCE_SAMPLER = None,
                spatial_sampling_type : ENUM_SPATIAL_COHERENCE_SAMPLER = None,
                num_modes_per_center_wavelength = None,
                spatial_frequencies = None
                ):

        
        if temporal_sampling_type == ENUM_TEMPORAL_COHERENCE_SAMPLER.CENTER_WAVELENGTHS_ONLY:
            assert(num_modes_per_center_wavelength == 1 or num_modes_per_center_wavelength == None ,
                "If we sample only the center wavelength num_modes_per_center_wavelength needs to be one")
            num_modes_per_center_wavelength = 1
            
        if num_modes_per_center_wavelength is None:
            num_modes_per_center_wavelength = self.num_modes_per_center_wavelength
        else:
            self.num_modes_per_center_wavelength = num_modes_per_center_wavelength

            
            
        #################################################################
        # Sample the wavelengths used for temporal coherence
        #################################################################
        self.resample_temporal_coherence(
            sampling_type = temporal_sampling_type,
            num_modes_per_center_wavelength = num_modes_per_center_wavelength
        )
    

        
        #################################################################
        # Sample SPATIAL PARTIAL COHERENCE with tilted plane waves
        #################################################################
        self.resample_spatial_coherence(
            sampling_type = spatial_sampling_type,
            spatial_frequencies = spatial_frequencies
        )

    
    
    def resample_temporal_coherence(self,
                sampling_type : ENUM_TEMPORAL_COHERENCE_SAMPLER = None,
                num_modes_per_center_wavelength = None,
                ):
        """ Resamples the wavelengths vectors

        Args:
            sampling_type (ENUM_TEMPORAL_COHERENCE_SAMPLER, optional): [description]. Defaults to None.
        """
        
        with torch.no_grad():
            
            if sampling_type is None:
                sampling_type = self.temporal_coherence_sampling_type
            
            # Fist we randomly sample different wavelengths from the source spectrum
            # E.g. for a center wavelength of 550 and 5 modes
            # we could sample 549.5,548.5,551.5,552.3 and 550.0
            # NOTE: We're doing this for every center wavelengths, i.e.
            # in the end we have tensor like [T,C] where T denotes t denotes
            # the number of center wavelengths and C the number of modes per channel

            if num_modes_per_center_wavelength is None:
                num_modes_per_center_wavelength = self.num_modes_per_center_wavelength
            
            self.num_modes_per_center_wavelength = num_modes_per_center_wavelength
                
            temporal_modes =  PartialCoherentSource.sample_temporal_modes(
                center_wavelengths = self.center_wavelengths,
                sigma_wavelengths  = self.sigma_wavelengths,
                num_modes_per_center_wavelength = num_modes_per_center_wavelength,
                sampling_type   = sampling_type
            )


            self.wavelengths = WavelengthContainer(
                wavelengths = temporal_modes,
                tensor_dimension = TC(n_time=temporal_modes.shape[0],n_channel=temporal_modes.shape[1]),
                center_wavelength = self.center_wavelengths,
            )

            #################################################################
            # Resample the spectral irradiance
            #################################################################
            self.resample_spectral_irradiance()
        
        # # Either set the wavelength container or write it into its memory
        # if self.wavelengths is None:
        #     # Next we create the wavelength container since we have [T x C] different wavvelengths
        #     # Sampled
        #     self.wavelengths = WavelengthContainer(
        #         wavelengths=temporal_modes,
        #         tensor_dimension= TC(n_time=temporal_modes.shape[0],n_channel=temporal_modes.shape[1])
        #     )
        # else:
        #     self.wavelengths.write_data_tensor(temporal_modes)
        
    def resample_spectral_irradiance(self) -> torch.Tensor:
        """Resamples the source irradiance from the spectrum
        """     
        
        with torch.no_grad():   
            center_wavelengths = self.center_wavelengths

            #################################################################
            # Get the irradiance from the spectrum
            #################################################################  
            # This is just a way to make the source_idx a tensor
            # Source IDX could be 1 for monochromatic and 3 for RGB

            if center_wavelengths.shape[0] == 1:
                source_idx = torch.tensor([0]) 
            elif center_wavelengths.shape[0] == 3:
                source_idx = torch.arange(0,3,1)
                
            # Get the irradiance values of the spectrum for the chosen wavelengths
            # and for each source (e.g. for RGB we have 3 different sources from 3 LEDs)
            self.source_irradiance = self.spectrum.get(
                wavelengths = self.wavelengths.data_tensor,
                source_idx = source_idx
                )  
                
    def resample_spatial_coherence(self,
                sampling_type : ENUM_SPATIAL_COHERENCE_SAMPLER = None,
                spatial_frequencies = None,
                source_diameter : float = None,
                aperture_type : ENUM_SOURCE_APERTURE_TYPE = ENUM_SOURCE_APERTURE_TYPE.GAUSSIAN
                ):
        """[summary]
        """
        if sampling_type is None:
            sampling_type = self.spatial_coherence_sampling_type
            
        if source_diameter is None:
            source_diameter = self.source_diameter    
            
        if sampling_type is ENUM_SPATIAL_COHERENCE_SAMPLER.PLANE_WAVES:
       
            source_out = PartialCoherentSource.create_tilted_plane_waves(
                        source_diameter               = source_diameter,
                        N_spatial_modes               = self.num_modes_per_center_wavelength,
                        grid_spacing                  = self.grid_spacing,
                        tensor_dimension              = self.tensor_dimension,
                        focal_length_collimating_lens = self.focal_length_collimating_lens,
                        wavelenghts_container         = self.wavelengths,
                        spatial_frequencies           = spatial_frequencies
                        )
        elif sampling_type is ENUM_SPATIAL_COHERENCE_SAMPLER.RANDOM_2PI:
            
            source_out = self.compute_partial_coherence_from_speckle_field(
                source_diameter = source_diameter,
                aperture_type = aperture_type
            )
    
        self.spatial_coherence_field = source_out
            
 
    
    @staticmethod
    def sample_temporal_modes(
            center_wavelengths : torch.Tensor,
            sigma_wavelengths  : torch.Tensor,
            num_modes_per_center_wavelength : int,
            sampling_type : ENUM_TEMPORAL_COHERENCE_SAMPLER = ENUM_TEMPORAL_COHERENCE_SAMPLER.LINEAR
            
        ) -> torch.Tensor:
        
        if sampling_type == ENUM_TEMPORAL_COHERENCE_SAMPLER.CENTER_WAVELENGTHS_ONLY:
            if num_modes_per_center_wavelength != 1:
                raise ValueError("If we sample only the center wavelength num_modes_per_center_wavelength needs to be one")

        num_center_wavelengths = len(center_wavelengths)
        
            
        
        wavelengths = torch.zeros(num_center_wavelengths, num_modes_per_center_wavelength)
        
        # Loop through the wavelenths
        for idx, center_lambda in enumerate(center_wavelengths):
            fwhm_sigma = sigma_wavelengths[idx]
            
            bandwidth_scale = 3
            wavelength_left = center_lambda - bandwidth_scale*fwhm_sigma
            wavelength_right = center_lambda + bandwidth_scale*fwhm_sigma
            
            bandwidth = 2*bandwidth_scale*fwhm_sigma

            if sampling_type == ENUM_TEMPORAL_COHERENCE_SAMPLER.LINEAR:
                if num_modes_per_center_wavelength == 1:
                    sub_lambda = 0 # Special case where length is one to be centered
                else:
                    sub_lambda = bandwidth * torch.linspace(-0.5,0.5, num_modes_per_center_wavelength)

                sub_wavelengths = center_lambda + sub_lambda
            
            elif sampling_type == ENUM_TEMPORAL_COHERENCE_SAMPLER.UNIFORM:
                sub_wavelengths =  center_lambda + 2 * bandwidth_scale * fwhm_sigma * (torch.rand(num_modes_per_center_wavelength) - 0.5)
            
            elif sampling_type == ENUM_TEMPORAL_COHERENCE_SAMPLER.GAUSSIAN:
                sub_wavelengths =  center_lambda + torch.randn(num_modes_per_center_wavelength) * 0.5*fwhm_sigma  
            
            elif sampling_type == ENUM_TEMPORAL_COHERENCE_SAMPLER.CENTER_WAVELENGTHS_ONLY:
                sub_wavelengths = center_lambda
            else:
                raise NotImplementedError("This sampling type is not implemented.")
            
            wavelengths[idx] = sub_wavelengths

        return wavelengths
    

    @property
    def center_wavelengths(self):
        return self.spectrum.center_wavelengths
    
    @center_wavelengths.setter
    def center_wavelengths(self, val):
        raise NotImplementedError("Center wavelengths cannot be set inside source")
        
    @property
    def sigma_wavelengths(self):
        return self.spectrum.sigma_wavelengths
    
    @sigma_wavelengths.setter
    def sigma_wavelengths(self, val):
        raise NotImplementedError("Center wavelengths cannot be set inside source")
        

    # @property
    # def sigma_wavelengths(self) -> torch.Tensor:
    #     return self._sigma_wavelengths

    # @sigma_wavelengths.setter
    # def sigma_wavelengths(self, sigma_wavelengths = torch.tensor) -> None:
    #     if sigma_wavelengths is not torch.Tensor:
    #         sigma_wavelengths = torch.tensor([sigma_wavelengths])
    #     self.register_buffer("_sigma_wavelengths", sigma_wavelengths)

    # @property
    # def center_wavelengths(self) -> torch.Tensor: 
    #     return self._center_wavelengths
    
    # @center_wavelengths.setter
    # def center_wavelengths(self, center_wavelengths = torch.tensor) -> None:
    #     self.register_buffer("_center_wavelengths", center_wavelengths)

        
    @property
    def num_center_wavelengths(self) -> int:
        """Returns the number of center wavelengths

        Returns:
            int: [description]
        """        
        return len(self.center_wavelengths)
 

    @property
    def num_modes_per_center_wavelength(self) -> int:
        """Return the number of channels per center wavelength

        Raises:
            NotImplementedError: [description]

        Returns:
            int: [description]
        """        
        return self._num_modes_per_center_wavelength
    
    @num_modes_per_center_wavelength.setter
    def num_modes_per_center_wavelength(self, val : int) -> None:
        self._num_modes_per_center_wavelength = val
    
    @staticmethod
    def compute_points_inside_circle(
                    N_points = 5,
                    R = 1,
                    center_x = 0,
                    center_y = 0
                    ):
        """[summary]

        Args:
            N_points (int, optional): [description]. Defaults to 5.
            R (int, optional): [description]. Defaults to 1.
            center_x (int, optional): [description]. Defaults to 0.
            center_y (int, optional): [description]. Defaults to 0.

        Returns:
            x,y : [description]
        """        
        
        r = R * torch.sqrt(torch.rand(N_points))
        theta = torch.rand(N_points) * 2 * torch.pi
        
        x = center_x + r * torch.cos(theta)
        y = center_y + r * torch.sin(theta)
        
        return x,y


    
    @staticmethod
    def create_tilted_plane_waves(
                    source_diameter : torch.Tensor,
                    N_spatial_modes : int,
                    grid_spacing    : SpacingContainer,
                    tensor_dimension,
                    focal_length_collimating_lens : float,
                    wavelenghts_container : WavelengthContainer,
                    source_amp_sigma = 0,
                    spatial_frequencies : torch.Tensor = None
                    ) -> torch.Tensor:
        
        dx, dy = grid_spacing.get_spacing_xy()
        dx = dx.squeeze()
        dy = dy.squeeze()
        
        num_pixel_x = tensor_dimension.height
        num_pixel_y = tensor_dimension.width
        
        wavelengths = wavelenghts_container.data_tensor
        # maximum incident angle from the edge of the source pinhole (in angular frequency, rad)
        # w_max and sigma_w are of shape [TC] as the wavelength container always is
        w_max = 2 * np.pi /  wavelengths * source_diameter / 2 / focal_length_collimating_lens
        #sigma_w = 2 * np.pi / wavelengths * source_amp_sigma / focal_length_collimating_lens
        
        x = torch.linspace(-dx * num_pixel_x / 2,
                        dx * num_pixel_x / 2,
                        num_pixel_x)
        y = torch.linspace(-dy * num_pixel_y / 2,
                        dy * num_pixel_y / 2,
                        num_pixel_y)
        
        X, Y = torch.meshgrid(x, y)
        
        if spatial_frequencies is None:
            
            wx, wy = PartialCoherentSource.compute_points_inside_circle(
                        N_points = N_spatial_modes,
                        R = w_max
                    )
        else:
            # Make spatial_frequencies a tensor
            if not torch.is_tensor(spatial_frequencies):
                spatial_frequencies = torch.tensor(spatial_frequencies)
                
            if spatial_frequencies.ndim == 1:
                spatial_frequencies = spatial_frequencies[None]
            
            if spatial_frequencies.ndim == 2:
                spatial_frequencies = spatial_frequencies[None]
                
            wx = spatial_frequencies[:,:,0]
            wy = spatial_frequencies[:,:,1]    
        
        phase = wx[:,:,None,None]*X[None,None,:,:] + wy[:,:,None,None]*Y[None,None,:,:]
        
        phase = torch.exp(1j * phase)
        
        return phase
        
    @staticmethod
    def compute_coherence_length(refractive_index = 1,
                                 wavelength = 660*nm,
                                 bandwidth = 10*nm
                                 ):
        refractive_index = torch.tensor(refractive_index)
        return 2 * np.log(2) / (torch.pi * refractive_index) * wavelength**2 / bandwidth

    @staticmethod
    def gaussian_kernel(R, sigma = 0.1, mu = [0,0]):
        """Creates a gaussian kernel

        Args:
            R ([type]): [description]
            sigma (float, optional): [description]. Defaults to 0.1.
            mu (list, optional): [description]. Defaults to [0,0].

        Returns:
            [type]: [description]
        """    
        
        gaussian = 1/np.sqrt(2 * torch.pi ) * torch.exp( -0.5 * R**2 / sigma**2 )
        return gaussian


    @staticmethod
    def get_source_profile(
        R : torch.Tensor,
        source_diameter : float,
        aperture_type : ENUM_SOURCE_APERTURE_TYPE = ENUM_SOURCE_APERTURE_TYPE.GAUSSIAN
        ):
        source = torch.zeros(R.shape)

        if aperture_type is ENUM_SOURCE_APERTURE_TYPE.DISK:
        
            source[R < source_diameter] = 1
        
        elif aperture_type is ENUM_SOURCE_APERTURE_TYPE.GAUSSIAN:
            
            source = PartialCoherentSource.gaussian_kernel(R, sigma = source_diameter)
        return source
    
    def compute_partial_coherence_from_speckle_field(self,
            source_diameter : float,
            aperture_type : ENUM_SOURCE_APERTURE_TYPE = ENUM_SOURCE_APERTURE_TYPE.GAUSSIAN
                ):
        
        f = self.focal_length_collimating_lens
        # Define Coordinate systems
        N_x = self.tensor_dimension.height
        N_y = self.tensor_dimension.width
        dx = self.grid_spacing.data_tensor[:,:,0].squeeze()
        dy = self.grid_spacing.data_tensor[:,:,1].squeeze()

        x = torch.linspace(- 0.5, 0.5 ,N_x)
        y = torch.linspace(- 0.5, 0.5 ,N_x)

        x = N_x * dx * x
        y = N_y * dy * y
        
        X,Y = torch.meshgrid(x,y)
        R = torch.sqrt(X**2 + Y**2)
        
        # Chose Focal Length so that dx == dx_output
        focal_length = dx**2 * N_x /  self.wavelengths.data_tensor    
        source = PartialCoherentSource.compute_speckle_field(
            R = R,
            coherence_diameter=1,
            source_diameter=source_diameter,
            aperture_type = aperture_type,
            dx = dx,
            N_modes = self.num_modes_per_center_wavelength
        )
        
        return source

    @staticmethod
    def compute_speckle_field(
        R : torch.Tensor,
        coherence_diameter : float,
        source_diameter : float,
        dx : float,
        N_modes = 1,
        aperture_type : ENUM_SOURCE_APERTURE_TYPE = ENUM_SOURCE_APERTURE_TYPE.GAUSSIAN
        ):

        source_profile = PartialCoherentSource.get_source_profile(
                    R = R,
                    source_diameter=source_diameter,
                    aperture_type=aperture_type
                    )

        new_shape = [N_modes, *source_profile.shape]
        rand_phase = 2*torch.pi * torch.rand(new_shape)
        rand_phase = rand_phase[None,:]
        
        coherence_diameter_pixel = coherence_diameter / dx
        
        # kernel_size = int(4*np.round(coherence_diameter_pixel)) + 1
        # kernel_size = max(kernel_size, 3) # Make the kernel size at least 3
        # kernel_size = [kernel_size, kernel_size]
        # sigma = [coherence_diameter_pixel,coherence_diameter_pixel]
        
        field = source_profile[None, None, :,:] * torch.exp(1j*rand_phase)
        
        # filter = kornia.filters.GaussianBlur2d(
        #     kernel_size = kernel_size,
        #     sigma = sigma,
        #     border_type='reflect',                       
        #     )
        
        #field = filter(field.real) + 1j * filter(field.imag)
        
        field = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field)))
                
        return field


                
        
    @property
    def coherence_size(self):
        return self._coherence_size
   
        
    @coherence_size.setter
    def coherence_size(self, size):
        self._coherence_size = size
        
    
    def forward(self,
        flag_resample_coherence : bool = False,
        channel_idxs : torch.Tensor = None
        ) -> ElectricField:
        
        if flag_resample_coherence is True:
            self.resample_partial_coherence_source()
        
        gmm_out = super().forward().data
        
        # Add singleton dimensions for time, height and width
        # TODO: This assume TCHW format for now  (could be more flexible in future)
        if channel_idxs is None:
            source_irradiance = self.source_irradiance[:,:,None,None]
        else:
            source_irradiance = self.source_irradiance[:,channel_idxs,None,None]
            # If we loose the dimension cause of the channel_idxs_crop we need to broadcast again
            if source_irradiance.ndim == 3:
                source_irradiance = source_irradiance[:,None]
                
        # Expand source to a BTPCHW (6D) object        
        source_irradiance = source_irradiance.expand(-1, -1,self.tensor_dimension.height,self.tensor_dimension.width)
        
        # NOTE: For now its phase only
        if channel_idxs is None:
            spatial_coherence_field = self.spatial_coherence_field
        else:
            spatial_coherence_field = self.spatial_coherence_field[:,channel_idxs]
            # If we lose the dimension we need to broadcast again
            if spatial_coherence_field.ndim < self.spatial_coherence_field.ndim:
                spatial_coherence_field = spatial_coherence_field[:,None]
            
        out = source_irradiance * spatial_coherence_field
        
        out = out[None,:,None,:] * gmm_out

        if channel_idxs is None:
            wavelengths = self.wavelengths
        else:
            # We also need to extract the correct slice for the wavelengths
            new_wavelengths = self.wavelengths.data_tensor[:,channel_idxs]     
            # Broadcast if dimension has changed
            if new_wavelengths.ndim < self.wavelengths.data_tensor.ndim:
                new_wavelengths = new_wavelengths[:,None]  
            
            tensor_dimension = TC(n_time = new_wavelengths.shape[0], n_channel= new_wavelengths.shape[1])
            wavelengths = WavelengthContainer(
                wavelengths=new_wavelengths,
                tensor_dimension=tensor_dimension,
                center_wavelength=self.center_wavelengths
                )
                    
        out = ElectricField(
                data = out,
                wavelengths = wavelengths,
                spacing = self.grid_spacing
                )
        
        return out     

    def visualize_temporal_sampling(self):
        plt.figure(figsize=(15,5))

        plt.grid(linestyle='--', alpha = 0.5)
        plt.plot(self.wavelengths.data_tensor.cpu().squeeze()/nm,
                self.source_irradiance.cpu().squeeze(),'.'
                )
        plt.xlabel("Wavelengths [nm]")
        plt.ylabel("Spectral Intensity [a.u.]")