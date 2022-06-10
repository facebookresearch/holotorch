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

from enum import Enum


class ENUM_SLM_TYPE(Enum):
    phase_only      = 0
    tranmissive     = 1
    complex_cart    = 2
    voltage         = 3
    complex_dpac    = 4
    complex_polar   = 5


class ENUM_HOLO_TYPE(Enum):
    phase_only      = 0
    tranmissive     = 1
    complex         = 2
    without         = -1

class ENUM_DATATYPE(Enum):
    FocalStackView  = 1
    ImageView       = 0
    LightField      = 2

class ENUM_DATASET(Enum):
    DIV2K_Dataset           = 0
    LightField_EXR_Loader   = 1
    Single_Image_Loader     = 2
    Directory_Image_Dataset = 3
    Captured_Image_Dataset  = 4

class ENUM_OPTIMIZER(Enum):
    Default     = 0
    Pupil       = 1
    Neural      = 2
    

class ENUM_SLM_INIT(Enum):
    RANDOM  = 0
    ZEROS   = 1
    ONES    = 2
    
class ENUM_HOLO_INIT(Enum):
    RANDOM  = 0
    ZEROS   = 1
    ONES    = 2

class INIT_TYPE(Enum):
    RANDOM  = 0
    ZEROS   = 1
    ONES    = 2
    
class ENUM_SOURCE_TYPE(Enum):
    COHERENT            = 0
    PARTIALLY_COHERENT  = 1

class ENUM_SENSOR_TYPE(Enum):
    MONOCHROMATIC       = 0
    BAYER               = 1
    TIME_MULTIPLEXED    = 2

class ENUM_PROP_TYPE(Enum):
    FOURIER   = 0
    ASM       = 1
    FOUR_4    = 2

class ENUM_SPECTRUM(Enum):
    STANFORD_LED  = 0
    STANFORD_SLED = 1
    NO_SPECTRUM   = 2
    PARAMETERIZED = 3

class ENUM_PROP_KERNEL_TYPE(Enum):
    PARAXIAL_KERNEL    = 0
    FULL_KERNEL        = 1
    QUADRATIC_KERNEL   = 2

class ENUM_TIME_MULTIPLEXING(Enum):
    """RGB Colors sequential or in one shot
    """   
    ONE_SHOT        = 0
    TIME_MULTIPLEX  = 1

class ENUM_TEMPORAL_COHERENCE_SAMPLER(Enum):
    LINEAR      = 0,
    UNIFORM     = 1,
    GAUSSIAN    = 2,
    CENTER_WAVELENGTHS_ONLY = 4

class ENUM_SPATIAL_COHERENCE_SAMPLER(Enum):
    PLANE_WAVES             = 1
    RANDOM_2PI              = 2
    NO_SPATIAL_COHERENCE    = 0

class ENUM_SOURCE_APERTURE_TYPE(Enum):
    DISK        = 0
    GAUSSIAN    = 1
    RECTANGLE   = 2
    

class ENUM_PLOT_TYPE(Enum):
    MAGNITUDE   = 1,
    PHASE       = 2

class ENUM_MATERIAL(Enum):
    Default     = 0 # Default is a constant refractive index
    HOLOGRAFIX  = 2
    Cauchy      = 3

class ENUM_OPT_STATE(Enum):
    DOE_TRAINING    = 0 # the default optimization state
    PUPIL_SGD       = 1
    SLM_TESTING     = 2
    CALIBRATING     = 3

class ENUM_LOSS_TYPE(Enum):
    MSE_LOSS = 0
    VGG_LOSS = 1
    
class ENUM_LOSS_TYPE(Enum):
    ZERO_VALUE_PAD  = 0
    MAX_VALUE_PAD   = 1
    
class ENUM_FILE_TYPE(Enum):
    PNG     = '.png'
    NUMPY   = '.np'
    TORCH   = '.pt'
    TIFF    = '.tiff'

class ENUM_SLM_FIELD_FRINGE_TYPE(Enum):
    SIMPLE_CONVOLUTION      = 1,
    NEURAL_NETWORK          = 2,
    NOTHING                 = 3,

class ENUM_COPY_MOVE(Enum):
    COPY = 1
    MOVE = 2

class ENUM_CITL_CAPTURE_TYPE(Enum):
    DUMMY           = 0
    REAL_CAPTURE    = 1
    PRE_CAPTURE     = 2
    
class ENUM_DPAC_ENCODING(Enum):
    HORIZONTAL   = 1
    VERTICAL     = 2
    CHECKERBOARD = 0



class CAMERA_TYPE(Enum):
    DUMMY = 0
    FLIR = 1
    XIMEA = 2

class DISPLAY_TYPE(Enum):
    SLM = 0
    DUMMY = 1

class MASK_FORWARD_TYPE(Enum):
    ADDITIVE = 0
    MULTIPLICATIVE = 1

class MASK_MODEL_TYPE(Enum):
    COMPLEX = 0
    REAL    = 1

class NORMALIZE_TYPE(Enum):
    MEAN = 0
    MAX = 1
    
class FIELD_IDENTIFIER(Enum):
    NOTHING     = 0
    MEASURED    = 1
    PATTERN     = 2
    TARGET      = 3
    