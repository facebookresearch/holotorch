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

from __future__ import annotations

from enum import Enum

import torch
from torch.types import _size

import numpy as np

class DIM(Enum):
    
    TIME    =   "T"
    CHANNEL =   "C"
    HEIGHT  =   "H"
    WIDTH   =   "W"
    PUPIL   =   "P"
    BATCH   =   "B"
    EXTRA   =   "E"
    
class UNITS(Enum):
    
    METER       =   "m"
    SECONDS     =   "s"
    UNITLESS    =   ""



class IdLoc():
    _batch = None
    _time = None
    _pupil = None
    _channel = None
    _height = None
    _width = None
    
    def __init__(self) -> None:
        pass
        
class Units_Spacing_Container():
    batch_spacing      =   None # unitless
    time_spacing       =   None # s
    pupil_spacing      =   None # unitless
    channel_spacing    =   None # m
    height_spacing     =   None # m
    width_spacing      =   None # m
    
    def __init__(self) -> None:
        pass
class TensorDimension():
    
    _shape = None
    id = None
        
    _id_loc = None
    _units = None
        
    def __init__(self,
                 ) -> None:
        
        self._id_loc = IdLoc()
        self._units_spacing = Units_Spacing_Container()
               
        for idx, dim in enumerate(self.id):
                
            if dim is DIM.TIME:
                self.idx_time = idx
            elif dim is DIM.CHANNEL:
                self.idx_channel = idx
            elif dim is DIM.PUPIL:
                self.idx_pupil = idx
            elif dim is DIM.HEIGHT:
                self.idx_height = idx
            elif dim is DIM.WIDTH:
                self.idx_width = idx
            elif dim is DIM.BATCH:
                self.idx_batch = idx      
            elif dim is DIM.EXTRA:
                self.idx_extra = idx
            else:
                raise ValueError("This dimensions doesn't exist. Please check self.id")              

        

    def __str__(self) -> str:
        mystr = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        mystr += "Tensor Dimension Type: " + type(self).__name__ + " (NOTE: This should be the same as Identifier)"
        mystr += "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        mystr += "Shape: " + str(self.shape)
        mystr += "\n"
        mystr += "ID-Classifier: " + str(self.id)
        mystr += "\n"
        mystr += "-----------------------------------------------------\n"
        mystr += "Units: " + str(self.units)
        mystr += "\n"
        #mystr += "Height Spacing = " + str(self.height_spacing / um) 
        #mystr += "  |  Width Spacing = " + str(self.width_spacing / um)

        # TODO: possibly include other spacings
        return mystr
    
    def __repr__(self) -> str:
        return self.__str__()

    def get_new_shape(self, new_dim : TensorDimension):
        """[summary]

        Args:
            dim (TensorDimension): [description]
            new_dim (TensorDimension): [description]
        """
        new_shape = torch.ones(len(new_dim.id),dtype=int)

        idx_changes = np.isin(new_dim.id, self.id)
        idx_changes2 = np.isin(self.id, new_dim.id)

        shape = torch.tensor(self.shape)
        new_shape[idx_changes] = shape[idx_changes2]
        
        new_shape = torch.Size(new_shape)
        return new_shape

   
    @property
    def ndim(self) -> int:
        return len(self.shape)
   
    @property
    def shape(self) -> _size:
        """ Returns the shape of the object type

        Returns:
            [type]: [description]
        """        
        return self._shape
    
    @property
    def id(self):
        if self._id is None:
            raise NotImplemented("Please set this in the extended class.")
        else:
            self._id_loc._id
    
    @property
    def idx_batch(self):
        if self._id_loc._batch is None:
            raise NotImplemented("Please set this in the extended class.")
        else:
            return self._id_loc._batch

    @idx_batch.setter
    def idx_batch(self, batch):
        self._id_loc._batch = batch
        
    @property
    def idx_time(self):
        if self._id_loc._time is None:
            raise NotImplemented("Please set this in the extended class.")
        else:
            return self._id_loc._time


    @idx_time.setter
    def idx_time(self, time):
        self._id_loc._time = time
        
    @property
    def idx_pupil(self):
        if self._id_loc._pupil is None:
            raise NotImplemented("Please set this in the extended class.")
        else:
            return self._id_loc._pupil

    @idx_pupil.setter
    def idx_pupil(self, pupil):
        self._id_loc._pupil = pupil
        
    @property
    def idx_channel(self):
     
        if self._id_loc._channel is None:
            raise NotImplemented("Please set this in the extended class.")
        else:
            return self._id_loc._channel

    @idx_channel.setter
    def idx_channel(self, channel):
        self._id_loc._channel = channel     

    @property
    def idx_height(self):
        if self._id_loc._height is None:
            raise NotImplemented("Please set this in the extended class.")
        else:
            return self._id_loc._height

    @idx_height.setter
    def idx_height(self, height):
        self._id_loc._height = height     
         
    @property
    def idx_width(self):
        if self._id_loc._width is None:
            raise NotImplemented("Please set this in the extended class.")
        else:
            return self._id_loc._width

    @idx_width.setter
    def idx_width(self, width):
        self._id_loc._width = width      
        
    @property
    def batch(self):
        return self.shape[self.idx_batch]  

    @batch.setter
    def batch(self, batch):
        self.shape[self.idx_batch] = batch
        
    @property
    def extra(self):
        return self.shape[self.idx_extra]  

    @extra.setter
    def extra(self, extra):
        self.shape[self.idx_extra] = extra
                

    @property
    def time(self):
        return self.shape[self.idx_time]  
    
    @time.setter
    def time(self, time):
        self.shape[self.idx_time] = time
    
    @property
    def pupil(self):
        return self.shape[self.idx_pupil]  
   
    @pupil.setter
    def pupil(self, pupil):
        self.shape[self.idx_pupil] = pupil   
       
    @property
    def height(self):
        return self.shape[self.idx_height]  

    @height.setter
    def height(self, height):
        self.shape[self.idx_height] = height   

    @property
    def width(self):
        return self.shape[self.idx_width]  
    
    @width.setter
    def width(self, width):
        self.shape[self.idx_width] = width   

    @property
    def channel(self):
        return self.shape[self.idx_channel]  
    
    @channel.setter
    def channel(self, channel):
        self.shape[self.idx_channel] = channel

    @property
    def units(self):
        """[summary]

        Returns:
            [type]: [description]
        """        
        return self._units

class TCHW(TensorDimension):
    """TIME CHANNEL HEIGHT WIDTH
    """    

    id = [DIM.TIME, DIM.CHANNEL, DIM.HEIGHT, DIM.WIDTH ]
    units = [UNITS.SECONDS, UNITS.METER, UNITS.METER, UNITS.METER]
    
    def __init__(self,
                n_time : int,
                n_channel : int,
                height : int,
                width : int,
                    ) -> None:
                
        super().__init__()
        self._shape = [n_time, n_channel, height, width]


class TCD(TensorDimension):
    """TIME CHANNEL DIMENSION
    """    

    id = [DIM.TIME, DIM.CHANNEL, DIM.HEIGHT]
    units = [UNITS.SECONDS, UNITS.METER, UNITS.METER]
    
    def __init__(self,
                n_time : int,
                n_channel : int,
                height : int,
                    ) -> None:
                
        super().__init__()
        self._shape = [n_time, n_channel, height]

class CHW(TensorDimension):
    """CHANNEL HEIGHT WIDTH
    """    

    id = [DIM.CHANNEL, DIM.HEIGHT, DIM.WIDTH ]
    units = [UNITS.METER, UNITS.METER, UNITS.METER]
    
    def __init__(self,
                n_channel : int,
                height : int,
                width : int
                ) -> None:
        
        super().__init__()
        self._shape = [n_channel, height, width]



class THW(TensorDimension):
    """TIME HEIGHT WIDTH
    """    

    id = [DIM.TIME, DIM.HEIGHT, DIM.WIDTH ]
    units = [UNITS.SECONDS, UNITS.METER, UNITS.METER]
    
    def __init__(self,
                n_time : int,
                height : int,
                width : int
                ) -> None:
        
        super().__init__()
        self._shape = [n_time, height, width]


class BTPCHW(TensorDimension):
    """BATCH TIME PUPIL CHANNEL HEIGHT WIDTH
    """    
    id = [DIM.BATCH, DIM.TIME, DIM.PUPIL, DIM.CHANNEL, DIM.HEIGHT, DIM.WIDTH ]
    units = [UNITS.UNITLESS, UNITS.SECONDS, UNITS.UNITLESS,  UNITS.METER, UNITS.METER, UNITS.METER]
    

    def __init__(self,
                n_batch : int,
                n_time : int,
                n_pupil : int,
                n_channel : int,
                height : int,
                width : int,
) -> None:

        super().__init__()
        self._shape = [n_batch, n_time, n_pupil, n_channel, height, width]

    @classmethod
    def from_shape(cls, shape : torch.Size):
        return BTPCHW(
            n_batch = shape[0],
            n_time = shape[1],
            n_pupil = shape[2],
            n_channel = shape[3],
            height = shape[-2],
            width = shape[-1],
        )


class BPCHW(TensorDimension):
    """BATCH PUPIL CHANNEL HEIGHT WIDTH
    """ 
    id = [DIM.BATCH, DIM.PUPIL, DIM.CHANNEL, DIM.HEIGHT, DIM.WIDTH ]
    units = [UNITS.UNITLESS, UNITS.UNITLESS, UNITS.METER, UNITS.METER, UNITS.METER]

    def __init__(self,
                n_batch : int,
                n_pupil : int,
                n_channel : int,
                height : int,
                width : int) -> None:

        super().__init__()
        self._shape = [n_batch, n_pupil, n_channel, height, width]

class BTCHW_E(TensorDimension):
    """BATCH TIME CHANNEL HEIGHT WIDTH EXTRA
    """    

    id = [DIM.BATCH, DIM.TIME, DIM.CHANNEL, DIM.HEIGHT, DIM.WIDTH, DIM.EXTRA]
    units = [UNITS.UNITLESS, UNITS.SECONDS, UNITS.METER, UNITS.METER, UNITS.METER, UNITS.UNITLESS]

    def __init__(self,
                n_batch     : int,
                n_time      : int,
                n_channel   : int,
                height      : int,
                width       : int,
                extra_dim   : int,
                    ) -> None:
        super().__init__()
        self._shape = [n_batch, n_time, n_channel, height, width, extra_dim]
        
    @classmethod
    def from_BTCHW(cls,
                   btchw : BTCHW,
                   extra_dim : int = 2
                ) -> BTCHW_E:
            expanded_tensor_dimension = BTCHW_E(
                n_batch         = btchw.batch,
                n_time          = btchw.time,
                n_channel       = btchw.channel,
                height          = btchw.height,
                width           = btchw.width,
                extra_dim       = extra_dim 
                )
            return expanded_tensor_dimension

class BTCHW(TensorDimension):
    """BATCH TIME CHANNEL HEIGHT WIDTH
    """    

    id = [DIM.BATCH, DIM.TIME, DIM.CHANNEL, DIM.HEIGHT, DIM.WIDTH]
    units = [UNITS.UNITLESS, UNITS.SECONDS, UNITS.METER, UNITS.METER, UNITS.METER]


    def __init__(self,
                n_batch : int,
                n_time : int,
                n_channel : int,
                height : int,
                width : int,
                    ) -> None:
        super().__init__()
        self._shape = [n_batch, n_time, n_channel, height, width]


class HW(TensorDimension):
    """BATCH TIME CHANNEL HEIGHT WIDTH
    """    

    id = [DIM.HEIGHT, DIM.WIDTH]
    units = [UNITS.METER, UNITS.METER]

    def __init__(self,
                height : int,
                width : int,
                    ) -> None:
        super().__init__()
        self._shape = [height, width]
        
class H(TensorDimension):
    """HEIGHT
    """    

    id = [DIM.HEIGHT]
    units = [UNITS.METER]

    def __init__(self,
                height : int,
                    ) -> None:
        super().__init__()
        self._shape = [height]

class TC(TensorDimension):
    """TIME CHANNEL
    """    

    id = [DIM.TIME, DIM.CHANNEL]
    units = [UNITS.SECONDS, UNITS.METER]

    def __init__(self,
                n_time : int,
                n_channel : int
                ) -> None:
        super().__init__()
        self._shape = [n_time, n_channel]
        

class T(TensorDimension):
    """ TIME
    """    

    id = [DIM.TIME]
    units = [UNITS.SECONDS]

    def __init__(self,
                n_time : int
                ) -> None:
        super().__init__()
        self._shape = [n_time]
        
class C(TensorDimension):
    """ CHANNEL
    """    

    id = [DIM.CHANNEL]
    units = [UNITS.METER]

    def __init__(self,
                n_channel : int
                ) -> None:
        super().__init__()
        self._shape = [n_channel]


class BTPC(TensorDimension):
    """BATCH TIME CHANNEL HEIGHT WIDTH
    """    

    id = [DIM.BATCH, DIM.TIME, DIM.PUPIL, DIM.CHANNEL]
    units = [UNITS.UNITLESS, UNITS.SECONDS, UNITS.UNITLESS, UNITS.METER]

    def __init__(self,
                n_batch : int,
                n_time : int,
                n_pupil : int,
                n_channel : int,
                    ) -> None:
        super().__init__()
        self._shape = [n_batch, n_time, n_pupil, n_channel]
        
class PCHW(TensorDimension):
    """PUPIL CHANNEL HEIGHT WIDTH
    """    

    id = [DIM.PUPIL, DIM.CHANNEL, DIM.HEIGHT, DIM.WIDTH]
    units = [UNITS.UNITLESS, UNITS.METER, UNITS.METER, UNITS.METER]

    def __init__(self,
                pupil : int,
                channel : int,
                height : int,
                width : int,
                    ) -> None:
        super().__init__()
        self._shape = [pupil, channel, height, width]

