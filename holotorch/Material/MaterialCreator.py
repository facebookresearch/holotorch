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

from holotorch.Material.CGH_Material import CGH_Material
from holotorch.Material.DefaultMaterial import DefaultMaterial
from holotorch.Material.CauchyMaterial import CauchyMaterial

from holotorch.utils.Enumerators import *

def create_material(
        material_type : ENUM_MATERIAL,
        A = None,
        B = None,
        name = None
        ) -> CGH_Material: 
    """Creates a specfic material given a number of arguments

    Args:
        material_type (ENUM_MATERIAL): Each material needs an entry in the ENUM_MATERIAL struct
        A (float, optional): A parameter for Cauchy Material. Defaults to None.
        B (float, optional): B parameter for Cauchy Material. Defaults to None.
        name (str, optional): Name of the material. Defaults to None.

    Returns:
        CGH_Material: The created material object
    """

    if material_type is ENUM_MATERIAL.Default:
        material = DefaultMaterial()

    elif material_type is ENUM_MATERIAL.HOLOGRAFIX:
        material = CauchyMaterial.holografix()

    elif material_type is ENUM_MATERIAL.Cauchy:
        assert A is not None, "A needs to have a value for init of Cauchy material"
        assert B is not None, "B needs to have a value for init of Cauchy material"
        material = CauchyMaterial(A=A, B=B, name = name)
        
    else:
        raise NotImplemented("material type" + str(material_type) +" is not yet implemented")

    return material
