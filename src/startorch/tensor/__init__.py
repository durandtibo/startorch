from __future__ import annotations

__all__ = [
    "BaseTensorGenerator",
    "BaseWrapperTensorGenerator",
    "Full",
    "RandUniform",
    "Uniform",
    "setup_tensor_generator",
]

from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator
from startorch.tensor.constant import FullTensorGenerator as Full
from startorch.tensor.uniform import RandUniformTensorGenerator as RandUniform
from startorch.tensor.uniform import UniformTensorGenerator as Uniform
from startorch.tensor.wrapper import BaseWrapperTensorGenerator
