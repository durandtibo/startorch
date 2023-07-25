from __future__ import annotations

__all__ = ["BaseTensorGenerator", "setup_tensor_generator", "RandUniform", "Uniform"]

from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator
from startorch.tensor.uniform import RandUniformTensorGenerator as RandUniform
from startorch.tensor.uniform import UniformTensorGenerator as Uniform
