# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
import numpy as np
from .base import TRANSFORMATIONS, BaseTransformation


@TRANSFORMATIONS.register_module('downsample')
class DownsampleTransformation(BaseTransformation):
    def __init__(
            self,
            axis: int,
            ratio: Optional[int] = 1,
        ):
        super().__init__()
        self.axis = axis
        self.ratio = ratio

    def transform(self, data):
        assert data.ndim > self.axis >= -data.ndim, "downsample index out of range"
        slices = [slice(None)] * data.ndim
        slices[self.axis] = slice(None, None, self.ratio)
        new_data = data[tuple(slices)]
        return new_data

    def shape_transform(self, shape):
        shape[self.axis] /= self.ratio
        shape[self.axis] = int(np.ceil(shape[self.axis]))
        return shape

    def __repr__(self):
        return f'Downsample Transformation | Axis: {self.axis} | Ratio: {self.ratio}'
