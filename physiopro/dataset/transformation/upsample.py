# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
import numpy as np
from .base import TRANSFORMATIONS, BaseTransformation


@TRANSFORMATIONS.register_module('upsample')
class UpsampleTransformation(BaseTransformation):
    def __init__(
            self,
            axis: int,
            ratio: Optional[int] = 1,
        ):
        super().__init__()
        self.axis = axis
        self.ratio = ratio

    def transform(self, data):
        assert data.ndim > self.axis >= -data.ndim, "upsampling index out of range"
        x = np.arange(data.shape[self.axis])
        new_x = np.linspace(0, data.shape[self.axis] - 1, self.ratio * data.shape[self.axis])
        new_data = np.apply_along_axis(lambda y: np.interp(new_x, x, y), self.axis, data)
        return new_data

    def shape_transform(self, shape):
        shape[self.axis] *= self.ratio
        return shape

    def __repr__(self):
        return f'Upsample Transformation | Axis: {self.axis} | Ratio: {self.ratio}'
