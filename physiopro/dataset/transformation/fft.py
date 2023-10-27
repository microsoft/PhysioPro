# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from .base import TRANSFORMATIONS, BaseTransformation
from ..utils import computeFFT


@TRANSFORMATIONS.register_module('fft')
class FFTTransformation(BaseTransformation):
    def __init__(
            self,
            axis: int
        ):
        super().__init__()
        self.axis = axis

    # fft transformation requires a lot of times
    def transform(self, data):
        assert data.ndim > self.axis >= -data.ndim, "fft index out of range"
        new_data = data.swapaxes(self.axis, -1)  # convert axis to the last dimension
        saved_shape = list(new_data.shape)
        new_data = new_data.reshape(-1, new_data.shape[-1])
        new_data = computeFFT(new_data, new_data.shape[-1])[0]
        saved_shape[-1] = new_data.shape[-1]
        new_data = new_data.reshape(*saved_shape)
        new_data = new_data.swapaxes(self.axis, -1)
        return new_data

    def shape_transform(self, shape):
        shape[self.axis] /= 2
        shape[self.axis] = int(np.ceil(shape[self.axis]))
        return shape

    def __repr__(self):
        return f'FFT Transformation | Axis: {self.axis}'
