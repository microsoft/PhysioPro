# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Union, List
import numpy as np
from .base import TRANSFORMATIONS, BaseTransformation


@TRANSFORMATIONS.register_module('zscore')
class ZScoreTransformation(BaseTransformation):
    def __init__(
            self,
            axis: int,
            mean_value: Optional[Union[int, List[int], np.ndarray]] = None,
            std_value: Optional[Union[int, List[int], np.ndarray]] = None
        ):
        super().__init__()
        if mean_value is not None:
            self.mean = np.array(mean_value)
        else:
            self.mean = None

        if std_value is not None:
            self.std = np.array(std_value)
        else:
            self.std = None
        self.axis = axis

    def initialize(self, data):
        if self.check_initialized():
            return
        assert data.ndim > self.axis >= -data.ndim, "zscore index out of range"
        ndim = data.ndim
        size = data.shape[self.axis]
        data = data.swapaxes(self.axis, -1).reshape(-1, size)
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        new_shape = [1] * ndim
        new_shape[self.axis] = size
        self.mean = self.mean.reshape(*new_shape)
        self.std = self.std.reshape(*new_shape)

    def transform(self, data):
        if not self.check_initialized():
            self.initialize(data)
        return (data - self.mean) / self.std

    def check_initialized(self):
        return self.mean is not None and self.std is not None

    def init_from_meta(self, meta, is_feature=True, pre_fft=False):
        if self.check_initialized():
            return

        mode = 'x' if is_feature else 'y'

        if pre_fft:
            if f'{mode}_mean@train' in meta:
                self.mean = np.array(meta[f'{mode}_mean@train'])
            elif 'mean@train' in meta:
                self.mean = np.array(meta['mean@train'])

            if f'{mode}_std@train' in meta:
                self.std = np.array(meta[f'{mode}_std@train'])
            elif 'std@train' in meta:
                self.std = np.array(meta['std@train'])
        else:
            if f'{mode}_mean_fft@train' in meta:
                self.mean = np.array(meta[f'{mode}_mean_fft@train'])
            elif 'mean_fft@train' in meta:
                self.mean = np.array(meta['mean_fft@train'])

            if f'{mode}_std_fft@train' in meta:
                self.std = np.array(meta[f'{mode}_std_fft@train'])
            elif 'std_fft@train' in meta:
                self.std = np.array(meta['std_fft@train'])

    def __repr__(self):
        return f'Zscore Transformation | Axis: {self.axis} | Mean: {self.mean.tolist()}| Std: {self.std.tolist()}'
