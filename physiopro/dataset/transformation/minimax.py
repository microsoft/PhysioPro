# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Union, List
import numpy as np
from .base import TRANSFORMATIONS, BaseTransformation


@TRANSFORMATIONS.register_module('minimax')
class MiniMaxTransformation(BaseTransformation):
    def __init__(
            self,
            axis: int,
            min_value: Optional[Union[int, List[int], np.ndarray]] = None,
            max_value: Optional[Union[int, List[int], np.ndarray]] = None
        ):
        super().__init__()
        if min_value is not None:
            self.min = np.array(min_value)
        else:
            self.min = None

        if max_value is not None:
            self.max = np.array(max_value)
        else:
            self.max = None
        self.axis = axis

    def initialize(self, data):
        if self.check_initialized():
            return
        assert data.ndim > self.axis >= -data.ndim, "zscore index out of range"
        ndim = data.ndim
        size = data.shape[self.axis]
        data = data.swapaxes(self.axis, -1).reshape(-1, size)
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)
        new_shape = [1] * ndim
        new_shape[self.axis] = size
        self.min = self.min.reshape(*new_shape)
        self.max = self.max.reshape(*new_shape)

    def transform(self, data):
        if not self.check_initialized():
            self.initialize(data)
        return (data - self.min) / (self.max - self.min + 1e-9)

    def check_initialized(self):
        return self.min is not None and self.max is not None

    def init_from_meta(self, meta, is_feature=True, pre_fft=False):
        if self.check_initialized():
            return

        mode = 'x' if is_feature else 'y'

        if pre_fft:
            if f'{mode}_min@train' in meta:
                self.min = np.array(meta[f'{mode}_min@train'])
            elif 'min@train' in meta:
                self.min = np.array(meta['min@train'])

            if f'{mode}_max@train' in meta:
                self.max = np.array(meta[f'{mode}_max@train'])
            elif 'max@train' in meta:
                self.max = np.array(meta['max@train'])
        else:
            if f'{mode}_min_fft@train' in meta:
                self.min = np.array(meta[f'{mode}_min_fft@train'])
            elif 'min_fft@train' in meta:
                self.min = np.array(meta['min_fft@train'])

            if f'{mode}_max_fft@train' in meta:
                self.max = np.array(meta[f'{mode}_max_fft@train'])
            elif 'max_fft@train' in meta:
                self.max = np.array(meta['max_fft@train'])

    def __repr__(self):
        return f'MiniMax Transformation | Axis: {self.axis} | Min: {self.min.tolist()}| Max: {self.max.tolist()}'
