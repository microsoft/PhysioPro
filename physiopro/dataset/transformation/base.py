# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from utilsd.config import Registry


class TRANSFORMATIONS(metaclass=Registry, name="transformation"):
    pass


@TRANSFORMATIONS.register_module('base')
class BaseTransformation():
    def __init__(self, **kwargs):
        pass

    def transform(self, data):
        return data

    def init_from_meta(self, meta, is_feature=True, pre_fft=False):
        pass

    def initialize(self, data):
        pass

    def shape_transform(self, shape):
        return shape

    def check_initialized(self):
        pass

    def __repr__(self):
        return 'Base Transformation'
