# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Optional
from utilsd.config import RegistryConfig
from .transformation import TRANSFORMATIONS


class Preprocessor():
    def __init__(
            self,
            x_transformation: Optional[List[RegistryConfig[TRANSFORMATIONS]]] = None,
            y_transformation: Optional[List[RegistryConfig[TRANSFORMATIONS]]] = None,
            online_preprocess: Optional[bool] = False,
            sample_ratio: Optional[float] = 1
        ):
        self.x_transformation_config = x_transformation
        self.y_transformation_config = y_transformation
        self.x_transformation = None
        self.y_transformation = None
        self.online_preprocess = online_preprocess
        self.sample_ratio = sample_ratio
        self.__initialize_preprocessor()

    def __initialize_preprocessor(self):
        x_preprocessor = self.x_transformation_config
        if x_preprocessor is not None:
            self.x_transformation = []
            for transformation in x_preprocessor:
                self.x_transformation.append(transformation.build())

        y_preprocessor = self.y_transformation_config
        if y_preprocessor is not None:
            self.y_transformation = []
            for transformation in y_preprocessor:
                self.y_transformation.append(transformation.build())

    def x_update(self, x_transformation):
        self.x_transformation = x_transformation

    def y_update(self, y_transformation):
        self.x_transformation = y_transformation
