# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
The dataloader for loading .ts data.
Refer to the code in https://github.com/gzerveas/mvts_transformer.
"""
from typing import Optional, Union
import numpy as np
from utilsd.config import ClassConfig
from .base import DATASETS
from .ts import TSDataset
from .preprocessor import Preprocessor


@DATASETS.register_module("mask_ts")
class MaskTSDataset(TSDataset):
    """
    A dataset for time series forecasting.
    """

    def __init__(
            self,
            prefix: str = "data/Multivariate_ts",
            name: str = "Heartbeat",
            max_seq_len: int = 0,
            dataset: str = "train",
            dataset_split_ratio: float = 0.,
            task: str = "classification",
            preprocessor: Optional[Union[ClassConfig[Preprocessor], Preprocessor]] = None,
            mask_ratio: Optional[float] = 0.3,
    ):
        super().__init__(prefix, name, max_seq_len, dataset, dataset_split_ratio, task, preprocessor)

        self.mask_ratio = mask_ratio

        for i in range(self.datasize):
            _length = self._max_seq_len - self.feature.loc[i]['dim_0'].isnull().sum()
            removed_points = np.random.choice(_length, int(_length * self.mask_ratio),
                                              replace=False) + self.feature.loc[i]['dim_0'].isnull().sum()
            for idx in removed_points:
                self.feature.loc[i].iloc[idx] = np.nan
