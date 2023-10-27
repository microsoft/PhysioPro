# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Optional, Union
from utilsd.config import ClassConfig
import pandas as pd
import numpy as np
from .preprocessor import Preprocessor
from .base import BaseDataset, DATASETS


@DATASETS.register_module("df")
class DfDataset(BaseDataset):
    def __init__(
            self,
            data_folder: Path,
            meta_path: Path,
            task: str = "classification",
            num_classes: Optional[int] = 2, # for classification only
            num_variables: int = 18,
            freq: int = 256,
            max_seq_len: int = 30,
            dataset: Optional[str] = "train",
            preprocessor: Optional[Union[ClassConfig[Preprocessor], Preprocessor]] = None,
    ):
        super().__init__(preprocessor=preprocessor)
        self.num_variables = num_variables
        self.freq = freq
        self.max_seq_len = max_seq_len

        # load meta, feature, label
        meta = pd.read_pickle(meta_path)
        self.meta = meta
        feature, label = [], []
        for idx in meta[dataset]:
            feature.append(pd.read_pickle(f"{data_folder}/feature_line/{idx.split('.')[0]}.pkl"))
            label.append(pd.read_pickle(f"{data_folder}/label_line/{idx.split('.')[0]}.pkl"))
        self.label = pd.concat(label).values

        feature = pd.concat(feature).values
        feature = feature.reshape((-1, max_seq_len, freq, self.num_variables))

        shape = self.initialize_transformations(feature, self.preprocessor.x_transformation, self.meta,
                                                (max_seq_len, freq, self.num_variables))

        if not self.preprocessor.online_preprocess:
            feature = self.perform_transformations(feature, self.preprocessor.x_transformation, (max_seq_len, freq, self.num_variables))

        self.num_variables = shape[-1]
        self.freq = shape[-2]
        self.max_seq_len = shape[-3] * shape[-2]


        self.feature_num_variables = feature.shape[-1]
        self.feature_freq = feature.shape[-2]
        self.feature_max_seq_len = feature.shape[-3]

        self.feature = feature.reshape(-1, self.feature_max_seq_len, self.feature_freq, self.feature_num_variables)
        self.label_index = np.arange(len(self.label))
        assert len(self.feature) == len(self.label)

        if task == 'classification':
            self.num_classes = num_classes

            self.inv_label_mapping = None
            if 'classes' not in self.meta:
                self.label = self.label.astype(int)
            else:
                classes = self.meta['classes']
                if isinstance(classes, list):
                    label_mapping = {k: idx for idx, k in enumerate(classes)}
                    self.inv_label_mapping = {v: k for (k, v) in label_mapping.items()}
                    mapped_func = np.vectorize(lambda x: label_mapping[x])
                    self.label = mapped_func(self.label)
                elif isinstance(classes, dict):
                    mapped_func = np.vectorize(lambda x: classes[x])
                    self.label = mapped_func(self.label)

            # only preform sampling for training
            if dataset == 'train' and self.num_classes == 2 and self.preprocessor.sample_ratio != 1:
                self.feature, self.label = self.perform_sampling(self.feature, self.label, self.preprocessor.sample_ratio)


    def __getitem__(self, index):
        assert self.feature is not None and self.label is not None
        feature = self.feature[index]  # time, nodes * freq

        if self.preprocessor.online_preprocess:
            feature = self.perform_transformations(feature, self.preprocessor.x_transformation,
                                                   (self.feature_max_seq_len, self.feature_freq, self.feature_num_variables))

        feature = feature.reshape(self.max_seq_len, self.num_variables)
        return feature.astype(np.float32), self.label[index]

    def get_index(self):
        assert self.label is not None and self.feature is not None
        return self.label_index

    def get_label(self, pred):
        if hasattr(self, 'inv_label_mapping') and self.inv_label_mapping is not None:
            indices = pred.argmax(axis=-1)
            return np.array([self.inv_label_mapping[idx] for idx in indices])
