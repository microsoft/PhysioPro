# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Union
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from utilsd.config import Registry, ClassConfig
from .preprocessor import Preprocessor
from .transformation import FFTTransformation


class DATASETS(metaclass=Registry, name="dataset"):
    pass


@DATASETS.register_module("base")
class BaseDataset(Dataset):
    def __init__(
            self,
            feature: Optional[pd.DataFrame] = None,
            label: Optional[pd.DataFrame] = None,
            preprocessor: Optional[Union[ClassConfig[Preprocessor], Preprocessor]] = None,
    ):
        super().__init__()

        if preprocessor is None:
            self.preprocessor = Preprocessor()  # create an default preprocessor
        elif not isinstance(preprocessor, Preprocessor):  # ClassConfig[Preprocessor]
            self.preprocessor = preprocessor.build()
        else:
            self.preprocessor = preprocessor

        self.feature = feature
        self.label = label

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):
        return self.feature.values[index], self.label.values[index]

    @staticmethod
    def perform_sampling(feature, label, ratio):
        assert (np.logical_or(label == 0, label == 1)).all(), "Sampling only support binary classification"
        assert ratio > 0, "Sampling ratio should be positive"

        if ratio < 1:
            neg = np.argwhere(label == 0).flatten()
            pos = np.argwhere(label == 1).flatten()
            neg_size = len(neg)
            selected = np.random.choice(neg, int(neg_size * ratio), replace=False)
            selected = np.concatenate([selected, pos])
            feature = feature[selected]
            label = label[selected]
            print(f"neg/pos ratio: {neg_size / label.sum():.2f} -> {len(selected) / label.sum():.2f}")
        elif ratio > 1:
            neg = np.argwhere(label == 0).flatten()
            pos = np.argwhere(label == 1).flatten()
            pos_size = len(pos)
            selected = np.random.choice(pos, int(pos_size * (ratio - 1)), replace=True)
            selected = np.concatenate([selected, neg])
            feature = feature[selected]
            label = label[selected]
            print(f"neg/pos ratio: {len(neg) / pos_size:.2f} -> {len(neg) / len(selected):.2f}")

        return feature, label

    @staticmethod
    def perform_transformations(feature, transformations, reshape_param):
        if transformations is None:
            return feature
        feature = feature.reshape(-1, *reshape_param)
        for t in transformations:
            feature = t.transform(feature)
        return feature

    @staticmethod
    def initialize_transformations(feature, transformations, meta, reshape_param, is_feature=True):
        feature = feature.reshape(-1, *reshape_param)
        shape = list(feature.shape)
        if transformations is None:
            return shape
        pre_fft = True  # if the current transformation is before fft
        for t in transformations:
            if isinstance(t, FFTTransformation):
                pre_fft = False
            if meta is not None:
                t.init_from_meta(meta, is_feature, pre_fft)
            t.initialize(feature)
            shape = t.shape_transform(shape)
        return shape

    def get_index(self):
        return self.label.index

    def load(self):
        pass

    def freeup(self):
        pass

    def get_label(self, pred):
        return None
