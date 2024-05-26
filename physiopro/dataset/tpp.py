# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Union
import os
import pickle
import numpy as np
import torch
from utilsd.config import ClassConfig
from .preprocessor import Preprocessor
from .base import BaseDataset, DATASETS


@DATASETS.register_module("tpp")
class EventDataset(BaseDataset):
    """ Event stream dataset. """
    PAD = 0

    def __init__(
        self,
        prefix: str = "data/temporal_point_process",
        name: str = "Neonate",
        fold: str = "folder1",
        dataset: Optional[str] = "train",
        preprocessor: Optional[Union[ClassConfig[Preprocessor], Preprocessor]] = None,
        max_len: Optional[int] = 500,
        clip_max: Optional[int] = -1,
    ):
        super().__init__(preprocessor=preprocessor)
        self.data_folder = os.path.join(prefix, name, fold)
        self.dataset = dataset
        self.max_len = max_len
        assert dataset in ['train', 'test'], 'Invalid dataset'

        with open(os.path.join(self.data_folder, self.dataset + '.pkl'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        self.num_types = data['dim_process']
        # filter data with length == 1
        data_idx = [i for i in range(len(data[dataset])) if len(data[dataset][i]) > 1]
        data = [data[dataset][i][:max_len] for i in range(len(data[dataset])) if i in data_idx]

        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        if clip_max == -1:
            self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        else:
            self.time_gap = [[min(elem['time_since_last_event'], clip_max) for elem in inst] for inst in data]
            self.time = [np.cumsum(inst).tolist() for inst in self.time_gap]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]
        self.length = len(data)

    @property
    def num_classes(self):
        return self.num_types

    @property
    def max_seq_len(self):
        return self.max_len

    @property
    def num_variables(self):
        return None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx]

    @staticmethod
    def pad_time(insts):
        """ Pad the instance to the max seq length in batch. """
        max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([
            inst + [EventDataset.PAD] * (max_len - len(inst))
            for inst in insts])

        return torch.tensor(batch_seq, dtype=torch.float32)

    @staticmethod
    def pad_type(insts):
        """ Pad the instance to the max seq length in batch. """
        max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([
            inst + [EventDataset.PAD] * (max_len - len(inst))
            for inst in insts])

        return torch.tensor(batch_seq, dtype=torch.long)

    def collate_fn(self, insts):
        """ Collate function, as required by PyTorch. """
        time, time_gap, event_type = list(zip(*insts))
        time = self.pad_time(time)
        time_gap = self.pad_time(time_gap)
        event_type = self.pad_type(event_type)
        return time, time_gap, event_type
