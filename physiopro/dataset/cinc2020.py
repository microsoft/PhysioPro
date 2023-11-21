# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
from utilsd.config import ClassConfig
from scipy.io import loadmat
from scipy.signal import decimate, resample
from biosppy.signals.tools import filter_signal
from .preprocessor import Preprocessor
from .base import DATASETS, BaseDataset


@DATASETS.register_module("CinC2020")
class CinC2020(BaseDataset):
    '''
    Code implemented based on the official implementation of paper: 
    A Wide and Deep Transformer Neural Network for 12-Lead ECG Classification
    Reference:
    https://ieeexplore.ieee.org/document/9344053
    '''
    def __init__(
            self,
            data_folder: Path,
            meta_path: Path,
            featname_path: Path,
            feats_path: Path,
            test_fold: int = 1,
            dataset: Optional[str] = "train",
            preprocessor: Optional[Union[ClassConfig[Preprocessor], Preprocessor]] = None,
            window: int = 7500,
            fs: int = 500,
            nb_windows: int = 1,
            nb_feats: int = 20,
            # filter_bandwidth: Optional[List[int]] = [3, 45],
            max_seq_len: int = 30,
    ):
        super().__init__(preprocessor=preprocessor)
        # Load all features dataframe
        data_df = pd.read_csv(meta_path, index_col=0)
        self.all_feats = pd.concat([pd.read_csv(f, index_col=0) for f in list(Path(feats_path).glob('*/*all_feats_ch_1.zip'))])

        # Get feature names in order of importance (remove duration and demo)
        self.feature_names = list(np.load(featname_path))
        self.feature_names.remove('full_waveform_duration')
        self.feature_names.remove('Age')
        self.feature_names.remove('Gender_Male')

        # Compute top feature means and stds
        # Get top feats (exclude signal duration)
        feats = self.all_feats[self.feature_names[:nb_feats]].values

        # First, convert any infs to nans
        feats[np.isinf(feats)] = np.nan

        # Store feature means and stds
        self.feat_means = np.nanmean(feats, axis=0)
        self.feat_stds = np.nanstd(feats, axis=0)

        valid_fold = (test_fold - 1) % 10
        train_fold = np.delete(np.arange(10), [valid_fold, test_fold])
        train_df = data_df[data_df.fold.isin(train_fold)]
        self.age_mean = train_df.Age.mean()
        self.age_std = train_df.Age.std()

        if dataset == 'train':
            self.df = data_df[data_df.fold.isin(train_fold)]
        elif dataset == 'test':
            self.df = data_df[data_df.fold == test_fold]
        elif dataset == 'valid':
            self.df = data_df[data_df.fold == valid_fold]
        self.window = window
        self.nb_windows = nb_windows
        self.fs = fs
        self.data_folder = data_folder
        self.nb_feats = nb_feats
        self.filter_bandwidth = [3, 45]
        self.classes = sorted(['270492004', '164889003', '164890007', '426627000', '713427006',
                               '713426002', '445118002', '39732003', '164909002', '251146004',
                               '698252002', '10370003', '284470004', '427172004', '164947007',
                               '111975006', '164917005', '47665007', '59118001', '427393009',
                               '426177001', '426783006', '427084000', '63593006', '164934002',
                               '59931005', '17338001'])
        self.classes_nb = {c: i + 1 for i, c in enumerate(self.classes)}
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx):
        row = self.df.iloc[idx]  # patient, gender, age, label_one_hop, fold
        filename = str(self.data_folder/(row.Patient + '.hea'))
        data, hdr = self.load_challenge_data(filename)  # [12, seq_len]
        seq_len = data.shape[-1]  # get the length of the ecg sequence

        # Get top (normalized) features (excludes signal duration and demo feats)
        top_feats = self.all_feats[self.all_feats.filename == row.Patient][self.feature_names[:self.nb_feats]].values
        # First, convert any infs to nans
        top_feats[np.isinf(top_feats)] = np.nan
        # Replace NaNs with feature means
        top_feats[np.isnan(top_feats)] = self.feat_means[None][np.isnan(top_feats)]
        # Normalize wide features
        feats_normalized = (top_feats - self.feat_means) / self.feat_stds
        # Use zeros (normalized mean) if cannot find patient features
        if len(feats_normalized) == 0:
            feats_normalized = np.zeros(self.nb_feats)[None]

        # Apply band pass filter
        if self.filter_bandwidth is not None:
            data = self.apply_filter(data, self.filter_bandwidth)  # dims not changed

        data = self.normalize(data)
        label = row[self.classes].values.astype(float)

        # Add just enough padding to allow window
        # pad = np.abs(np.min(seq_len - window, 0))
        pad = np.abs(min(seq_len - self.window, 0))

        if pad > 0:
            data = np.pad(data, ((0, 0), (0, pad)))
            # data = np.pad(data, ((0,0),(0,pad+1)))
            # note = (pad, seq_len, self.window, data.shape[-1])
            seq_len = data.shape[-1]  # get the new length of the ecg sequence

        starts = np.random.randint(seq_len - self.window + 1, size=self.nb_windows)  # get start indices of ecg segment
        ecg_segs = np.array([data[:, start:start+self.window] for start in starts])

        age = (self.get_age(hdr[13]) - self.age_mean) / self.age_std
        sex = float(1.) if hdr[14].find('Female') >= 0. else float(0)
        feats_normalized = np.concatenate((np.array([[age, sex]]), feats_normalized), axis=1)

        return (ecg_segs, feats_normalized), label

    def get_index(self):
        return self.df.index

    def load_challenge_data(self, header_file):
        with open(header_file, 'r') as f:
            header = f.readlines()
        sampling_rate = int(header[0].split()[2])
        mat_file = header_file.replace('.hea', '.mat')
        x = loadmat(mat_file)
        recording = np.asarray(x['val'], dtype=np.float64)

        if sampling_rate > self.fs:
            recording = decimate(recording, int(sampling_rate / self.fs))
        elif sampling_rate < self.fs:
            recording = resample(recording, int(recording.shape[-1] * (self.fs / sampling_rate)), axis=1)

        return recording, header

    def normalize(self, seq, smooth=1e-8):
        ''' Normalize each sequence between -1 and 1 '''
        return 2 * (seq - np.min(seq, axis=1)[None].T) / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T - 1

    def extract_templates(self, signal, rpeaks, before=0.2, after=0.4, fs=500):
        # convert delimiters to samples
        before = int(before * fs)
        after = int(after * fs)

        # Sort R-Peaks in ascending order
        rpeaks = np.sort(rpeaks)

        # Get number of sample points in waveform
        length = len(signal)

        # Create empty list for templates
        templates = []

        # Create empty list for new rpeaks that match templates dimension
        rpeaks_new = np.empty(0, dtype=int)

        # Loop through R-Peaks
        for rpeak in rpeaks:

            # Before R-Peak
            a = rpeak - before
            if a < 0:
                continue

            # After R-Peak
            b = rpeak + after
            if b > length:
                break

            # Append template list
            templates.append(signal[a:b])

            # Append new rpeaks list
            rpeaks_new = np.append(rpeaks_new, rpeak)

        # Convert list to numpy array
        templates = np.array(templates).T

        return templates, rpeaks_new

    def apply_filter(self, signal, filter_bandwidth, fs=500):
        # Calculate filter order
        order = int(0.3 * fs)
        # Filter signal
        signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                     order=order, frequency=filter_bandwidth,
                                     sampling_rate=fs)
        return signal

    def get_age(self, hdr):
        ''' Get list of ages as integers from list of hdrs '''
        res = re.search(r': (\d+)\n', hdr)
        if res is None:
            res = 0
        else:
            res = float(res.group(1))
        return res

    def __len__(self):
        return len(self.df)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_variables(self):
        return 0
