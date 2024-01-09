# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
from scipy.fftpack import fft
from scipy.signal import resample


def resample_data(signals, to_freq=200, window_size=4):
    """
    Resample signals from its original sampling freq to another freqency
    refer to https://github.com/tsy935/eeg-gnn-ssl/blob/main/data/resample_signals.py

    Parameters:
    -----------
    signals: numpy.ndarray
        EEG signal slice, (num_channels, num_data_points)
    to_freq: int
        Re-sampled frequency in Hz
    window_size: int
        time window in seconds

    Returns:
    --------
    numpy.ndarray
        shape (num_channels, resampled_data_points)
    """
    num = int(to_freq * window_size)
    resampled = resample(signals, num=num, axis=1)

    return resampled


def compute_fft(signals, n):
    """
    Parameters:
    -----------
    signals: numpy.ndarray
        EEG signals, (number of channels, number of data points)
        shape (525960, 19, 200) for v1.5.2
    n: integer
        length of positive frequency terms of fourier transform

    Returns:
    --------
    numpy.ndarray
        log amplitude of FFT of signals, (number of channels, number of data points)
    """
    # FFT on the last dimension
    fourier_signal = fft(signals, n=n, axis=-1)

    # only take the positive freq part
    idx_pos = int(np.floor(n / 2))
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = np.abs(fourier_signal)
    amp[amp == 0.] = 1e-8  # avoid log of 0

    FT = np.log(amp)
    # P = np.angle(fourier_signal) # risk of OOM

    return FT


def create_output_dirs(output_dir, montage_type, annotation_type='term'):
    """
    create output directories for:
    - meta
    - features
    - labels
    - line-wise features
    - line-wise labels
    """
    meta_dir = f"{output_dir}/{montage_type}/{annotation_type}/"  # meta
    feature_dir = f"{output_dir}/{montage_type}/{annotation_type}/feature/"  # features
    label_dir = f"{output_dir}/{montage_type}/{annotation_type}/label/"  # labels
    feature_linewise_dir = f"{output_dir}/{montage_type}/{annotation_type}/feature_line/"  # line-wise features
    label_linewise_dir = f"{output_dir}/{montage_type}/{annotation_type}/label_line/"  # line-wise labels

    for path in [output_dir, meta_dir, feature_dir, feature_linewise_dir, label_dir, label_linewise_dir]:
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    return meta_dir, feature_dir, label_dir, feature_linewise_dir, label_linewise_dir
