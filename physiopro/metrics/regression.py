# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
from .kernel import K

EPS = 1e-5


def single_mse(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        loss = (y_true - y_pred) ** 2
        return np.nanmean(loss)

    mask = ~torch.isnan(y_true)
    y_pred = torch.masked_select(y_pred, mask)
    y_true = torch.masked_select(y_true, mask)
    loss = (y_true - y_pred) ** 2
    loss = loss.mean()

    return loss


def single_mae(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        loss = np.abs(y_true - y_pred)
        return np.nanmean(loss)
    loss = (y_true - y_pred).abs()
    return loss.mean()


def sequence_mse(y_true, y_pred):
    loss = (y_true - y_pred) ** 2
    # return torch.mean(loss)

    return K.seq_mean(loss, keepdims=False)


def sequence_mae(y_true, y_pred):
    loss = torch.abs(y_true - y_pred)
    # return torch.mean(loss)
    return K.seq_mean(loss, keepdims=False)


def sequence_mase(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        loss = (y_true - y_pred) ** 2 + np.abs(y_true - y_pred)
    else:
        loss = (y_true - y_pred) ** 2 + (y_true - y_pred).abs()
    return K.seq_mean(loss, keepdims=False)


def single_mase(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        loss = (y_true - y_pred) ** 2 + np.abs(y_true - y_pred)
    else:
        loss = (y_true - y_pred) ** 2 + (y_true - y_pred).abs()
    return K.mean(loss, keepdims=False)


def rrse(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        y_bar = y_true.mean(axis=0)
        loss = np.sqrt(((y_pred - y_true) ** 2).sum()) / np.sqrt(((y_true - y_bar) ** 2).sum())
        return np.nanmean(loss)
    y_bar = y_true.mean(dim=0)
    loss = torch.sqrt(((y_pred - y_true) ** 2).sum()) / torch.sqrt(((y_true - y_bar) ** 2).sum())
    return loss.mean()


def r2(y, preds):
    return K.r2_score(y, preds)


def mape(y_true, y_pred, log=False):
    if isinstance(y_true, np.ndarray):
        if log:
            y_true = np.exp(y_true)
            y_pred = np.exp(y_pred)
        loss = np.abs(y_true - y_pred) / y_true
    else:
        if log:
            y_true = y_true.exp()
            y_pred = y_pred.exp()
        loss = (y_true - y_pred).abs()
    return loss.mean()


def mape_log(y_true, y_pred, log=True):
    if isinstance(y_true, np.ndarray):
        if log:
            y_true = np.exp(y_true)
            y_pred = np.exp(y_pred)
        loss = np.abs(y_true - y_pred) / y_true
    else:
        if log:
            y_true = torch.exp(y_true)
            y_pred = torch.exp(y_pred)
        loss = (y_true - y_pred).abs()
    return K.mean(loss, keepdims=False)


def zscore(x, axis=0):
    mean = K.mean(x, axis=axis)
    std = K.std(x, axis=axis)
    return (x - mean) / (std + EPS)


def robust_zscore(x, axis=0):
    med = K.median(x, axis=axis)
    mad = K.median(K.abs(x - med), axis=axis)
    x = (x - med) / (mad * 1.4826 + EPS)
    return K.clip(x, -3, 3)


def batch_corr(x, y, axis=0, keepdims=True):
    x = zscore(x, axis=axis)
    y = zscore(y, axis=axis)
    return (x * y).mean()


def robust_batch_corr(x, y, axis=0, keepdims=True):
    x = robust_zscore(x, axis=axis)
    y = robust_zscore(y, axis=axis)
    return batch_corr(x, y)
