# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from numba import njit, prange
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    r2_score,
    precision_recall_curve,
)
from sklearn.metrics import auc as area_under_curve


@njit
def fast_auc(y_true, y_prob):
    mask = np.logical_not(np.isnan(y_true))
    ratio = np.sum(mask) / mask.size
    y_true = np.extract(mask, y_true)
    y_prob = np.extract(mask, y_prob)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += 1 - y_i
        auc += y_i * nfalse
    auc /= nfalse * (n - nfalse)
    return auc * ratio, ratio


def fast_auprc(y, p):
    mask = np.logical_not(np.isnan(y))
    ratio = np.sum(mask) / mask.size
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
        y, p = y.reshape(-1), p.reshape(-1)
        assert len(y) == len(p), "Shapes of labels and predictions not match."
        return average_precision_score(y, p) * ratio, ratio
    if isinstance(y, list) and isinstance(p, list):
        assert len(y) == len(p), "Shapes of labels and predictions not match."
        return average_precision_score(y, p) * ratio, ratio
    raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))


class K():
    """backend kernel"""

    @staticmethod
    def sum(x, axis=0, keepdims=True):
        if isinstance(x, np.ndarray):
            return x.sum(axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return x.sum(dim=axis, keepdim=keepdims)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def clip(x, min_val, max_val):
        if isinstance(x, np.ndarray):
            return np.clip(x, min_val, max_val)
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, min_val, max_val)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def mean(x, axis=0, keepdims=True):
        # print(x.max())
        if isinstance(x, np.ndarray):
            return x.mean(axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return x.mean(dim=axis, keepdim=keepdims)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def seq_mean(x, keepdims=True):
        if isinstance(x, torch.Tensor):
            return x.mean()
        if isinstance(x, np.ndarray):
            return x.mean()
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def std(x, axis=0, keepdims=True):
        if isinstance(x, np.ndarray):
            return x.std(axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return x.std(dim=axis, unbiased=False, keepdim=keepdims)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def median(x, axis=0, keepdims=True):
        # NOTE: numpy will average when size is even,
        # but tensorflow and pytorch don't average
        if isinstance(x, np.ndarray):
            return np.median(x, axis=axis, keepdims=keepdims)
        if isinstance(x, torch.Tensor):
            return torch.median(x, dim=axis, keepdim=keepdims)[0]
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def shape(x):
        if isinstance(x, np.ndarray):
            return x.shape
        if isinstance(x, torch.Tensor):
            return list(x.shape)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def cast(x, dtype="float"):
        if isinstance(x, np.ndarray):
            return x.astype(dtype)
        if isinstance(x, torch.Tensor):
            return x.type(getattr(torch, dtype))
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def maximum(x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return np.minimum(x, y)
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return torch.max(x, y)
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, max=y)
        if isinstance(y, torch.Tensor):
            return torch.clamp(y, max=x)
        raise NotImplementedError("unsupported data type %s" % type(x))

    @staticmethod
    def auc(y, p):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y = y.reshape(-1)
            p = p.reshape(len(y), -1)
            if p.shape[-1] == 1 and len(p.shape) == 2:
                p = p.squeeze(-1)
            if len(p.shape) == 2:
                assert p.shape[1] == 2, "AUC calculation only works for binary classification"
                p = p[:, 1]
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            return roc_auc_score(y, p)
        if isinstance(y, list) and isinstance(p, list):
            y, p = np.array(y), np.array(p)
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            return K.auc(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def accuracy(y, p):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y = y.reshape(-1)
            p = p.reshape(len(y), -1)
            if p.shape[-1] == 1 & len(p.shape) == 2:
                p = p.squeeze(-1)
            if len(p.shape) == 2:
                p = np.argmax(p, axis=1)
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            try:
                return accuracy_score(y, p)
            except ValueError:
                return K.accuracy(y, (p > 0.5).astype(int))
        if isinstance(y, list) or isinstance(p, list):
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            y, p = np.array(y), np.array(p)
            return K.accuracy(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def bce(y, p, reduce=True):
        assert type(y) == type(p), "Type of label and prediction not match."
        if isinstance(y, torch.Tensor):
            y = y.reshape(-1).to(torch.float)
            p = p.reshape(-1)
            assert len(p) == len(y)
            loss = torch.nn.BCELoss(reduction="mean" if reduce else "none")
            return loss(p, y)
        if isinstance(y, np.ndarray):
            y = y.reshape(-1)
            p = p.reshape(len(y), -1)
            if p.shape[-1] == 1:
                p = p.squeeze(-1)
            return log_loss(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def cross_entropy(y, p, reduce=True):
        assert type(y) == type(p), "Type of label and prediction not match."
        if isinstance(y, torch.Tensor):
            y = y.reshape(-1).to(torch.long)
            p = p.reshape(len(y), -1).squeeze(-1)
            assert len(p) == len(y)
            loss = torch.nn.NLLLoss(reduction="mean" if reduce else "none")
            return loss(p, y)
        if isinstance(y, np.ndarray):
            y = y.reshape(-1)
            p = p.reshape(len(y), -1)
            if p.shape[-1] == 1:
                p = p.squeeze(-1)
            p = np.exp(p)
            return log_loss(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def nll(y, p, reduce=True):
        assert type(y) == type(p), "Type of label and prediction not match."
        if isinstance(y, torch.Tensor):
            loss = torch.nn.NLLLoss(reduction="mean" if reduce else "none")
            return loss(p, y)

        p = np.exp(p)
        p = np.transpose(p, (0, 2, 1))
        p = p.reshape(-1, p.shape[-1])
        y = y.reshape(-1)
        return log_loss(y, p)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def mauc(y, p):
        aucs = 0
        ratios = 0
        _, m = y.shape
        for t in prange(m):
            auc, ratio = fast_auc(y[:, t], p[:, t])
            aucs += auc
            ratios += ratio
        # print(ratios)
        return aucs / ratios

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def dauc(y, p):
        aucs = 0
        ratios = 0
        n, _ = y.shape
        for i in prange(n):
            auc, ratio = fast_auc(y[i, :], p[i, :])
            aucs += auc
            ratios += ratio
        print(ratios)
        return aucs / ratios

    @staticmethod
    def mauprc(y, p):
        aucs = 0
        ratios = 0
        _, m = y.shape
        for t in prange(m):
            auc, ratio = fast_auprc(y[:, t], p[:, t])
            aucs += auc
            ratios += ratio
        return aucs / ratios

    @staticmethod
    def r2_score(y, p):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y = y.reshape(-1)
            p = p.reshape(len(y), -1)
            if p.shape[-1] == 1:
                p = p.squeeze(-1)
            return r2_score(y, p)
        if isinstance(y, list) and isinstance(p, list):
            y, p = np.array(y), np.array(p)
            return K.r2_score(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def ap(y, p):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y, p = y.reshape(-1), p.reshape(-1)
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            return average_precision_score(y, p)
        if isinstance(y, list) and isinstance(p, list):
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            return average_precision_score(y, p)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))

    @staticmethod
    def auprc(y, p):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and isinstance(p, np.ndarray):
            y, p = y.reshape(-1), p.reshape(-1)
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            precision, recall, _ = precision_recall_curve(y, p)
            return area_under_curve(recall, precision)
        if isinstance(y, list) and isinstance(p, list):
            assert len(y) == len(p), "Shapes of labels and predictions not match."
            precision, recall, _ = precision_recall_curve(y, p)
            return area_under_curve(recall, precision)
        raise NotImplementedError("unsupported data type %s or %s" % (type(y), type(p)))


# Add Static Methods
def generic_ops(method):
    def wrapper(x, *args):
        if isinstance(x, np.ndarray):
            return getattr(np, method)(x, *args)
        if isinstance(x, torch.Tensor):
            return getattr(torch, method)(x, *args)
        raise NotImplementedError("unsupported data type %s" % type(x))

    return wrapper


for method in [
    "abs",
    "log",
    "sqrt",
    "exp",
    "log1p",
    "tanh",
    "cosh",
    "squeeze",
    "reshape",
    "zeros_like",
]:
    setattr(K, method, staticmethod(generic_ops(method)))
