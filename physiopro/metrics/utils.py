# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from .classification import *
from .regression import *


def neg_wrapper(func):
    def wrapper(*args, **kwargs):
        return -1 * func(*args, **kwargs)

    return wrapper


def get_loss_fn(loss_fn):
    # reflection: legacy name
    if loss_fn in ["mse", "single_mse"] or loss_fn.startswith("label"):
        return single_mse
    if loss_fn == "outside_bce":
        return outside_cross_entropy
    if loss_fn == "mase":
        return sequence_mase
    if loss_fn == "mae":
        return single_mae
    if loss_fn == "cross_entropy":
        return cross_entropy

    # return function by name
    try:
        return eval(loss_fn)  # dangerous eval
    except Exception:
        pass
    # return negative function by name
    try:
        return neg_wrapper(eval(re.sub("^neg_", "", loss_fn)))
    except Exception:
        raise NotImplementedError("loss function %s is not implemented" % loss_fn)


def get_metric_fn(eval_metric):
    # reflection: legacy name
    if eval_metric == "corr":
        return neg_wrapper(robust_batch_corr) # more stable
    if eval_metric == "mse":
        return single_mse
    if eval_metric == "mae":
        return single_mae
    if eval_metric in ["rse", "rrse"]:
        return rrse

    try:
        return eval(eval_metric)  # dangerous eval
    except Exception:
        pass
    # return negative function by name
    try:
        return neg_wrapper(eval(re.sub("^neg_", "", eval_metric)))
    except Exception:
        raise NotImplementedError("metric function %s is not implemented" % eval_metric)
