# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import numpy as np
import torch
from ..metrics.utils import K


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def type_loss(prediction, types, loss_func):
    """ Event prediction loss, label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    # compute cross entropy loss
    loss = loss_func(prediction, truth)

    loss = torch.sum(loss)
    return loss, correct_num


def pad_sequence(seq, max_len):
    seq_len = min(seq.shape[1], max_len)
    res = np.zeros((seq.shape[0], max_len), dtype=np.float32)
    res[:, :seq_len] = seq[:, :seq_len]
    return res


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def time_loss(prediction, event_time):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    true = torch.where(true >= 0., true, torch.zeros(true.shape).to(prediction.device))
    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    se = torch.sum(diff * diff)
    return se


def mean(y, pred):
    return K.seq_mean(pred) if y is None else K.seq_mean(pred - y)


def rmse(y, pred):
    loss = (y - pred) ** 2
    return np.sqrt(K.seq_mean(loss, keepdims=False))
