# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from numbers import Number
from typing import Any, Iterable, Optional, Union

import numpy as np
import torch


def printt(s=None):
    if s is None:
        print()
    else:
        print(str(s), end="\t")

def to_torch(
        x: Any,
        dtype: Optional[torch.dtype] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> Optional[Union[torch.Tensor, Iterable]]:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray) and issubclass(x.dtype.type, (np.bool_, np.number)):
        x = torch.from_numpy(x).to(device)  # type: ignore
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)  # type: ignore
    if isinstance(x, (np.number, np.bool_, Number)):
        return to_torch(np.asanyarray(x), dtype, device)
    if isinstance(x, dict):
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return (to_torch(i, dtype, device) for i in x)
    raise TypeError(f"object {x} cannot be converted to torch.")

# Evaluation Metrics

class MovingAverage():
    def __init__(self, decay, init_val=0, shape=None):
        self.decay = decay
        if type(init_val) == int:
            self.value = init_val * np.ones(shape)
        else:
            self.value = init_val

    def add(self, val):
        mask = np.isnan(val)
        self.value[~mask] = (1 - self.decay) * self.value[~mask] + self.decay * val[~mask]


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.avg = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def performance(self, care="avg"):
        return getattr(self, care)

    def status(self):
        return str(self.performance())


class GlobalMeter():
    def __init__(self, f=lambda x, y: 0):
        self.reset()
        self.f = f

    def reset(self):
        self.ys = []  # np.array([], dtype=np.int) # ground truths
        self.preds = []  # np.array([], dtype=np.float) # predictions

    def update(self, ys, preds):
        if isinstance(ys, torch.Tensor):
            ys = ys.detach().squeeze(-1).cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().squeeze(-1).cpu().numpy()
        assert isinstance(ys, np.ndarray) and isinstance(preds, np.ndarray), "Please input as type of ndarray."
        self.ys.append(ys)
        self.preds.append(preds)

    def concat(self):
        if isinstance(self.ys, list) and isinstance(self.preds, list):
            self.ys = [np.expand_dims(ys, 0) if len(ys.shape) == 0 else ys for ys in self.ys]
            self.preds = [np.expand_dims(preds, 0) if len(preds.shape) == 0 else preds for preds in self.preds]
            self.ys = np.concatenate(self.ys, axis=0)
            self.preds = np.concatenate(self.preds, axis=0)

    def performance(self):
        return self.f(self.ys, self.preds)

    def status(self):
        return str(self.performance())


class AverageTracker():
    def __init__(self, metrics):
        self.metrics = metrics  # isolated metric list to guarantee metric order
        self.trackers = {}
        self.ss = {}  # snapshot status
        for m in self.metrics:
            self.trackers[m] = AverageMeter()

    def update(self, metric, val, n=1):
        try:
            meter = self.trackers[metric]
        except Exception:
            raise KeyError("Metric has not been found. %s" % metric)
        meter.update(val, n)

    def get(self, metric, care="avg"):
        """cared_value"""
        assert metric in self.metrics, "Metric %s not found." % metric
        return getattr(self.trackers[metric], care)

    def performance(self, metric="all", care="avg"):
        """{metric: cared_value}"""
        stat = {}
        if isinstance(metric, str) and isinstance(care, str):
            assert (metric == "all") or (metric in self.metrics), "Not support %s metric." % metric
            assert care in ["val", "avg", "sum", "count"], "Not support %s in performance meter." % care
            if metric == "all":
                for m in self.metrics:
                    stat[m] = getattr(self.trackers[m], care)
            else:
                stat[metric] = getattr(self.trackers[metric], care)
        else:
            # TODO metrics=[m1, m2, ...] care=[c1, c2, ...]
            # TODO metrics==[m1, m2, ...] care=care
            raise NotImplementedError("TODO")
        return stat

    def _snapshot(self):
        """Refresh the performance"""
        stat = self.performance()
        self.ss = stat
        return self.ss

    def snapshot_metric(self, metric):
        """Return the latest performance of the given metric without refresh"""
        assert metric in self.metrics, "Metric %s not found." % metric
        if len(self.ss) == 0:
            self._snapshot()
        return self.ss[metric]

    def snapshot(self):
        # assert len(self.snapshot) > 0, "Please update Tracker.performance() first!"
        if len(self.ss) == 0:
            return self._snapshot()
        return self.ss

    def status(self):
        """Refresh and return all the performance"""
        self.snapshot()
        return "\t".join([str(self.ss[m]) for m in self.metrics])

    def __str__(self):
        return self.status()


class GlobalTracker(GlobalMeter):
    def __init__(self, metrics, metric_fn):
        self.reset()
        self.metrics = metrics
        self.metric_fn = metric_fn
        self.ss = {}

    def performance(self, metric="all"):
        stat = {}
        if isinstance(metric, str):
            assert (metric == "all") or (metric in self.metrics), "Not support %s metric." % metric
            if metric == "all":
                for m in self.metrics:
                    res = self.metric_fn[m](self.ys, self.preds)
                    if hasattr(res, "item"):
                        res = res.item()
                    stat[m] = res
                    self.ss[m] = stat[m]
            else:
                res = self.metric_fn[metric](self.ys, self.preds)
                if hasattr(res, "item"):
                    res = res.item()
                stat[metric] = res
                self.ss[metric] = stat[metric]
        else:
            raise NotImplementedError("TODO")
        return stat

    def snapshot(self, metric="all"):
        stat = {}
        if isinstance(metric, str):
            assert (metric == "all") or (metric in self.metrics), "Not support %s metric." % metric
            if metric == "all":
                for m in self.metrics:
                    try:
                        stat[m] = self.metric_fn[m]
                    except Exception:
                        raise KeyError("Run performance first")
            else:
                try:
                    stat[metric] = self.ss[metric]
                except Exception:
                    raise KeyError("Run performance first")
        else:
            raise NotImplementedError("TODO")
        return stat
