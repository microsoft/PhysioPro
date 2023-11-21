# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch.nn.functional as F
from .kernel import K


def accuracy(y_true, y_pred):
    return K.accuracy(y_true, y_pred)


def cross_entropy(y_true, y_pred, reduce=True):
    return K.cross_entropy(y_true, y_pred, reduce)


def bce(y_true, y_pred, reduce=True):
    return K.bce(y_true, y_pred, reduce)


def outside_cross_entropy(y_true, y_pred, reduce=True):
    # https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/ops/nn_impl.py#L142
    y = K.cast(y_true > 0, "float")
    p = y_pred
    loss = K.maximum(p, 0) - p * y + K.log1p(K.exp(-K.abs(p)))
    if reduce:
        return K.mean(loss, keepdims=False)
    return loss


def mce(y_true, y_pred):
    return F.binary_cross_entropy_with_logits(y_pred, y_true)


def m_auroc(labels, outputs, label_one_hot_encoded=False):
    if not label_one_hot_encoded:
        num_classes = len(np.unique(labels))
        num_recordings = len(labels)
        labels = np.eye(num_classes)[labels]
    else:
        num_recordings, num_classes = labels.shape

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k] == 1)  # number of pos
        tn[0] = np.sum(labels[:, k] == 0)  # number of neg

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        # ppv = np.zeros(num_thresholds)
        # npv = np.zeros(num_thresholds)

        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float('nan')
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float('nan')

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auroc[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])

    # Compute macro AUROC and macro AUPRC across classes.
    macro_auroc = np.nanmean(auroc)

    return macro_auroc


def m_auprc(labels, outputs, label_one_hot_encoded=False):
    # Convert to one-hot encoding
    if not label_one_hot_encoded:
        num_classes = len(np.unique(labels))
        num_recordings = len(labels)
        labels = np.eye(num_classes)[labels]
    else:
        num_recordings, num_classes = labels.shape

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k] == 1)  # number of pos
        tn[0] = np.sum(labels[:, k] == 0)  # number of neg

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        # tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        # npv = np.zeros(num_thresholds)

        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float('nan')
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float('nan')

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auprc[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    # Compute macro AUROC and macro AUPRC across classes.
    macro_auprc = np.nanmean(auprc)

    return macro_auprc


def multilabel_auroc(labels, outputs):
    """
    Multi-label AUROC

    Parameters:
    -----------
    labels:
        shape (n_samples, n_classes)
    outputs:
        shape (n_samples, n_classes)
    """
    return m_auroc(labels, outputs, True)


def multilabel_auprc(labels, outputs):
    """
    Multi-label AUPRC

    Parameters:
    -----------
    labels:
        shape (n_samples, n_classes)
    outputs:
        shape (n_samples, n_classes)
    """
    return m_auprc(labels, outputs, True)


def mauprc(y, preds):
    return K.mauprc(y, preds)


def mauc(y, preds):
    return K.mauc(y, preds)


def auc(y, preds):
    return K.auc(y, preds)


def dauc(y, preds):
    return K.dauc(y, preds)


def auprc(y, preds):
    return K.auprc(y, preds)


def ap(y, preds):
    return K.ap(y, preds)


def nll(y_true, y_pred, reduce=True):
    return K.nll(y_true, y_pred, reduce)
