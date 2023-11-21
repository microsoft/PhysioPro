# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from torch import nn
from .base import MODELS
from .timeseries import TS


@MODELS.register_module("cinc2020")
class CinC2020(TS):
    def __init__(
            self,
            task: str,
            optimizer: str,
            lr: float,
            weight_decay: float,
            loss_fn: str,
            metrics: List[str],
            observe: str,
            lower_is_better: bool,
            max_epochs: int,
            batch_size: int,
            network: Optional[nn.Module] = None,
            output_dir: Optional[Path] = None,
            checkpoint_dir: Optional[Path] = None,
            num_workers: Optional[int] = 1,
            early_stop: Optional[int] = None,
            out_ranges: Optional[List[Union[Tuple[int, int], Tuple[int, int, int]]]] = None,
            model_path: Optional[str] = None,
            out_size: int = 1,
            aggregate: bool = True,
            weights_file: Optional[Path] = None,
    ):
        super().__init__(
            task,
            optimizer,
            lr,
            weight_decay,
            loss_fn,
            metrics,
            observe,
            lower_is_better,
            max_epochs,
            batch_size,
            network,
            output_dir,
            checkpoint_dir,
            num_workers,
            early_stop,
            out_ranges,
            model_path,
            out_size,
            aggregate,
        )
        self.act_out = nn.Identity()
        self.load_classes_and_weights(weights_file)
        self.normal_class = '426783006'
        self.cur_epoch_thrs = float(0)
        self.best_epoch_thrs = float(0.5)

        self.metrics.append('CinC2020_valid')
        self.metric_fn['CinC2020_valid'] = self.cinc2020_metric_valid
        self.metrics.append('CinC2020_test')
        self.metric_fn['CinC2020_test'] = self.cinc2020_metric_test

    def load_classes_and_weights(self, weight_file):
        """
        Load the weight matrix.
        The weight matrix should have the following form:
        ,    c1,   c2,   c3
        c1, 1.2, 2.3, 3.4
        c2, 4.5, 5.6, 6.7
        c3, 7.8, 8.9, 9.0
        """
        with open(weight_file) as f:
            self.classes = f.readline().strip().split(',')[1:]
            self.challenge_weight = [[float(x) for x in line.strip().split(',')[1:]] for line in f]

    def _post_epoch_valid_best(self):
        self.best_epoch_thrs = self.cur_epoch_thrs

    def cinc2020_metric_valid(self, labels, probs, step = 0.02):
        scores = []
        for thr in np.arange(0., 1., step):
            preds = (probs > thr).astype(int)
            challenge_metric = self.compute_cinc2020_metric_with_preds(labels, preds)
            scores.append(challenge_metric)
        scores = np.array(scores)

        # Best thrs and preds of current epoch
        idxs = np.argmax(scores, axis=0)
        self.cur_epoch_thrs = np.array([idxs*step])
        return scores[idxs]

    def cinc2020_metric_test(self, labels, probs):
        preds = (probs > self.best_epoch_thrs).astype(int)
        return self.compute_cinc2020_metric_with_preds(labels, preds)

    def compute_cinc2020_metric_with_preds(self, labels, outputs):
        """
        Compute the evaluation metric for the Challenge.
        Reference:
        https://github.com/physionetchallenges/evaluation-2020
        """
        num_recordings, num_classes = np.shape(labels)
        def compute_modified_confusion_matrix(labels, outputs):
            """
            Compute modified confusion matrix for multi-class, multi-label tasks.
            Compute a binary multi-class, multi-label confusion matrix, where the rows
            are the labels and the columns are the outputs.
            """
            res = np.zeros((num_classes, num_classes))
            for i in range(num_recordings):
                # Calculate the number of positive labels and/or outputs.
                normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
                for j in range(num_classes):
                    # Assign full and/or partial credit for each positive class.
                    if labels[i, j]:
                        for k in range(num_classes):
                            if outputs[i, k]:
                                res[j, k] += 1.0/normalization
            return res

        # Compute the observed score.
        A = compute_modified_confusion_matrix(labels, outputs)
        observed_score = np.nansum(self.challenge_weight * A)

        # Compute the score for the model that always choose the correct label(s).
        correct_outputs = labels
        A = compute_modified_confusion_matrix(labels, correct_outputs)
        correct_score = np.nansum(self.challenge_weight * A)

        # Compute the score for the model that always choose the normal class.
        inactive_outputs = np.zeros((num_recordings, num_classes), dtype=bool)
        normal_class_index = self.classes.index(self.normal_class)
        inactive_outputs[:, normal_class_index] = 1
        A = compute_modified_confusion_matrix(labels, inactive_outputs)
        inactive_score = np.nansum(self.challenge_weight * A)

        if correct_score != inactive_score:
            normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
        else:
            normalized_score = float('nan')

        return normalized_score
