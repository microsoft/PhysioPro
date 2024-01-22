# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Tuple, Optional, Union
from pathlib import Path
import copy
import json
import time
import datetime
import torch
import numpy as np
import pandas as pd
from torch import optim
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from utilsd.earlystop import EarlyStopStatus, EarlyStop
from utilsd import use_cuda
from ..common.utils import AverageMeter, to_torch, GlobalTracker, printt
from ..metrics.tpp import LabelSmoothingLoss
from ..metrics.utils import K
from ..module.rnn import RNNLayers
from .utils import get_non_pad_mask, type_loss, compute_event, pad_sequence, softplus, time_loss, mean, rmse
from .base import MODELS, BaseModel


@MODELS.register_module("tpp")
class TPP(BaseModel):
    def __init__(
            self,
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
            num_workers: Optional[int] = 8,
            early_stop: Optional[int] = None,
            out_ranges: Optional[List[Union[Tuple[int, int], Tuple[int, int, int]]]] = None,
            model_path: Optional[str] = None,
            out_size: int = 1,
            aggregate: bool = True,
            scale_event: Optional[int] = 1,
            scale_time: Optional[int] = 10,
            tmax: Optional[int] = 5,
            use_likelihood: Optional[bool] = True,
            step_size: Optional[int] = 100,
            intensity_type: Optional[str] = 'thp',
            use_rnn: Optional[bool] = True,
            temporal_encoding: Optional[bool] = True,
            input_time_flag: Optional[bool] = False,
    ):
        """
        The model for general masked temporal point process.

        Args:
            task: the prediction task, classification or regression.
            optimizer: which optimizer to use.
            lr: learning rate.
            weight_decay: L2 normlize weight
            loss_fn: loss function.
            metrics: metrics to evaluate model.
            observe: metric for model selection (earlystop).
            lower_is_better: whether a lower observed metric means better result.
            max_epochs: maximum epoch to learn.
            batch_size: batch size.
            early_stop: earlystop rounds.
            out_ranges: a list of final ranges to take as final output. Should have form [(start, end), (start, end, step), ...]
            model_path: the path to existing model parameters for continued training or finetuning
            out_size: the output size for multi-class classification or multi-variant regression task.
            aggregate: whether to aggregate across whole sequence. [nouse]
            scale_event: hyper-parameter for event prediction
            scale_time: hyper-parameter for time prediction
            tmax: max time gap between consecutive events
            use_likelihood: whether to use log-likelihood loss or linear prediction
            step_size: number of timestamp for log-likelihood estimation per second
            intensity_type: type of intensity function, choice: [thp, sahp]
            use_rnn: whether to use rnn after encoder
            temporal_encoding: whether to use temporal encoding outside network
            input_time_flag: whether to input times to network
        """
        self.hyper_paras = {
            "intensity_type": intensity_type,
            "use_rnn": use_rnn,
            "out_ranges": out_ranges,
            "out_size": out_size,
            "aggregate": aggregate,
        }
        super().__init__(
            loss_fn,
            metrics,
            observe,
            lr,
            lower_is_better,
            max_epochs,
            batch_size,
            early_stop,
            optimizer,
            weight_decay,
            network,
            model_path,
            output_dir,
            checkpoint_dir,
            num_workers,
        )

        # Task specific initialization
        self.scale_event = scale_event
        self.scale_time = scale_time
        self.tmax = tmax
        self.use_likelihood = use_likelihood
        self.use_linear = not use_likelihood
        self.step_size = step_size
        self.temporal_encoding = temporal_encoding
        self.input_time_flag = input_time_flag

    def _build_network(
            self,
            network,
            out_ranges: Optional[List[Union[Tuple[int, int, int], Tuple[int, int]]]] = None,
            out_size: int = 1,
            aggregate: bool = True,
            intensity_type: str = 'thp',
            use_rnn: bool = True,
        ) -> None:
        """Initilize the network parameters

        Args:
            out_ranges: a list of final ranges to take as final output. Should have form [(start, end), (start, end, step), ...]
            out_size: the output size for multi-class classification or multi-variant regression task.
            aggregate: whether to aggregate across whole sequence.
        """

        self.network = network
        self.aggregate = aggregate
        self.event_emb = nn.Embedding(self.hyper_paras['out_size'] + 1, network.hidden_size, padding_idx=0)

        if use_rnn:
            self.rnn = RNNLayers(network.hidden_size, network.hidden_size)

        if intensity_type == 'sahp':
            self.gelu = nn.GELU()

            self.start_layer = nn.Sequential(
                nn.Linear(network.output_size, network.output_size, bias=True), self.gelu
            )

            self.converge_layer = nn.Sequential(
                nn.Linear(network.output_size, network.output_size, bias=True), self.gelu
            )

            self.decay_layer = nn.Sequential(
                nn.Linear(network.output_size, network.output_size, bias=True), nn.Softplus(beta=10.0)
            )

            # for calculating intersity function !!!
            self.intensity_layer = nn.Sequential(
                nn.Linear(network.output_size, self.hyper_paras['out_size'], bias=True), nn.Softplus(beta=1.)
            )
        elif intensity_type == "thp":
            self.linear = nn.Linear(network.output_size, self.hyper_paras['out_size'])
            self.alpha = nn.Parameter(torch.tensor([-0.1] * self.hyper_paras['out_size']))
            self.beta = nn.Parameter(torch.tensor([1.0] * self.hyper_paras['out_size']))
        else:
            raise NotImplementedError

    def calculate_intensity_thp(self, **kwargs):
        """
        Input: hidden: batch*seq_len*d_model
        Output: conditional intensity function: batch*seq_len*sample*event_type
        Using the intensity function defined in Transformer Hawkes Processw
        """
        enc_out = kwargs['hidden']
        timestamp = torch.linspace(start=0, end=self.tmax, steps=self.tmax * self.step_size).to(enc_out.device)

        hidden = self.linear(enc_out)  # batch*seq*type
        return softplus(hidden.unsqueeze(-2) + self.alpha * timestamp.unsqueeze(-1), self.beta)

    @staticmethod
    def state_decay(converge_point, start_point, omega, delta_t):
        # * element-wise product
        cell_t = torch.tanh(converge_point + (start_point - converge_point) * torch.exp(-omega * delta_t))
        return cell_t

    def calculate_intensity_sahp(self, **kwargs):
        enc_out = kwargs['hidden']
        start_point = self.start_layer(enc_out).unsqueeze(-2)
        converge_point = self.converge_layer(enc_out).unsqueeze(-2)
        omega = self.decay_layer(enc_out).unsqueeze(-2)
        timestamp = torch.linspace(start=0, end=self.tmax, steps=self.tmax * self.step_size).to(enc_out.device)
        timestamp = timestamp.reshape(1, 1, -1, 1)

        state = self.state_decay(converge_point, start_point, omega, timestamp)
        return self.intensity_layer(state)

    def calculate_intensity(self, **kwargs):
        if self.hyper_paras["intensity_type"] == 'thp':
            intensity = self.calculate_intensity_thp(**kwargs)
        elif self.hyper_paras["intensity_type"] == 'sahp':
            intensity = self.calculate_intensity_sahp(**kwargs)
        else:
            raise NotImplementedError
        return intensity

    def compute_integral_unbiased(self, all_lambda, event_time, non_pad_mask):
        """ Log-likelihood of non-events, using Monte Carlo integration. """
        input_lambda = all_lambda.clone()
        diff_time = (event_time[:, 1:] - event_time[:, :-1]) * non_pad_mask[:, 1:]

        num_sample = diff_time * self.step_size

        # for safety
        num_sample = torch.clip(num_sample, 1, self.tmax * self.step_size).long()
        input_lambda = input_lambda[:, :-1, ...]

        for i in range(num_sample.shape[0]):
            for j in range(num_sample.shape[1]):
                input_lambda[i, j, num_sample[i][j]:, :] = 0

        input_lambda = torch.sum(input_lambda, dim=(2, 3)) / num_sample

        unbiased_integral = input_lambda * diff_time
        return unbiased_integral

    def temporal_log_likelihood(self, all_lambda, event_time, event_type):
        non_pad_mask = get_non_pad_mask(event_type).squeeze(2)

        type_mask = torch.zeros([*event_type.size(), self.hyper_paras["out_size"]], device=event_type.device)
        for i in range(self.hyper_paras["out_size"]):
            type_mask[:, :, i] = (event_type == i + 1).bool().to(event_type.device)

        diff_time = (event_time[:, 1:] - event_time[:, :-1]) * non_pad_mask[:, 1:]
        num_sample = diff_time * self.step_size

        # for safety
        num_sample = torch.clip(num_sample, 0, self.tmax * self.step_size - 1).long()
        event_lambda = all_lambda[:, :-1, ...]
        num_sample = num_sample.reshape(-1).cpu()
        event_lambda = event_lambda.reshape(num_sample.shape[0], -1, self.hyper_paras["out_size"])

        event_lambda = event_lambda[torch.arange(num_sample.shape[0]), num_sample, :]
        event_lambda = event_lambda.reshape(diff_time.shape[0], diff_time.shape[1], -1)

        type_lambda = torch.sum(event_lambda * type_mask[:, 1:, :], dim=2)

        # event log-likelihood
        event_ll = compute_event(type_lambda, non_pad_mask[:, 1:])
        event_ll = torch.sum(event_ll, dim=-1)

        # non-event log-likelihood, either numerical integration or MC integration
        non_event_ll = self.compute_integral_unbiased(all_lambda, event_time, non_pad_mask)
        non_event_ll = torch.sum(non_event_ll, dim=-1)

        return event_ll, non_event_ll

    def temporal_prediction_from_linear(self, all_lambda, enc_out, event_time, event_type):
        non_pad_mask = get_non_pad_mask(event_type)
        time_prediction = self.time_predictor(enc_out, non_pad_mask)
        type_prediction = self.type_predictor(enc_out, non_pad_mask)
        return type_prediction, time_prediction

    def temporal_prediction_from_integral(self, all_lambda, enc_out, event_time, event_type):
        """ Prediction time using equation in the paper"""
        non_pad_mask = get_non_pad_mask(event_type).squeeze(2)

        timestep = 1.0 / self.step_size
        temp_time = torch.linspace(start=0, end=self.tmax, steps=self.tmax * self.step_size).to(all_lambda.device)
        sample_time = torch.ones_like(event_time.unsqueeze(2)) * temp_time

        sum_lambda = all_lambda.sum(dim=-1)  # remove event type [B, T, sample]

        integral_lambda = torch.cumsum(sum_lambda, dim=-1) * timestep
        prob_lambda = torch.exp(-integral_lambda) * sum_lambda

        expected_lambda = sample_time * prob_lambda

        # trapeze method
        time_prediction = 0.5 * (expected_lambda[:, :, 1:] + expected_lambda[:, :, :-1]).sum(dim=-1) * timestep
        # time_prediction = (sample_time * prob_lambda).sum(dim=-1) * (tmax / num_samples)

        ratio_lambda = all_lambda / (sum_lambda.unsqueeze(-1) + 1e-6)
        prob_type = ratio_lambda * prob_lambda.unsqueeze(-1)
        event_prediction = 0.5 * (prob_type[:, :, 1:, :] + prob_type[:, :, :-1, :]).sum(dim=-2) * timestep

        time_prediction = time_prediction * non_pad_mask
        event_prediction = event_prediction * non_pad_mask.unsqueeze(-1)

        return event_prediction, time_prediction.unsqueeze(-1)

    def temporal_prediction(self, all_lambda, enc_out, event_time, event_type):
        if self.use_linear:
            predictions = self.temporal_prediction_from_linear(all_lambda, enc_out, event_time, event_type)
        else:
            predictions = self.temporal_prediction_from_integral(all_lambda, enc_out, event_time, event_type)
        return predictions

    def calculate_loss(self, all_lambda, enc_out, event_time, event_type):
        loss_fn = LabelSmoothingLoss(0.1, self.hyper_paras['out_size'], ignore_index=-1, use_softmax=self.use_linear)
        event_ll, non_event_ll = self.temporal_log_likelihood(all_lambda, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        prediction = self.temporal_prediction(all_lambda, enc_out, event_time, event_type)

        # type prediction
        pred_loss, _ = type_loss(prediction[0], event_type, loss_fn)

        # time prediction
        reg_loss = time_loss(prediction[1], event_time)

        loss = event_loss + self.scale_event * pred_loss + reg_loss / self.scale_time

        # fix bug: divide by batch size
        loss = loss / all_lambda.shape[0]
        return loss, event_loss, pred_loss, reg_loss

    def calculate_loglikelihood_loss(self, all_lambda, enc_out, event_time, event_type):
        event_ll, non_event_ll = self.temporal_log_likelihood(all_lambda, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)
        return event_loss, event_loss

    def loglikelihood(self, **kwargs):
        all_lambda = kwargs['all_lambda']
        event_time = kwargs['event_time']
        event_type = kwargs['event_type']
        event_ll, non_event_ll = self.temporal_log_likelihood(all_lambda, event_time, event_type)
        return event_ll - non_event_ll

    @staticmethod
    def get_metric_fn(metric):
        if metric == 'll':
            metric_fn = mean
        elif metric == 'ap':
            metric_fn = K.accuracy
        elif metric == 'rmse':
            metric_fn = rmse
        elif metric == 'r2':
            metric_fn = K.r2_score
        elif metric == 'auc':
            metric_fn = K.auc
        else:
            raise NotImplementedError
        return metric_fn

    def get_loss_fn(self, loss_fn):
        if loss_fn == 'multitask':
            _loss_fn = self.calculate_loss
        elif loss_fn == 'll':
            _loss_fn = self.calculate_loglikelihood_loss
        else:
            raise NotImplementedError
        return _loss_fn

    def _init_optimization(
            self,
            optimizer: str,
            lr: float,
            weight_decay: float,
            loss_fn: str,
            metrics: List[str],
            observe: str,
            lower_is_better: bool,
            max_epochs: int,
            batch_size: int,
            early_stop: Optional[int] = None,
    ) -> None:
        """Setup loss function, evaluation metrics and optimizer"""
        for k, v in locals().items():
            if k not in ["self", "metrics", "observe", "lower_is_better", "loss_fn"]:
                self.hyper_paras[k] = v
        self.loss_fn = self.get_loss_fn(loss_fn)
        self.metric_fn = {}
        for f in metrics:
            self.metric_fn[f] = self.get_metric_fn(f)
        self.metrics = metrics
        if early_stop is not None:
            self.early_stop = EarlyStop(patience=early_stop, mode="min" if lower_is_better else "max")
        else:
            self.early_stop = EarlyStop(patience=max_epochs, mode="min" if lower_is_better else "max")
        self.max_epoches = max_epochs
        self.batch_size = batch_size
        self.observe = observe
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = getattr(optim, optimizer)(self.parameters(), lr=lr, weight_decay=weight_decay)

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, inputs):
        # previous encoding
        event_type, event_time = inputs
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.event_emb(event_type)

        if self.temporal_encoding:
            tem_enc = self.temporal_enc(event_time, non_pad_mask)
            enc_output += tem_enc

        enc_out = self.network(enc_output)

        if self.hyper_paras["use_rnn"]:
            enc_out = self.rnn(enc_output, non_pad_mask)
        all_lambda = self.calculate_intensity(hidden=enc_out, t0=event_time, t1=event_time + self.tmax)
        return all_lambda, enc_out

    def _init_scheduler(self, loader_length=False):
        """Setup learning rate scheduler"""
        if loader_length:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 10, gamma=0.5)

    def fit(
        self,
        trainset: Dataset,
        validset: Optional[Dataset] = None,
        testset: Optional[Dataset] = None,
    ) -> nn.Module:

        """Fit the model to data, if evaluation dataset is offered,
           model selection (early stopping) would be conducted on it.

        Args:
            trainset (Dataset): The training dataset.
            validset (Dataset, optional): The evaluation dataset. Defaults to None.
            testset (Dataset, optional): The test dataset. Defaults to None.

        Returns:
            nn.Module: return the model itself.
        """
        trainset.load()
        if validset is not None:
            validset.load()
        eval_metrics = []

        loader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=trainset.collate_fn,
            num_workers=self.num_workers,
        )
        self._init_scheduler(False)

        self.best_params = copy.deepcopy(self.state_dict())
        self.best_network_params = copy.deepcopy(self.network.state_dict())
        iterations = 0
        start_epoch, best_res = self._resume()
        best_epoch = best_res.pop("best_epoch", 0)
        best_score = self.early_stop.best
        for epoch in range(start_epoch, self.max_epoches):
            # if self._early_stop() and self.early_stop_epoches is not None and stop_epoches >= self.early_stop_epoches:
            #     print("earlystop")
            #     break
            # stop_epoches += 1
            # training
            self.train()
            train_loss = AverageMeter()
            train_event_loss = AverageMeter()
            train_pred_loss = AverageMeter()
            train_reg_loss = AverageMeter()

            train_global_tracker = {
                metric: GlobalTracker([metric], {
                    metric: self.metric_fn[metric]
                }) for metric in self.metrics
            }
            start_time = time.time()
            for _, (event_time, _, event_type) in enumerate(loader):
                if use_cuda():
                    event_time, event_type = (
                        to_torch(event_time, device="cuda"),
                        to_torch(event_type, device="cuda"),
                    )

                all_lambda, enc_out = self((event_type, event_time))
                # negative log-likelihood
                loss, event_loss, pred_loss, reg_loss = self.loss_fn(all_lambda, enc_out, event_time, event_type)
                type_prediction, time_prediction = self.temporal_prediction(all_lambda, enc_out,  event_time, event_type)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.optimizer.step()
                loss = loss.item()
                event_loss = event_loss.item()
                pred_loss = pred_loss.item()
                reg_loss = reg_loss.item()

                train_loss.update(loss, event_type.size(0))
                train_event_loss.update(event_loss, event_type.size(0))
                train_pred_loss.update(pred_loss, event_type.size(0))
                train_reg_loss.update(reg_loss, event_type.size(0))

                for metric in self.metrics:
                    mask = get_non_pad_mask(event_type).squeeze(-1)
                    seq_len = mask.sum(dim=-1).int().cpu() - 1
                    if metric == 'll':
                        pred = self.loglikelihood(all_lambda=all_lambda, event_time=event_time,
                                                  event_type=event_type) / mask.sum(dim=-1).float()
                        train_global_tracker[metric].update(torch.zeros_like(pred), pred)
                    elif metric in ['rmse', 'r2']:
                        pred = pack_padded_sequence(time_prediction.squeeze(-1)[:, :-1], seq_len, batch_first=True,
                                                    enforce_sorted=False)
                        y = pack_padded_sequence(event_time[:, 1:] - event_time[:, :-1], seq_len, batch_first=True,
                                                 enforce_sorted=False)
                        train_global_tracker[metric].update(y.data, pred.data)
                    elif metric in ['ap']:
                        pred = pack_padded_sequence(type_prediction[:, :-1, :], seq_len, batch_first=True,
                                                    enforce_sorted=False)
                        y = pack_padded_sequence(event_type[:, 1:] - 1, seq_len, batch_first=True, enforce_sorted=False)
                        train_global_tracker[metric].update(y.data, pred.data)
                    elif metric in ['auc']:
                        pred = pack_padded_sequence(type_prediction[:, :-1, :], seq_len, batch_first=True,
                                                    enforce_sorted=False)
                        y = pack_padded_sequence(event_type[:, 1:] - 1, seq_len, batch_first=True, enforce_sorted=False)
                        train_global_tracker[metric].update(y.data, pred.data.argmax(dim=-1))

                iterations += 1
                self._post_batch(iterations, epoch, train_loss, train_global_tracker, validset, testset)

            # Update scheduler once per batch
            # if self.scheduler is not None:
            #     self.scheduler.step()

            train_time = time.time() - start_time
            loss = train_loss.performance()  # loss
            reg_loss = train_reg_loss.performance()  # reg_loss
            event_loss = train_event_loss.performance()  # event_loss
            pred_loss = train_pred_loss.performance()  # pred_loss

            start_time = time.time()

            for metric in self.metrics:
                train_global_tracker[metric].concat()

            metric_res = {
                metric: train_global_tracker[metric].performance()[metric] for metric in self.metrics
            }
            metric_time = time.time() - start_time
            metric_res["loss"] = loss
            metric_res["reg_loss"] = reg_loss
            metric_res["event_loss"] = event_loss
            metric_res["pred_loss"] = pred_loss
            # print log
            printt(f"{epoch}\t'train'\tTime:{train_time:.2f}\tMetricT: {metric_time:.2f}")
            for metric, value in metric_res.items():
                printt(f"{metric}: {value:.4f}")
            print(f"{datetime.datetime.today()}")
            for k, v in metric_res.items():
                self.writer.add_scalar(f"{k}/train", v, epoch)
            self.writer.flush()

            if validset is not None:
                with torch.no_grad():
                    eval_res = self.evaluate(validset, epoch)
                eval_metrics.append(eval_res)
                value = eval_res[self.observe]
                es = self.early_stop.step(value)
                if es == EarlyStopStatus.BEST:
                    best_score = value
                    best_epoch = epoch
                    self.best_params = copy.deepcopy(self.state_dict())
                    self.best_network_params = copy.deepcopy(self.network.state_dict())
                    best_res = {"train": metric_res, "valid": eval_res}
                    torch.save(self.best_params, f"{self.checkpoint_dir}/model_best.pkl")
                    torch.save(self.best_network_params, f"{self.checkpoint_dir}/network_best.pkl")
                elif es == EarlyStopStatus.STOP and self._early_stop():
                    break
            else:
                es = self.early_stop.step(metric_res[self.observe])
                if es == EarlyStopStatus.BEST:
                    best_score = metric_res[self.observe]
                    best_epoch = epoch
                    self.best_params = copy.deepcopy(self.state_dict())
                    self.best_network_params = copy.deepcopy(self.network.state_dict())
                    best_res = {"train": metric_res}
                    torch.save(self.best_params, f"{self.checkpoint_dir}/model_best.pkl")
                    torch.save(self.best_network_params, f"{self.checkpoint_dir}/network_best.pkl")
                elif es == EarlyStopStatus.STOP and self._early_stop():
                    break
            self._checkpoint(epoch, {**best_res, "best_epoch": best_epoch})

        # release the space of train and valid dataset
        trainset.freeup()
        if validset is not None:
            validset.freeup()

        # finish training, test on origin train and test, save model and write logs
        self._load_weight(self.best_params)
        print("Begin evaluate on original trainset ...")
        with torch.no_grad():
            origin_train_res = self.evaluate(trainset)
        for k, v in origin_train_res.items():
            self.writer.add_scalar(f"{k}/train_origin", v, epoch)
        best_res["train_origin"] = origin_train_res
        trainset.freeup()

        best_res['valid_best'] = eval_metrics[0]
        for metric in eval_metrics[1:]:
            for key in best_res['valid_best'].keys():
                best_res['valid_best'][key] = max(best_res['valid_best'][key], metric[key]) if key in ['ap', 'll', 'r2'] \
                    else min(best_res['valid_best'][key], metric[key])

        if testset is not None:
            testset.load()
            print("Begin evaluate on testset ...")
            with torch.no_grad():
                test_res = self.evaluate(testset)
            for k, v in test_res.items():
                self.writer.add_scalar(f"{k}/test", v, epoch)
            value = test_res[self.observe]
            best_score = value
            best_res["test"] = test_res
            testset.freeup()
        torch.save(self.best_params, f"{self.checkpoint_dir}/model_best.pkl")
        torch.save(self.best_network_params, f"{self.checkpoint_dir}/network_best.pkl")
        with open(f"{self.checkpoint_dir}/res.json", "w") as f:
            json.dump(best_res, f, indent=4, sort_keys=True)
        keys = list(self.hyper_paras.keys())
        for k in keys:
            if type(self.hyper_paras[k]) not in [int, float, str, bool, torch.Tensor]:
                self.hyper_paras.pop(k)
        self.writer.add_hparams(self.hyper_paras, {"result": best_score, "best_epoch": best_epoch})

        return self

    def evaluate(self, validset: Dataset, epoch: Optional[int] = None) -> dict:
        """Evaluate the model on the given dataset.

        Args:
            validset (Dataset): The dataset to be evaluated on.
            epoch (int, optional): If given, would write log to tensorboard and stdout. Defaults to None.

        Returns:
            dict: The results of evaluation.
        """
        loader = DataLoader(
            validset,
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=validset.collate_fn,
            num_workers=self.num_workers,
        )

        self.eval()
        eval_loss = AverageMeter()
        eval_event_loss = AverageMeter()
        eval_pred_loss = AverageMeter()
        eval_reg_loss = AverageMeter()
        eval_global_tracker = {
                metric: GlobalTracker([metric], {
                    metric: self.metric_fn[metric]
                }) for metric in self.metrics
            }
        start_time = time.time()
        validset.load()
        with torch.no_grad():
            for _, (event_time, _, event_type) in enumerate(loader):
                if use_cuda():
                    event_time, event_type = (
                        to_torch(event_time, device="cuda"),
                        to_torch(event_type, device="cuda"),
                    )

                all_lambda, enc_out = self((event_type, event_time))
                # negative log-likelihood
                loss, event_loss, pred_loss, reg_loss = self.loss_fn(all_lambda, enc_out, event_time, event_type)
                type_prediction, time_prediction = self.temporal_prediction(all_lambda, enc_out, event_time, event_type)
                loss = loss.item()
                event_loss = event_loss.item()
                pred_loss = pred_loss.item()
                reg_loss = reg_loss.item()

                eval_loss.update(loss, event_type.size(0))
                eval_event_loss.update(event_loss, event_type.size(0))
                eval_pred_loss.update(pred_loss, event_type.size(0))
                eval_reg_loss.update(reg_loss, event_type.size(0))

                for metric in self.metrics:
                    mask = get_non_pad_mask(event_type).squeeze(-1)
                    seq_len = mask.sum(dim=-1).int().cpu() - 1
                    if metric == 'll':
                        pred = self.loglikelihood(all_lambda=all_lambda, event_time=event_time,
                                                  event_type=event_type) / mask.sum(dim=-1).float()
                        eval_global_tracker[metric].update(torch.zeros_like(pred), pred)
                    elif metric in ['rmse', 'r2']:
                        pred = pack_padded_sequence(time_prediction.squeeze(-1)[:, :-1], seq_len, batch_first=True,
                                                    enforce_sorted=False)
                        y = pack_padded_sequence(event_time[:, 1:] - event_time[:, :-1], seq_len, batch_first=True,
                                                 enforce_sorted=False)
                        eval_global_tracker[metric].update(y.data, pred.data)
                    elif metric in ['ap']:
                        pred = pack_padded_sequence(type_prediction[:, :-1, :], seq_len, batch_first=True,
                                                    enforce_sorted=False)
                        y = pack_padded_sequence(event_type[:, 1:] - 1, seq_len, batch_first=True, enforce_sorted=False)
                        eval_global_tracker[metric].update(y.data, pred.data)
                    elif metric in ['auc']:
                        pred = pack_padded_sequence(type_prediction[:, :-1, :], seq_len, batch_first=True,
                                                    enforce_sorted=False)
                        y = pack_padded_sequence(event_type[:, 1:] - 1, seq_len, batch_first=True, enforce_sorted=False)
                        eval_global_tracker[metric].update(y.data, pred.data.argmax(dim=-1))

        eval_time = time.time() - start_time
        loss = eval_loss.performance()  # loss
        reg_loss = eval_reg_loss.performance()  # reg_loss
        event_loss = eval_event_loss.performance()  # event_loss
        pred_loss = eval_pred_loss.performance()  # pred_loss

        start_time = time.time()

        for metric in self.metrics:
            eval_global_tracker[metric].concat()

        metric_res = {
            metric: eval_global_tracker[metric].performance()[metric] for metric in self.metrics
        }
        metric_time = time.time() - start_time
        metric_res["loss"] = loss
        metric_res["reg_loss"] = reg_loss
        metric_res["event_loss"] = event_loss
        metric_res["pred_loss"] = pred_loss

        if epoch is not None:
            printt(f"{epoch}\t'valid'\tTime:{eval_time:.2f}\tMetricT: {metric_time:.2f}")
            for metric, value in metric_res.items():
                printt(f"{metric}: {value:.4f}")
            print(f"{datetime.datetime.today()}")
            for k, v in metric_res.items():
                self.writer.add_scalar(f"{k}/valid", v, epoch)

        return metric_res

    def predict(self, dataset: Dataset, name: str):
        """Output the prediction on given data.

        Args:
            datasets (Dataset): The dataset to predict on.
            name (str): The results would be saved to {name}_pre.pkl.

        Returns:
            np.ndarray: The model output.
        """
        self.eval()
        preds_time = []
        preds_event = []
        gt_time = []
        gt_event = []
        dataset.load()

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            num_workers=self.num_workers,
        )

        for _, (event_time, _, event_type) in enumerate(loader):
            if use_cuda():
                event_time, event_type = (
                    to_torch(event_time, device="cuda"),
                    to_torch(event_type, device="cuda"),
                )

            all_lambda, enc_out = self((event_type, event_time))
            # self.network.visualize((event_type, event_time))
            type_prediction, time_prediction = self.temporal_prediction(all_lambda, enc_out, event_time, event_type)

            non_pad_mask = get_non_pad_mask(event_type)
            time_prediction = torch.cat((event_time[:, :1], time_prediction.squeeze(-1)), dim=-1)[:, :-1]
            time_prediction = time_prediction.cumsum(dim=-1) * non_pad_mask.squeeze(-1)
            type_prediction = (type_prediction.argmax(dim=-1) + 1) * non_pad_mask.squeeze(-1)

            preds_time.append(pad_sequence(time_prediction.detach().cpu().numpy(), dataset.max_len))
            preds_event.append(pad_sequence(type_prediction.detach().cpu().numpy(), dataset.max_len))
            gt_time.append(pad_sequence(event_time.detach().cpu().numpy(), dataset.max_len))
            gt_event.append(pad_sequence(event_type.detach().cpu().numpy(), dataset.max_len))

        pred_time = np.concatenate(preds_time, axis=0)
        pred_event = np.concatenate(preds_event, axis=0)
        gt_time = np.concatenate(gt_time, axis=0)
        gt_event = np.concatenate(gt_event, axis=0)

        # data_length = len(dataset.get_index())
        # prediction = prediction.reshape(data_length, -1)

        prediction = {
            'predict_time': pd.DataFrame(data=pred_time, index=np.arange(pred_time.shape[0])),
            'predict_event': pd.DataFrame(data=pred_event, index=np.arange(pred_event.shape[0])),
        }
        ground_truth = {
            'truth_time': pd.DataFrame(data=gt_time, index=np.arange(gt_time.shape[0])),
            'truth_event': pd.DataFrame(data=gt_event, index=np.arange(gt_event.shape[0])),
        }

        pd.to_pickle(prediction, self.checkpoint_dir / (name + "_pre.pkl"))
        pd.to_pickle(ground_truth, self.checkpoint_dir / (name + "_gt.pkl"))
        return prediction
