# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import datetime
import json
import time
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from utilsd import use_cuda
from utilsd.config import Registry
from utilsd.earlystop import EarlyStop, EarlyStopStatus

from ..metrics import get_loss_fn, get_metric_fn
from ..common.utils import AverageMeter, GlobalTracker, to_torch, printt


class MODELS(metaclass=Registry, name="model"):
    pass


@MODELS.register_module()
class BaseModel(nn.Module):
    def __init__(
            self,
            loss_fn: str,
            metrics: List[str],
            observe: str,
            lr: float = 1e-3,
            lower_is_better: bool = True,
            max_epochs: int = 50,
            batch_size: int = 512,
            early_stop: Optional[int] = None,
            optimizer: str = "Adam",
            weight_decay: float = 1e-5,
            network: Optional[nn.Module] = None,
            model_path: Optional[str] = None,
            output_dir: Optional[Path] = None,
            checkpoint_dir: Optional[Path] = None,
            num_workers: Optional[int] = 8,
        ) -> None:
        super().__init__()
        if not hasattr(self, "hyper_paras"):
            self.hyper_paras = {}
        self._build_network(network, **self.hyper_paras)
        self._init_optimization(
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            loss_fn=loss_fn,
            metrics=metrics,
            observe=observe,
            lower_is_better=lower_is_better,
            max_epochs=max_epochs,
            batch_size=batch_size,
            early_stop=early_stop,
        )
        self._init_logger(output_dir)
        self.checkpoint_dir = checkpoint_dir
        self.num_workers = num_workers

        if model_path is not None:
            self.load(model_path)
        if use_cuda():
            print("Using GPU")
            self.cuda()

    def _build_network(self, network, *args, **kwargs) -> None:
        """Initilize the network parameters"""
        self.network = network
        raise NotImplementedError()

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
        self.loss_fn = get_loss_fn(loss_fn)
        self.metric_fn = {}
        for f in metrics:
            self.metric_fn[f] = get_metric_fn(f)
        self.metrics = metrics
        if early_stop is not None:
            self.early_stop = EarlyStop(patience=early_stop, mode="min" if lower_is_better else "max")
        else:
            self.early_stop = EarlyStop(patience=max_epochs, mode="min" if lower_is_better else "max")
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.observe = observe
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = getattr(optim, optimizer)(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _init_logger(self, log_dir: Path) -> None:
        """initilize the tensorboard writer

        Args:
            log_dir (str): The log directory.
        """
        self.writer = SummaryWriter(log_dir)
        self.writer.flush()

    def forward(self, inputs: torch.Tensor):
        """The pytorch module forward function

        Args:
            X (torch.Tensor): Tensorlized feature.
        """

    def _init_scheduler(self, loader_length):
        """Setup learning rate scheduler"""
        self.scheduler = None

    def _post_batch(self, iterations: int, epoch, train_loss, train_global_tracker, validset, testset):
        pass

    def _post_epoch_valid_best(self):
        pass

    def _load_weight(self, params):
        """Load the trained model parameter weights"""
        self.load_state_dict(params, strict=True)

    def _early_stop(self):
        """Use early stopping on the validation set"""
        return True

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

        loader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        self._init_scheduler(len(loader))
        self.best_params = copy.deepcopy(self.state_dict())
        self.best_network_params = copy.deepcopy(self.network.state_dict())
        iterations = 0
        start_epoch, best_res = self._resume()
        best_epoch = best_res.pop("best_epoch", 0)
        best_score = self.early_stop.best
        for epoch in range(start_epoch, self.max_epochs):
            self.train()
            train_loss = AverageMeter()
            train_global_tracker = GlobalTracker(self.metrics, self.metric_fn)
            start_time = time.time()
            for _, (data, label) in tqdm(enumerate(loader), total=len(loader)):
                if use_cuda():
                    data, label = to_torch(data, device="cuda"), to_torch(label, device="cuda")
                pred = self(data)
                if self.out_ranges is not None:
                    pred = pred[:, self.out_ranges]
                    label = label[:, self.out_ranges]
                loss = self.loss_fn(label.squeeze(-1), pred.squeeze(-1))

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.optimizer.step()
                loss = loss.item()
                train_loss.update(loss, np.prod(label.shape))
                train_global_tracker.update(label, pred)
                if self.scheduler is not None:
                    self.scheduler.step()
                iterations += 1
                self._post_batch(iterations, epoch, train_loss, train_global_tracker, validset, testset)

            train_time = time.time() - start_time
            loss = train_loss.performance()  # loss
            start_time = time.time()
            train_global_tracker.concat()
            metric_res = train_global_tracker.performance()
            metric_time = time.time() - start_time
            metric_res["loss"] = loss

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
                    self._post_epoch_valid_best()
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

        # finish training, test, save model and write logs
        self._load_weight(self.best_params)
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

    def _checkpoint(self, cur_epoch, best_res, checkpoint_dir=None):
        torch.save(
            {
                "earlystop": self.early_stop.state_dict(),
                "model": self.state_dict(),
                "optim": self.optimizer.state_dict(),
                "epoch": cur_epoch,
                "best_res": best_res,
                "best_params": self.best_params,
                "best_network_params": self.best_network_params,
            },
            self.checkpoint_dir / "resume.pth" if checkpoint_dir is None else checkpoint_dir / "resume.pth",
        )
        print(
            f"Checkpoint saved to {self.checkpoint_dir / 'resume.pth' if checkpoint_dir is None else checkpoint_dir / 'resume.pth'}",
            __name__,
        )

    def _resume(self):
        if (self.checkpoint_dir / "resume.pth").exists():
            print(f"Resume from {self.checkpoint_dir / 'resume.pth'}", __name__)
            checkpoint = torch.load(self.checkpoint_dir / "resume.pth")
            self.early_stop.load_state_dict(checkpoint["earlystop"])
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optim"])
            self.best_params = checkpoint["best_params"]
            self.best_network_params = checkpoint["best_network_params"]
            return checkpoint["epoch"], checkpoint["best_res"]

        print(f"No checkpoint found in {self.checkpoint_dir}", __name__)
        return 0, {}

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
            shuffle=False,
            pin_memory=True,
            num_workers=8,
        )
        self.eval()
        eval_loss = AverageMeter()
        eval_global_tracker = GlobalTracker(self.metrics, self.metric_fn)
        start_time = time.time()
        validset.load()
        with torch.no_grad():
            for _, (data, label) in enumerate(loader):
                if use_cuda():
                    data, label = to_torch(data, device="cuda"), to_torch(label, device="cuda")
                pred = self(data)
                if self.out_ranges is not None:
                    pred = pred[:, self.out_ranges]
                    label = label[:, self.out_ranges]
                loss = self.loss_fn(label.squeeze(-1), pred.squeeze(-1))
                loss = loss.item()
                eval_loss.update(loss, np.prod(label.shape))
                eval_global_tracker.update(label, pred)

        eval_time = time.time() - start_time
        loss = eval_loss.performance()  # loss
        start_time = time.time()
        eval_global_tracker.concat()
        metric_res = eval_global_tracker.performance()
        metric_time = time.time() - start_time
        metric_res["loss"] = loss

        if epoch is not None:
            printt(f"{epoch}\t'valid'\tTime:{eval_time:.2f}\tMetricT: {metric_time:.2f}")
            for metric, value in metric_res.items():
                printt(f"{metric}: {value:.4f}")
            print(f"{datetime.datetime.today()}")
            for k, v in metric_res.items():
                self.writer.add_scalar(f"{k}/valid", v, epoch)

        return metric_res

    def load(self, model_path: str, strict=True):
        """Load the model parameter from model path

        Args:
            model_path (str): The location where the model parameters are saved.
            strict (bool, optional): [description]. Defaults to True.
        """
        self.load_state_dict(torch.load(model_path, map_location="cpu"), strict=strict)

    def predict(self, dataset: Dataset, name: str):
        """Output the prediction on given data.

        Args:
            datasets (Dataset): The dataset to predict on.
            name (str): The results would be saved to {name}_pre.pkl.

        Returns:
            np.ndarray: The model output.
        """
        self.eval()
        preds = []
        dataset.load()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8)
        for _, (data, _) in enumerate(loader):
            if use_cuda():
                data = to_torch(data, device="cuda")
            pred = self(data)
            if self.out_ranges is not None:
                pred = pred[:, self.out_ranges]
            pred = pred.squeeze(-1).cpu().detach().numpy()
            preds.append(pred)

        prediction = np.concatenate(preds, axis=0)
        data_length = len(dataset.get_index())
        prediction = prediction.reshape(data_length, -1)

        # obtain reverse labels
        label = dataset.get_label(prediction)
        if label is not None:
            label = label.reshape(data_length, -1)
            prediction = np.concatenate((prediction, label), axis=-1)

        prediction = pd.DataFrame(data=prediction, index=dataset.get_index())
        prediction.to_pickle(self.checkpoint_dir / (name + "_pre.pkl"))
        return prediction
