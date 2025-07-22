import os
import copy
import time
import json
import datetime
from typing import List, Optional
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utilsd import use_cuda
from utilsd.earlystop import EarlyStop, EarlyStopStatus
from ..metrics import get_loss_fn, get_metric_fn
from ..common.utils import AverageMeter, GlobalTracker, to_torch,printt
from .base import MODELS,BaseModel


@MODELS.register_module()
class MMM_Finetune(BaseModel):
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
        region: int = 17,
        mask_ratio: float=0.,
        network: Optional[nn.Module] = None,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        early_stop: Optional[int] = None,
        out_size: Optional[int] = 3,
        model_path: Optional[str] = None,
        aggregate: str = "mean",
    ):
        self.mask_ratio = mask_ratio
        self.region = region
        self.hyper_paras = {
            "task": task,
            "aggregate": aggregate,
            "out_size": out_size,
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
        )

    def _build_network(
        self,
        network,
        task: str,
        aggregate: str = "all",
        out_size: int = 3,
    ) -> None:
        """Initilize the network parameters

        Args:
            task: the prediction task, classification or regression.
            out_size: the output size for multi-class classification or multi-variant regression task.
            aggregate: whether to aggregate across whole sequence.
        """

        self.network = network
        self.aggregate = aggregate

        # Setup Suernodes
        self.supernodes = nn.Parameter(torch.zeros(1,self.region,network.in_chans))
        torch.nn.init.normal_(self.supernodes)

        # Output layer
        self.fc_norm1 = nn.BatchNorm1d(network.embed_dim*(self.region+1))
        self.fc_out1 = nn.Linear(network.embed_dim*(self.region+1), 64)
        self.fc_norm2 = nn.BatchNorm1d(64)
        self.fc_out2 = nn.Linear(64, out_size)
        if task == "classification":
            self.act_out = nn.Sigmoid()
            out_size = 1
        elif task == "multiclassification":
            self.act_out = nn.Softmax(-1)


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

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.observe = observe
        self.lr = lr
        self.weight_decay = weight_decay

        # Setup early stop
        if early_stop is not None:
            self.early_stop = EarlyStop(
                patience=early_stop, mode="min" if lower_is_better else "max"
            )
        else:
            self.early_stop = EarlyStop(
                patience=max_epochs, mode="min" if lower_is_better else "max"
            )

        self.optimizer = getattr(optim, optimizer)(
            [
                {
                    "params": filter(lambda p: p.requires_grad, self.parameters()),
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
            ],
        )

    def forward(self, inputs, mask_type='random'):
        N, C, F = inputs.shape
        func_areas = [
            [0,1,2,3,4],[5,6,7],[8,9,10,11,12,13],[14,15,16],
            [17,18,19],[20,21,22],[23,24,25],[26,27,28],[29,30,31],
            [32,33,34],[35,36,37,38,39,40],[41,42,43],[44,45,46],
            [47,48,49],[50,51,52],[53,54,55,56,57,58],[59,60,61]
        ]
        if mask_type == 'block':
            mask, _ = random_masking(N, len(func_areas), self.mask_ratio)
            mask_res=[]
            for mask_for_singlebs in mask:
                res=[]
                for i, mask_for_area in enumerate(mask_for_singlebs):
                    res.extend([mask_for_area]*len(func_areas[i]))
                mask_res.append(res)
            mask=torch.tensor(mask_res)
            mask = mask.unsqueeze(-1).repeat(1, 1, F)
        elif mask_type == "random":
            mask, _ = random_masking(N, C, self.mask_ratio)
            mask = mask.unsqueeze(-1).repeat(1, 1, F)
        mask = mask.to(device="cuda")
        mask_inverse = 1 - mask

        inputs = torch.mul(mask_inverse, inputs)
        supernodes=torch.repeat_interleave(self.supernodes,inputs.shape[0],0)
        inputs=torch.cat([inputs,supernodes],1)
        out = self.network(inputs,region=self.region)
        bs = out.shape[0]
        out = out.reshape(bs, -1)
        out = self.fc_norm1(out)
        preds = self.act_out(self.fc_out1(out))
        out = self.fc_norm2(preds)
        preds = self.act_out(self.fc_out2(out))

        return preds

    def load(self, model_path: str, strict=False):
        """Load the model parameter from model path

        Args:
            model_path (str): The location where the model parameters are saved.
            strict (bool, optional): [description]. Defaults to False. 
            **This is not the case in the BaseModel class because MMM_finetune only loads the network(encoder) parameters.**
        """
        state_dict = torch.load(model_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=strict)

    def fit(
        self,
        trainset: Dataset,
        validset: Optional[Dataset] = None,
        testset: Optional[Dataset] = None,
    ) -> nn.Module:
        loader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )
        self.best_acc = 0
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_params = copy.deepcopy(self.state_dict())
        self.best_network_params = copy.deepcopy(self.network.state_dict())
        start_epoch, best_res = self._resume()
        if start_epoch+1 >= self.max_epochs:
            return self.best_acc
        best_epoch = best_res.pop("best_epoch", 0)
        best_score = self.early_stop.best


        for epoch in range(start_epoch + 1, self.max_epochs):
            self.train()
            train_loss = AverageMeter()
            train_global_tracker = GlobalTracker(self.metrics, self.metric_fn)
            start_time = time.time()
            for x, label in loader:
                if use_cuda():
                    x, label = to_torch(x, device="cuda"), to_torch(
                        label, device="cuda"
                    )

                outputs = self(x)
                loss = self.loss_fn(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss = loss.item()
                train_loss.update(loss, np.prod(label.shape))
                train_global_tracker.update(label, outputs)

            train_time = time.time() - start_time
            loss = train_loss.performance()
            start_time = time.time()

            train_global_tracker.concat()
            metric_res = train_global_tracker.performance()
            metric_res["loss"] = loss
            # print log
            printt(
                f"{epoch}\t'train'\tTime:{train_time:.2f}"
            )
            print(f"{datetime.datetime.today()}")
            for metric, value in metric_res.items():
                printt(f"{metric}: {value:.4f}")
            for k, v in metric_res.items():
                self.writer.add_scalar(f"{k}/train", v, epoch)
            self.writer.flush()

            if validset is not None:
                with torch.no_grad():
                    eval_res = self.evaluate(validset)
                value = eval_res[self.observe]
                es = self.early_stop.step(value)
                if es == EarlyStopStatus.BEST:
                    best_score = value
                    best_epoch = epoch
                    self.best_params = copy.deepcopy(self.state_dict())
                    self.best_network_params = copy.deepcopy(self.network.state_dict())
                    best_res = {"train": metric_res, "valid": eval_res}
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
        for kys, vals in best_res.items():
            for ky, val in vals.items():
                best_res[kys][ky] = np.float64(val)
        print(best_res)
        with open(f"{self.checkpoint_dir}/res.json", "w") as f:
            json.dump(best_res, f, indent=4, sort_keys=True)
        keys = list(self.hyper_paras.keys())
        for k in keys:
            if type(self.hyper_paras[k]) not in [int, float, str, bool, torch.Tensor]:
                self.hyper_paras.pop(k)
        self.writer.add_hparams(
            self.hyper_paras, {"result": best_score, "best_epoch": best_epoch}
        )
        self._unload(epoch, {**best_res, "best_epoch": best_epoch})
        return self.best_acc

    def evaluate(self, validset: Dataset, epoch: Optional[int] = None) -> dict:

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
                    data, label = to_torch(data, device="cuda"), to_torch(
                        label, device="cuda"
                    )
                pred = self(data)

                loss = self.loss_fn(pred, label)
                eval_loss.update(loss, data.shape[0])
                eval_global_tracker.update(label, pred)

        eval_time = time.time() - start_time
        loss = eval_loss.performance()  # loss
        eval_global_tracker.concat()
        metric_res = eval_global_tracker.performance()
        metric_res["loss"] = loss

        printt(f"{epoch}\t'valid'\tTime:{eval_time:.2f}\t")
        for metric, value in metric_res.items():
            printt(f"{metric}: {value:.4f}")
        print(f"{datetime.datetime.today()}")
        for k, v in metric_res.items():
            if k == 'accuracy':
                self.best_acc = max(self.best_acc, v)
            self.writer.add_scalar(f"{k}/valid", v, epoch)
        return metric_res

    def _checkpoint(self, cur_epoch, best_res):
        if os.path.exists(self.checkpoint_dir / "down_buffer.pth"):
            try:
                os.remove(self.checkpoint_dir / "down_buffer.pth")
            except Exception:
                pass
        torch.save(
            {
                "earlystop": self.early_stop.state_dict(),
                "model": self.state_dict(),
                "optim": self.optimizer.state_dict(),
                "epoch": cur_epoch,
                "best_res": best_res,
                "best_params": self.best_params,
                'acc': self.best_acc,
                'finish': False,
                # 'scaler': self.loss_scaler.state_dict(),
                "best_network_params": self.best_network_params,
            },
            self.checkpoint_dir / "down_buffer.pth",
        )
        os.replace(self.checkpoint_dir / "down_buffer.pth", self.checkpoint_dir / "down.pth")
        print(f"Checkpoint saved to {self.checkpoint_dir / 'down.pth'}", __name__)

    def _resume(self):
        if (self.checkpoint_dir / "down.pth").exists():
            print(f"Resume from {self.checkpoint_dir /'down.pth'}", __name__)
            checkpoint = torch.load(self.checkpoint_dir / "down.pth")
            self.best_acc = checkpoint['acc']
            if checkpoint["finish"]:
                return checkpoint["epoch"], checkpoint["best_res"]
            self.early_stop.load_state_dict(checkpoint["earlystop"])
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optim"])
            # self.loss_scaler.load_state_dict(checkpoint["scaler"])
            self.best_params = checkpoint["best_params"]
            self.best_network_params = checkpoint["best_network_params"]
            return checkpoint["epoch"], checkpoint["best_res"]

        print(f"No checkpoint found in {self.checkpoint_dir}", __name__)
        return 0, {}

    def _unload(self, cur_epoch, best_res):
        torch.save(
            {
                "epoch": cur_epoch,
                "best_res": best_res,
                'acc': self.best_acc,
                'finish': True,
            },
            self.checkpoint_dir / "down_buffer.pth",
        )
        os.replace(self.checkpoint_dir / "down_buffer.pth", self.checkpoint_dir / "down.pth")
        print(f"Checkpoint saved to {self.checkpoint_dir / 'down.pth'}", __name__)

def random_masking(batch_size, length, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    # N, L, D = x.shape  # batch, length, dim
    len_keep = int(length * (1 - mask_ratio))

    noise = torch.rand(batch_size, length)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([batch_size, length], dtype=torch.long)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return mask, ids_restore

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]
