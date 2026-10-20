#!/usr/bin/env python

# SPDX-FileCopyrightText: 2024 Osnabrück University of Applied Sciences
# SPDX-FileContributor: Andreas Schliebitz
# SPDX-FileContributor: Henri Graf
# SPDX-FileContributor: Jonas Tüpker
# SPDX-FileContributor: Lukas Hesse
# SPDX-FileContributor: Maik Fruhner
# SPDX-FileContributor: Prof. Dr.-Ing. Heiko Tapken
# SPDX-FileContributor: Tobias Wamhof
# SPDX-FileContributor: Philipp Schröer
#
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-

import torch
import inspect
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, Dict, Optional
from util import create_metrics, get_sync_dist
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


class EfficientNet(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes: int,
        epochs: int = 100,
        batch_size: int = 32,
        optimizer: str = "AdamW",
        learning_rate_scheduler: str = "ReduceLROnPlateau",
        learning_rate_step_size: int = 40,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        metrics_average: str = "micro",
        strategy: Optional[str] = None,
        label_smoothing: float = 0,
    ):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_rate_step_size = learning_rate_step_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.strategy = strategy
        self.sync_dist = get_sync_dist(strategy)
        self.label_smoothing = label_smoothing

        metrics = create_metrics(
            num_classes=num_classes, average=metrics_average, strategy=strategy
        )
        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone()

        # Per-epoch values read by the JsonlMetrics callback.
        self._train_loss_sum = 0.0
        self._train_loss_count = 0
        self.epoch_stats: Dict[str, float] = {}

    def forward(self, images: Tensor):
        return self.model(images)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict:
        images, targets = batch
        logits = self.model(images)
        class_probabilities = F.softmax(logits, dim=1)
        train_loss = F.cross_entropy(
            logits, targets, label_smoothing=self.label_smoothing
        )
        self.log(
            "train_loss",
            train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.sync_dist,
        )
        self.train_metrics(class_probabilities, targets)
        self._train_loss_sum += float(train_loss.detach())
        self._train_loss_count += 1
        return train_loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        images, targets = batch
        logits = self.model(images)
        class_probabilities = F.softmax(logits, dim=1)
        val_loss = F.cross_entropy(
            logits, targets, label_smoothing=self.label_smoothing
        )
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.sync_dist,
        )
        self.val_metrics(class_probabilities, targets)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self.val_metrics.reset()
            return

        values = {k: float(v) for k, v in self.val_metrics.compute().items()}
        stats = {}
        for key, value in values.items():
            if "Accuracy" in key:
                stats["val_acc"] = value
            elif "F1" in key:
                stats["val_f1"] = value
        loss = self.trainer.callback_metrics.get("val_loss")
        if loss is not None:
            stats["val_loss"] = float(loss)
        self.epoch_stats = stats

        self.log_dict(
            self.val_metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=self.sync_dist,
        )

    def on_train_epoch_end(self) -> None:
        self.log_dict(
            self.train_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.sync_dist,
        )
        self._train_loss_sum = 0.0
        self._train_loss_count = 0

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict:
        images, targets = batch
        logits = self.model(images)
        class_probabilities = F.softmax(logits, dim=1)
        self.test_metrics(class_probabilities, targets)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics, sync_dist=self.sync_dist)

    def configure_optimizers(self):
        optimizer = eval(f"torch.optim.{self.optimizer}")
        supported_optimizer_args = set(
            inspect.signature(optimizer.__init__).parameters.keys()
        )
        optimizer_args = {
            k: v
            for k, v in {
                "lr": self.learning_rate,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay,
            }.items()
            if v is not None and k in supported_optimizer_args
        }
        optimizer = optimizer(self.parameters(), **optimizer_args)

        if self.learning_rate_scheduler == "ReduceLROnPlateau":
            learning_rate_scheduler = ReduceLROnPlateau(optimizer)
            self.learning_rate_step_size = None
        elif self.learning_rate_scheduler == "StepLR":
            learning_rate_scheduler = StepLR(
                optimizer, step_size=self.learning_rate_step_size
            )
        else:
            raise ValueError(
                f"Unsupported learning rate scheduler '{self.learning_rate_scheduler}'."
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": learning_rate_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "train_loss",
                "strict": True,
                "name": None,
            },
        }
