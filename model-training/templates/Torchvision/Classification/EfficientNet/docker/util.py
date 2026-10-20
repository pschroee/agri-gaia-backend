#!/usr/bin/env python

# SPDX-FileCopyrightText: 2024 Osnabrück University of Applied Sciences
# SPDX-FileContributor: Andreas Schliebitz
# SPDX-FileContributor: Henri Graf
# SPDX-FileContributor: Jonas Tüpker
# SPDX-FileContributor: Lukas Hesse
# SPDX-FileContributor: Maik Fruhner
# SPDX-FileContributor: Prof. Dr.-Ing. Heiko Tapken
# SPDX-FileContributor: Tobias Wamhof
#
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-

import json
import torchvision
import pytorch_lightning as pl

from torch import nn
from pathlib import Path
from efficientnet_datamodule import EfficientNetDataModule
from torchvision.models.efficientnet import EfficientNet as TEfficientNet

from typing import Optional, Union, TYPE_CHECKING
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.tuner import Tuner
from metrics_callback import JsonlMetrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score

if TYPE_CHECKING:
    from efficientnet import EfficientNet


def get_weights_name(architecture: str) -> str:
    _, version = architecture.split("_", maxsplit=1)
    return f"EfficientNet_{version.upper()}_Weights"


def get_model(
    architecture: str,
    pretrained: bool,
    num_classes: int,
) -> TEfficientNet:
    weights_name = get_weights_name(architecture)
    weights = f"torchvision.models.{weights_name}.DEFAULT" if pretrained else None

    model = eval(f"torchvision.models.{architecture}(weights={weights})")
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    return model


def train_and_test_model(
    efficientnet: "EfficientNet",
    epochs: int,
    data: EfficientNetDataModule,
    swa_lrs: Optional[float],
    auto_lr_find: bool,
    auto_batch_size: bool,
    patience: Optional[int],
    strategy: Optional[str],
) -> "EfficientNet":
    auto_scale_batch_size = False
    if auto_batch_size:
        auto_scale_batch_size = "binsearch"
        print(
            f"Automatically determining batch size using '{auto_scale_batch_size}' method."
        )

    callbacks = [JsonlMetrics()]
    if swa_lrs:
        assert swa_lrs > 0, "SWA learning rate has to be positive."

        # See: https://github.com/Lightning-AI/lightning/issues/14755
        auto_lr_find = False
        auto_scale_batch_size = False
        print(
            "Disabled auto learning rate and batch size finders as they are currently incompatible with SWA."
        )
        callbacks.append(StochasticWeightAveraging(swa_lrs=swa_lrs))
        print("Using Stochastic Weight Averaging with learning rate:", swa_lrs)
    else:
        print("Stochastic Weight Averaging is disabled.")

    if patience:
        assert patience > 0, "Early stopping patience has to be positive."
        callbacks.append(
            EarlyStopping(
                monitor="train_loss",
                mode="min",
                patience=patience,
                check_on_train_epoch_end=True,
            )
        )
    else:
        print("Early Stopping is disabled.")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        strategy=configure_strategy(strategy),
        callbacks=callbacks,
    )

    if auto_lr_find or auto_batch_size:
        tuner = Tuner(trainer)
        
        if auto_batch_size:
            print(f"Automatically determining batch size using 'binsearch' method.")
            tuner.scale_batch_size(efficientnet, datamodule=data, mode="binsearch")
            
        if auto_lr_find:
            print("Finding optimal learning rate...")
            # This automatically updates efficientnet.learning_rate or efficientnet.lr
            tuner.lr_find(efficientnet, datamodule=data)
            print("Automatically determined learning rate:", efficientnet.learning_rate)

    trainer.fit(model=efficientnet, datamodule=data)

    test_trainer = pl.Trainer(devices=1, num_nodes=1, accelerator="auto")
    test_trainer.test(model=efficientnet, datamodule=data)

    return efficientnet


def configure_strategy(
    strategy: Optional[str],
) -> Union[DDPStrategy, Optional[str]]:
    if strategy == "ddp":
        return DDPStrategy(find_unused_parameters=False)

    if strategy == "ddp_spawn":
        return DDPStrategy(start_method="spawn", find_unused_parameters=False)

    return "auto"


def get_num_classes(labels_filepath: str) -> int:
    with open(labels_filepath, "r") as fh:
        return len(json.load(fh)["categories"])


def get_sync_dist(strategy: Optional[str]) -> bool:
    return strategy is not None and (
        strategy.startswith("ddp") or strategy.startswith("dp")
    )


def create_metrics(
    num_classes: int, average: str, strategy: Optional[str], task='multiclass'
) -> MetricCollection:
    sync_on_compute = strategy == "dp"

    metric_params = {
        "num_classes": num_classes,
        "average": average,
        "sync_on_compute": sync_on_compute,
        "task": task
    }

    return MetricCollection(
        [
            Accuracy(**metric_params),
            Precision(**metric_params),
            Recall(**metric_params),
            F1Score(**metric_params),
        ]
    )


def create_model_filepath(output_dir: str) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path.joinpath("model.pt"))


def save_model(trained_efficientnet: "EfficientNet", model_filepath: str) -> None:
    trained_efficientnet.to_torchscript(file_path=model_filepath)
