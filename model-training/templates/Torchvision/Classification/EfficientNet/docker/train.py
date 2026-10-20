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

import os

from args import get_args
from efficientnet import EfficientNet
from efficientnet_datamodule import EfficientNetDataModule
from util import (
    get_model,
    train_and_test_model,
    create_model_filepath,
    save_model,
    get_num_classes,
)

from export import export_model
from metrics_callback import mark_saved

if __name__ == "__main__":
    ARGS = get_args()

    num_classes = get_num_classes(os.path.join(ARGS.labels_dir, "val", "labels.json"))

    efficientnet_data: EfficientNetDataModule = EfficientNetDataModule(
        images_dir=ARGS.images_dir,
        labels_dir=ARGS.labels_dir,
        batch_size=ARGS.batch_size,
        img_size=ARGS.img_size,
        num_classes=num_classes,
    )

    model = get_model(
        architecture=ARGS.architecture,
        pretrained=ARGS.pretrained,
        num_classes=num_classes,
    )

    efficientnet: EfficientNet = EfficientNet(
        model=model,
        epochs=ARGS.epochs,
        batch_size=ARGS.batch_size,
        optimizer=ARGS.optimizer,
        learning_rate_scheduler=ARGS.lr_scheduler,
        learning_rate_step_size=ARGS.lr_step_size,
        learning_rate=ARGS.learning_rate,
        momentum=ARGS.momentum,
        weight_decay=ARGS.weight_decay,
        num_classes=num_classes,
        metrics_average=ARGS.metrics_average,
        strategy=ARGS.strategy,
        label_smoothing=ARGS.label_smoothing,
    )

    trained_efficientnet: EfficientNet = train_and_test_model(
        efficientnet=efficientnet,
        epochs=ARGS.epochs,
        data=efficientnet_data,
        swa_lrs=ARGS.swa_learning_rate,
        auto_lr_find=ARGS.auto_learning_rate,
        auto_batch_size=ARGS.auto_batch_size,
        patience=ARGS.patience,
        strategy=ARGS.strategy,
    )

    model_filepath = create_model_filepath(output_dir=ARGS.output_dir)

    export_model(
        model=trained_efficientnet,
        model_filepath=model_filepath,
        model_format="pytorch",
        default_model_export_func=save_model,
    )

    mark_saved()
