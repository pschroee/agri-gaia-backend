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

import os
import torch
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import DataLoader
from efficientnet_dataset import EfficientNetDataset

from dataset import load_dataset_filepaths


class EfficientNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        batch_size: int,
        img_size: int,
        num_classes: int,
        num_workers: int = -1,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_workers = os.cpu_count() if num_workers <= 0 else num_workers
        self.pin_memory = torch.cuda.is_available()

        self.train_split = None
        self.test_split = None

    def setup(self, stage: str) -> None:

        self.train_split = EfficientNetDataset(
            root=os.path.join(self.images_dir, "train"),
            annFile=os.path.join(self.labels_dir, "train", "labels.json"),
            img_size=self.img_size,
        )

        self.test_split = EfficientNetDataset(
            root=os.path.join(self.images_dir, "val"),
            annFile=os.path.join(self.labels_dir, "val", "labels.json"),
            img_size=self.img_size,
        )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_split,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        # Validation uses the same "val" split as testing.
        return DataLoader(
            self.test_split,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=True,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_split,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=self.pin_memory,
        )
