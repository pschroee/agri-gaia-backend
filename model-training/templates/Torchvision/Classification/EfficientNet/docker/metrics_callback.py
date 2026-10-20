#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 Philipp Schröer
#
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-

"""Writes one JSON line per epoch to /train/metrics.jsonl (label agrigaia.metrics=jsonl/1)."""

import json
import math
import os

from datetime import datetime, timezone
from typing import Dict, Optional

import pytorch_lightning as pl

METRICS_PATH = os.environ.get("AGRIGAIA_METRICS_PATH", "/train/metrics.jsonl")


def _finite(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_record(
    epoch: int, epochs: int, values: Dict[str, Optional[float]], at: Optional[str] = None
) -> Dict:
    """Builds one metrics line; values that are missing or not finite are left out."""
    record = {
        "epoch": int(epoch),
        "epochs": int(epochs),
        "at": at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for key, value in values.items():
        number = _finite(value)
        if number is not None:
            record[key] = number
    return record


def append_record(record: Dict, path: str = METRICS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")


def mark_saved(path: str = METRICS_PATH) -> bool:
    """Repeats the last line with "saved": true once the model file is written."""
    try:
        with open(path, "r") as fh:
            lines = [line for line in fh.read().splitlines() if line.strip()]
        record = json.loads(lines[-1])
    except (OSError, IndexError, ValueError):
        return False
    if not isinstance(record, dict):
        return False
    record["saved"] = True
    append_record(record, path)
    return True


class JsonlMetrics(pl.Callback):
    """Appends the metrics of every finished epoch (training and validation) to a JSONL file.

    Runs before the LightningModule's own on_train_epoch_end, so it reads the
    values the module collected (epoch_stats, loss sums) and computes the
    training metric without resetting it.
    """

    def __init__(self, path: str = METRICS_PATH) -> None:
        super().__init__()
        self.path = path

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        values: Dict[str, Optional[float]] = {}
        count = getattr(pl_module, "_train_loss_count", 0)
        if count:
            values["train_loss"] = pl_module._train_loss_sum / count
        try:
            for key, value in pl_module.train_metrics.compute().items():
                if "Accuracy" in key:
                    values["train_acc"] = float(value)
        except Exception:
            pass
        values.update(getattr(pl_module, "epoch_stats", {}))
        try:
            values["lr"] = trainer.optimizers[0].param_groups[0]["lr"]
        except (IndexError, KeyError):
            pass

        append_record(
            build_record(trainer.current_epoch + 1, trainer.max_epochs, values),
            self.path,
        )
