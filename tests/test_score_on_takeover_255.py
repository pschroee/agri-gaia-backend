# SPDX-FileCopyrightText: 2026 Philipp Schröer
#
# SPDX-License-Identifier: MIT

"""Tests for #255: the score of a training is read when its model is taken over.

No infrastructure is needed (same fakes as test_build_gpu_score_247.py).
Run with `--noconftest`.
"""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("MINIO_ENDPOINT", "minio:9000")

import python_on_whales
import docker
import docker.tls

python_on_whales.DockerClient = MagicMock()
docker.APIClient = MagicMock()
docker.tls.TLSConfig = MagicMock()
docker.from_env = MagicMock()
docker.DockerClient = MagicMock()
import python_on_whales.docker_client

python_on_whales.docker_client.DockerClient = MagicMock()

import pytest
import requests
from fastapi import HTTPException
from unittest import mock

with mock.patch("requests.post", side_effect=requests.HTTPError("no portainer")):
    from agri_gaia_backend.routers import train as train_router

LIGHTNING_TABLE = """
┃        Test metric        ┃       DataLoader 0        ┃
│   MulticlassAccuracy      │    1.0                    │
│         test_loss         │    0.04120000079274178    │
"""


def _container(score=None):
    return SimpleNamespace(
        id=1,
        container_id="c",
        score=score,
        score_regexp=r"val_acc=([0-9.]+)",
        score_name="Acc.",
        architecture="EfficientNet",
        provider="Torchvision",
        category="Detection",
        dataset_id=7,
        model_filepath="model.pt",
    )


def _setup(monkeypatch, container, logs, status="exited"):
    updated = []
    monkeypatch.setattr(train_router.sql_api, "get_train_container", lambda db, _id: container)
    monkeypatch.setattr(train_router.sql_api, "update_train_container", lambda db, tc: updated.append(tc))
    monkeypatch.setattr(train_router, "get_container_status", lambda _id: status)
    monkeypatch.setattr(
        train_router,
        "get_container",
        lambda _id: SimpleNamespace(logs=lambda tail=None: logs.encode()),
    )
    monkeypatch.setattr(
        train_router.dataset_sql_api, "get_dataset", lambda db, _id: SimpleNamespace(name="ds")
    )
    monkeypatch.setattr(train_router, "read_from_container", lambda _id, path: b"data" if path.endswith(".pt") else None)
    names = []

    def persist(**kwargs):
        names.append(kwargs["name"])
        return SimpleNamespace(output_shape=None)

    monkeypatch.setattr(train_router, "persist_model", persist)
    monkeypatch.setattr(train_router, "persist_model_artifact", lambda **kw: None)
    return updated, names


def _take_over():
    request = SimpleNamespace(user=SimpleNamespace())
    return asyncio.run(train_router.get_train_container_model(request, 1, db=MagicMock()))


# ---- helper -----------------------------------------------------------------


def test_helper_reads_lightning_table(monkeypatch):
    _setup(monkeypatch, _container(), LIGHTNING_TABLE)
    assert train_router.read_score_from_logs(_container()) == 1.0


def test_helper_prefers_template_regexp(monkeypatch):
    _setup(monkeypatch, _container(), "val_acc=0.5\n" + LIGHTNING_TABLE)
    assert train_router.read_score_from_logs(_container()) == 0.5


def test_helper_returns_none_without_match(monkeypatch):
    _setup(monkeypatch, _container(), "nothing here")
    assert train_router.read_score_from_logs(_container()) is None


# ---- route unchanged --------------------------------------------------------


def test_route_400_while_running(monkeypatch):
    _setup(monkeypatch, _container(), LIGHTNING_TABLE, status="running")
    with pytest.raises(HTTPException) as exc:
        train_router.update_train_container_score(1, db=MagicMock())
    assert exc.value.status_code == 400


def test_route_404_without_score(monkeypatch):
    _setup(monkeypatch, _container(), "nothing here")
    with pytest.raises(HTTPException) as exc:
        train_router.update_train_container_score(1, db=MagicMock())
    assert exc.value.status_code == 404


def test_route_sets_score(monkeypatch):
    container = _container()
    updated, _ = _setup(monkeypatch, container, LIGHTNING_TABLE)
    train_router.update_train_container_score(1, db=MagicMock())
    assert container.score == 1.0 and updated


# ---- takeover ---------------------------------------------------------------


def test_takeover_sets_score_and_name(monkeypatch):
    container = _container()
    updated, names = _setup(monkeypatch, container, LIGHTNING_TABLE)
    _take_over()
    assert container.score == 1.0 and updated
    assert names == ["EfficientNet (Torchvision) [ds, 1.0 Acc.]"]


def test_takeover_without_value_keeps_plain_name(monkeypatch):
    container = _container()
    updated, names = _setup(monkeypatch, container, "nothing here")
    _take_over()
    assert container.score is None and not updated
    assert names == ["EfficientNet (Torchvision) [ds]"]


def test_takeover_keeps_existing_score(monkeypatch):
    container = _container(score=0.8)
    updated, names = _setup(monkeypatch, container, LIGHTNING_TABLE)
    _take_over()
    assert not updated and names == ["EfficientNet (Torchvision) [ds, 0.8 Acc.]"]


def test_takeover_survives_failing_log_read(monkeypatch):
    container = _container()
    _, names = _setup(monkeypatch, container, "")

    def logs(tail=None):
        if tail is not None:
            raise RuntimeError("gone")
        return b""

    monkeypatch.setattr(train_router, "get_container", lambda _id: SimpleNamespace(logs=logs))
    _take_over()
    assert names == ["EfficientNet (Torchvision) [ds]"]
