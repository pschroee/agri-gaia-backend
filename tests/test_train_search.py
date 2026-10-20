# SPDX-FileCopyrightText: 2026 Philipp Schröer
#
# SPDX-License-Identifier: MIT

"""Tests for the metrics and clone routes of the training search (issue #168).

The Docker calls are replaced; the parsing tests need no infrastructure.
"""

import datetime
import importlib
import json
import os
import sys
import types

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from starlette.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_400_BAD_REQUEST

from agri_gaia_backend.db import dataset_api, train_api
from agri_gaia_backend.routers import train as train_router
from agri_gaia_backend.util import train as train_util
from agri_gaia_backend.util.train_metrics import (
    metrics_supported,
    parse_gpu_csv,
    parse_metrics_jsonl,
)

TEMPLATE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model-training",
    "templates",
    "Torchvision",
    "Classification",
    "EfficientNet",
    "docker",
)


class TestMetricsParsing:
    def test_jsonl_keeps_objects_and_skips_broken_lines(self):
        data = b'{"epoch":1,"val_acc":0.5}\nnot json\n[1]\n\n{"epoch":2}\n'
        assert parse_metrics_jsonl(data) == [
            {"epoch": 1, "val_acc": 0.5},
            {"epoch": 2},
        ]

    def test_jsonl_returns_the_last_lines(self):
        data = "\n".join(json.dumps({"epoch": i}) for i in range(10)).encode()
        assert [r["epoch"] for r in parse_metrics_jsonl(data, limit=3)] == [7, 8, 9]

    def test_jsonl_empty(self):
        assert parse_metrics_jsonl(None) == []
        assert parse_metrics_jsonl(b"") == []

    def test_gpu_csv_converts_time_to_utc(self):
        rows = parse_gpu_csv(b"2026/10/20 12:00:00.123, 45, 1234, 16384\n")
        assert rows == [
            {
                "at": "2026-10-20T10:00:00Z",
                "util": 45.0,
                "mem_used_mb": 1234.0,
                "mem_total_mb": 16384.0,
            }
        ]

    def test_gpu_csv_skips_malformed_lines(self):
        data = b"garbage\n2026/10/20 12:00:00.1, x, 1, 2\n2026/10/20 12:00:05.1, 3, 4, 5, 6\n"
        assert parse_gpu_csv(data) == []

    def test_supported_needs_the_label(self):
        assert metrics_supported({"agrigaia.metrics": "jsonl/1"})
        assert not metrics_supported({"agrigaia.metrics": "jsonl/2"})
        assert not metrics_supported({})
        assert not metrics_supported(None)


def test_merge_train_config_overrides_the_baked_config():
    merged = json.loads(
        train_util.merge_train_config(
            {"epochs": 5, "batch-size": 32}, {"batch-size": 64, "optimizer": "SGD"}
        )
    )
    assert merged == {"epochs": 5, "batch-size": 64, "optimizer": "SGD"}


class TestMetricsCallback:
    @pytest.fixture
    def callback_module(self):
        pytest.importorskip("pytorch_lightning")
        sys.path.insert(0, TEMPLATE_DIR)
        try:
            yield importlib.import_module("metrics_callback")
        finally:
            sys.path.remove(TEMPLATE_DIR)
            sys.modules.pop("metrics_callback", None)

    def test_record_leaves_out_missing_and_non_finite_values(self, callback_module):
        record = callback_module.build_record(
            2,
            10,
            {"train_loss": 0.5, "val_loss": float("nan"), "val_acc": None, "lr": 1e-3},
            at="2026-10-20T10:00:00Z",
        )
        assert record == {
            "epoch": 2,
            "epochs": 10,
            "at": "2026-10-20T10:00:00Z",
            "train_loss": 0.5,
            "lr": 1e-3,
        }

    def test_lines_are_appended_and_the_last_one_is_repeated_as_saved(
        self, callback_module, tmp_path
    ):
        path = str(tmp_path / "train" / "metrics.jsonl")
        for epoch in (1, 2):
            callback_module.append_record(
                callback_module.build_record(epoch, 2, {"val_acc": 0.1 * epoch}), path
            )
        assert callback_module.mark_saved(path)
        rows = parse_metrics_jsonl(open(path, "rb").read())
        assert [r["epoch"] for r in rows] == [1, 2, 2]
        assert "saved" not in rows[1] and rows[2]["saved"] is True

    def test_mark_saved_without_file(self, callback_module, tmp_path):
        assert not callback_module.mark_saved(str(tmp_path / "missing.jsonl"))


@pytest.fixture
def base_container(db: Session, test_user):
    # A dataset row is needed for the foreign key; reuse any existing one.
    dataset = dataset_api.get_datasets(db, limit=1)
    if not dataset:
        pytest.skip("needs one dataset in the test database")
    base = train_api.create_train_container(
        db,
        owner=test_user.username,
        image_id="registry.invalid/test/torchvision-efficientnet_b0:t-train",
        repository=f"{test_user.username}/torchvision-efficientnet_b0",
        tag="t-train",
        last_modified=datetime.datetime.now(),
        provider="Torchvision",
        category="Classification",
        architecture="EfficientNet",
        dataset_id=dataset[0].id,
        dataset=dataset[0],
        model_filepath="/train/model.pt",
        score_name="Acc.",
        score_regexp="x",
    )
    yield base
    for row in list(db.query(type(base)).filter(type(base).image_id == base.image_id)):
        train_api.delete_train_container(db, row)


class TestCloneRoute:
    def test_clone_copies_the_base_and_keeps_the_override(
        self, authenticated_client: TestClient, base_container, db: Session, monkeypatch
    ):
        monkeypatch.setattr(train_router, "validate", lambda *a, **k: None)
        config = {"epochs": 3, "batch-size": 16}
        response = authenticated_client.post(
            f"/train/containers/{base_container.id}/clone",
            json={"train_config": config},
        )
        assert response.status_code == HTTP_201_CREATED, response.text
        body = response.json()
        assert body["id"] != base_container.id
        assert body["image_id"] == base_container.image_id
        assert body["dataset_id"] == base_container.dataset_id
        assert body["container_id"] is None and body["score"] is None
        row = train_api.get_train_container(db, body["id"])
        assert row.config_override == config

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"train_config": []},
            {"train_config": {}, "image_id": "evil"},
            {"train_config": {}, "owner": "someone"},
        ],
    )
    def test_clone_rejects_other_bodies(
        self, authenticated_client: TestClient, base_container, body
    ):
        response = authenticated_client.post(
            f"/train/containers/{base_container.id}/clone", json=body
        )
        assert response.status_code == HTTP_400_BAD_REQUEST

    def test_clone_of_a_missing_container_is_404(
        self, authenticated_client: TestClient
    ):
        response = authenticated_client.post(
            "/train/containers/999999999/clone", json={"train_config": {}}
        )
        assert response.status_code == 404


class TestDeleteKeepsSharedImage:
    def test_image_stays_while_a_clone_uses_it(
        self, authenticated_client: TestClient, base_container, monkeypatch
    ):
        removed = []
        monkeypatch.setattr(train_router, "validate", lambda *a, **k: None)
        monkeypatch.setattr(
            train_router, "remove_image", lambda name, force=False: removed.append(name)
        )
        clone = authenticated_client.post(
            f"/train/containers/{base_container.id}/clone", json={"train_config": {}}
        ).json()

        assert (
            authenticated_client.delete(f"/train/containers/{base_container.id}").status_code
            == 204
        )
        assert removed == []
        assert authenticated_client.delete(f"/train/containers/{clone['id']}").status_code == 204
        assert removed == [base_container.image_id]


class TestMetricsRoute:
    def _fake_container(self, labels):
        return types.SimpleNamespace(
            id="abc", image=types.SimpleNamespace(labels=labels)
        )

    def test_reads_both_files_of_a_finished_container(
        self, authenticated_client: TestClient, base_container, db: Session, monkeypatch
    ):
        base_container.container_id = "abc"
        train_api.update_train_container(db, base_container)
        files = {
            "/train/metrics.jsonl": b'{"epoch":1,"epochs":2,"val_acc":0.4}\n',
            "/train/gpu.csv": b"2026/10/20 12:00:00.1, 50, 100, 200\n",
        }
        monkeypatch.setattr(train_router, "container_exists", lambda cid: True)
        monkeypatch.setattr(
            train_router,
            "get_container",
            lambda cid: self._fake_container({"agrigaia.metrics": "jsonl/1"}),
        )
        monkeypatch.setattr(
            train_router, "read_from_container", lambda cid, path: files.get(path)
        )
        response = authenticated_client.get(
            f"/train/containers/{base_container.id}/metrics"
        )
        assert response.status_code == HTTP_200_OK
        body = response.json()
        assert body["supported"] is True and body["gpu_available"] is True
        assert body["epochs"] == [{"epoch": 1, "epochs": 2, "val_acc": 0.4}]
        assert body["gpu"][0]["util"] == 50.0

    def test_image_without_label_is_not_supported(
        self, authenticated_client: TestClient, base_container, db: Session, monkeypatch
    ):
        base_container.container_id = "abc"
        train_api.update_train_container(db, base_container)
        monkeypatch.setattr(train_router, "container_exists", lambda cid: True)
        monkeypatch.setattr(
            train_router, "get_container", lambda cid: self._fake_container({})
        )
        body = authenticated_client.get(
            f"/train/containers/{base_container.id}/metrics"
        ).json()
        assert body == {
            "supported": False,
            "epochs": [],
            "gpu": [],
            "gpu_available": False,
        }

    def test_container_without_a_run_has_empty_figures(
        self, authenticated_client: TestClient, base_container, monkeypatch
    ):
        class _Images:
            def get(self, name):
                return types.SimpleNamespace(labels={"agrigaia.metrics": "jsonl/1"})

        monkeypatch.setattr(
            train_router, "docker_host_client", types.SimpleNamespace(images=_Images())
        )
        body = authenticated_client.get(
            f"/train/containers/{base_container.id}/metrics"
        ).json()
        assert body["supported"] is True and body["epochs"] == [] and body["gpu"] == []

    def test_missing_row_is_404(self, authenticated_client: TestClient):
        assert (
            authenticated_client.get("/train/containers/999999999/metrics").status_code
            == 404
        )
