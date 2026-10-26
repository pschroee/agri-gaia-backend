# SPDX-FileCopyrightText: 2026 Philipp Schröer
#
# SPDX-License-Identifier: MIT

"""Tests for the class order of trained models (issue #226).

The Docker, database and MinIO calls are replaced; no infrastructure is needed.
"""

import ast
import asyncio
import json
import os
import subprocess
import sys
import types

import pytest

from agri_gaia_backend.routers import train as train_router
from agri_gaia_backend.util import train as train_util

CVAT_DATASET = os.path.join(
    os.path.dirname(__file__), "..", "model-training", "datasets", "cvat_dataset.py"
)


def _function_source(name: str) -> str:
    """Source of one top-level function of the template, without its heavy imports."""
    with open(CVAT_DATASET, encoding="utf-8") as fh:
        source = fh.read()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(name)


_HELPERS = "from typing import Dict, List\n" + _function_source(
    "_get_tasks"
) + "\n" + _function_source("get_label_names")


def _load_label_names():
    namespace = {}
    exec(_HELPERS, namespace)
    return namespace["get_label_names"]


def _annotation_file(*tasks):
    return {
        "meta": {
            "task": [
                {"labels": {"label": [{"name": name} for name in names]}}
                for names in tasks
            ]
        }
    }


def test_label_names_keep_annotation_order():
    get_label_names = _load_label_names()
    assert get_label_names(_annotation_file(["0", "1"])) == ["0", "1"]
    assert get_label_names(_annotation_file(["1", "0"])) == ["1", "0"]
    assert get_label_names(_annotation_file(["b", "a"], ["c", "a"])) == ["b", "a", "c"]


def test_label_names_same_for_every_hash_seed():
    code = (
        "import json\n"
        + _HELPERS
        + "\nprint(json.dumps(get_label_names("
        + repr(_annotation_file(["zebra", "cow", "pig", "10", "2", "sheep"]))
        + ")))\n"
    )
    results = set()
    for seed in range(6):
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        out = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        results.add(out.strip())
    assert results == {'["zebra", "cow", "pig", "10", "2", "sheep"]'}


def _labels_json(categories) -> bytes:
    return json.dumps({"categories": categories}).encode()


class TestReadClassOrder:
    def test_read_class_order_sorts_by_id(self, monkeypatch):
        data = _labels_json([{"id": 2, "name": "0"}, {"id": 1, "name": "1"}])
        monkeypatch.setattr(train_util, "read_from_container", lambda cid, path: data)
        assert train_util.read_class_order("c1") == ["1", "0"]

    def test_falls_back_to_the_train_split(self, monkeypatch):
        data = _labels_json([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])
        files = {"/datasets/labels/train/labels.json": data}
        monkeypatch.setattr(
            train_util, "read_from_container", lambda cid, path: files.get(path)
        )
        assert train_util.read_class_order("c1") == ["a", "b"]

    @pytest.mark.parametrize(
        "data",
        [
            None,
            b"not json",
            b"{}",
            _labels_json([]),
            _labels_json([{"id": 1, "name": "a"}, {"id": 2, "name": "a"}]),
        ],
    )
    def test_read_class_order_missing_or_broken(self, monkeypatch, data):
        monkeypatch.setattr(train_util, "read_from_container", lambda cid, path: data)
        assert train_util.read_class_order("c1") is None

    def test_read_error_gives_none(self, monkeypatch):
        def boom(cid, path):
            raise RuntimeError("no such container")

        monkeypatch.setattr(train_util, "read_from_container", boom)
        assert train_util.read_class_order("c1") is None


class _FakeDb:
    def commit(self):
        pass

    def refresh(self, obj):
        pass


class TestModelFromTraining:
    def _run(self, monkeypatch, category, output_shape, categories):
        container = types.SimpleNamespace(
            container_id="c1",
            dataset_id=1,
            model_filepath="/train/model.pt",
            architecture="EfficientNet",
            provider="Torchvision",
            score=0.9,
            score_name="Acc.",
            category=category,
        )
        model = types.SimpleNamespace(
            id=7, output_shape=output_shape, output_labels=None
        )
        files = {
            "/train/model.onnx": b"onnx",
            "/datasets/labels/val/labels.json": _labels_json(categories),
        }
        monkeypatch.setattr(train_router, "check_exists", lambda x: x)
        monkeypatch.setattr(
            train_router.sql_api, "get_train_container", lambda db, i: container
        )
        monkeypatch.setattr(
            train_router.dataset_sql_api,
            "get_dataset",
            lambda db, i: types.SimpleNamespace(name="ds"),
        )
        monkeypatch.setattr(
            train_router, "read_from_container", lambda cid, path: files.get(path)
        )
        monkeypatch.setattr(train_util, "read_from_container", lambda cid, path: files.get(path))
        monkeypatch.setattr(
            train_router,
            "get_container",
            lambda cid: types.SimpleNamespace(logs=lambda: b"log"),
        )
        monkeypatch.setattr(train_router, "persist_model", lambda **kw: model)
        monkeypatch.setattr(train_router, "persist_model_artifact", lambda **kw: "x")
        request = types.SimpleNamespace(user=types.SimpleNamespace(username="u"))
        result = asyncio.run(
            train_router.get_train_container_model(request, 1, db=_FakeDb())
        )
        assert result is model
        return result

    TWO = [{"id": 1, "name": "1"}, {"id": 2, "name": "0"}]

    def test_model_from_training_sets_output_labels(self, monkeypatch):
        model = self._run(monkeypatch, "Classification", [1, 2], self.TWO)
        assert model.output_labels == ["1", "0"]

    def test_model_from_training_skips_shape_mismatch(self, monkeypatch):
        three = self.TWO + [{"id": 3, "name": "2"}]
        model = self._run(monkeypatch, "Classification", [1, 2], three)
        assert model.output_labels is None

    def test_model_from_training_skips_other_categories(self, monkeypatch):
        model = self._run(monkeypatch, "Instance Segmentation", [1, 2], self.TWO)
        assert model.output_labels is None
