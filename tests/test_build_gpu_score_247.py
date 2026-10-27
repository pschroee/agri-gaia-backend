# SPDX-FileCopyrightText: 2026 Philipp Schröer
#
# SPDX-License-Identifier: MIT

"""Tests for #247: build progress, the GPU request without the backend's NVML,
and model names without "None Acc.".

No infrastructure is needed: Docker clients are replaced before the backend is
imported; nvidia-smi and the container logs are fakes. Run with `--noconftest`.
"""

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
from fastapi import HTTPException

from agri_gaia_backend.routers import train as train_router
from agri_gaia_backend.services.docker import image_builder as ib

# ---- build progress -------------------------------------------------------


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _progress(min_interval=2.0):
    updates, clock = [], _Clock()
    progress = ib.BuildProgress(
        lambda text, fraction: updates.append((text, fraction)),
        min_interval=min_interval,
        clock=clock,
    )
    return progress, updates, clock


def test_progress_starts_with_preparing():
    _, updates, _ = _progress()
    assert updates == [("Preparing the build", 0.01)]


def test_progress_reads_build_steps():
    progress, updates, clock = _progress()
    clock.now += 3
    progress.feed("#7 [builder 3/8] RUN pip install torch")
    assert updates[-1] == (
        "Building the image · step 3/8 · RUN pip install torch",
        round(3 / 8 * 0.85, 3),
    )


def test_progress_ignores_other_lines():
    progress, updates, clock = _progress()
    clock.now += 3
    for line in ("", "#7 DONE 1.2s", "#7 0.5 Collecting torch", "plain text"):
        progress.feed(line)
    assert len(updates) == 1


def test_progress_truncates_long_commands():
    progress, updates, clock = _progress()
    clock.now += 3
    progress.feed("#7 [2/4] RUN " + "x" * 200)
    assert len(updates[-1][0]) < 100 and updates[-1][0].endswith("…")


def test_progress_is_throttled_and_keeps_the_latest_state():
    progress, updates, clock = _progress()
    clock.now += 3
    progress.feed("#7 [1/4] RUN a")
    clock.now += 0.5
    progress.feed("#8 [2/4] RUN b")
    progress.feed("#9 [3/4] RUN c")
    assert len(updates) == 2  # preparing + first step
    progress.finish()
    assert updates[-1][0].startswith("Building the image · step 3/4")
    progress.finish()  # nothing pending any more
    assert len(updates) == 3


def test_progress_never_runs_backwards():
    progress, updates, clock = _progress(min_interval=0)
    progress.feed("#7 [builder 6/8] RUN a")
    progress.feed("#8 [final 1/4] RUN b")
    fractions = [f for _, f in updates]
    assert fractions == sorted(fractions)


def test_progress_reports_push_once_and_stops_building_updates():
    progress, updates, clock = _progress()
    progress.feed("#20 exporting to image")
    progress.feed("#20 pushing layers 3.1s")
    progress.feed("#21 [4/4] RUN late")
    assert updates[1:] == [("Pushing the image to the registry", 0.9)]


def test_build_and_push_reports_phases(monkeypatch):
    lines = ["#1 [1/2] RUN a\n", "#2 exporting to image\n"]
    build = MagicMock(return_value=iter(lines))
    monkeypatch.setattr(ib, "docker", SimpleNamespace(buildx=SimpleNamespace(build=build)))
    phases = []
    ib._build_and_push_image(
        "/ctx",
        "reg/x",
        "t",
        status_callback=lambda *_: None,
        on_phase=lambda text, fraction: phases.append(text),
    )
    assert phases[0] == "Preparing the build"
    assert phases[-1] == "Pushing the image to the registry"
    assert "on_phase" not in build.call_args.kwargs


def test_build_and_push_works_without_on_phase(monkeypatch):
    build = MagicMock(return_value=iter(["#1 [1/2] RUN a\n"]))
    monkeypatch.setattr(ib, "docker", SimpleNamespace(buildx=SimpleNamespace(build=build)))
    ib._build_and_push_image("/ctx", "reg/x", "t", status_callback=lambda *_: None)
    build.assert_called_once()


# ---- GPU request ----------------------------------------------------------


def test_gpu_requested_when_gpus_set_although_backend_cannot_see_it(monkeypatch):
    monkeypatch.setenv("GPUS", "all")
    monkeypatch.setattr(train_router, "gpu_available", lambda: False)
    messages = []
    request = train_router.gpu_device_request_for_run(messages.append)
    assert request is not None
    assert messages == [train_router.GPU_NOT_VISIBLE_MESSAGE]


def test_gpu_loss_is_logged_as_error(monkeypatch):
    monkeypatch.setenv("GPUS", "0")
    monkeypatch.setattr(train_router, "gpu_available", lambda: False)
    logged = []
    monkeypatch.setattr(train_router.logger, "error", lambda msg, *a: logged.append(msg))
    train_router.gpu_device_request_for_run()
    assert logged and "NVML" in logged[0]


def test_gpu_requested_without_message_when_visible(monkeypatch):
    monkeypatch.setenv("GPUS", "0,1")
    monkeypatch.setattr(train_router, "gpu_available", lambda: True)
    messages = []
    request = train_router.gpu_device_request_for_run(messages.append)
    assert request is not None and messages == []


def test_gpu_ids_are_passed_on(monkeypatch):
    monkeypatch.setenv("GPUS", "1")
    monkeypatch.setattr(train_router, "gpu_available", lambda: True)
    request = train_router.gpu_device_request_for_run()
    assert request.device_ids == ["1"]


def test_without_gpus_the_backend_view_decides(monkeypatch):
    monkeypatch.delenv("GPUS", raising=False)
    monkeypatch.setattr(train_router, "gpu_available", lambda: False)
    assert train_router.gpu_device_request_for_run() is None
    monkeypatch.setattr(train_router, "gpu_available", lambda: True)
    assert train_router.gpu_device_request_for_run() is not None


def test_empty_gpus_counts_as_unset(monkeypatch):
    monkeypatch.setenv("GPUS", "")
    monkeypatch.setattr(train_router, "gpu_available", lambda: False)
    assert train_router.gpu_device_request_for_run() is None


# ---- score fallback and model name -----------------------------------------

LIGHTNING_TABLE = """
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│   MulticlassAccuracy      │    0.9871999979019165     │
│         test_loss         │    0.04120000079274178    │
└───────────────────────────┴───────────────────────────┘
"""


class _Db:
    pass


def _score_setup(monkeypatch, logs, regexp):
    container = SimpleNamespace(
        id=1, container_id="c", score=None, score_regexp=regexp, score_name="Acc."
    )
    updated = []
    monkeypatch.setattr(train_router.sql_api, "get_train_container", lambda db, _id: container)
    monkeypatch.setattr(train_router.sql_api, "update_train_container", lambda db, tc: updated.append(tc))
    monkeypatch.setattr(train_router, "get_container_status", lambda _id: "exited")
    monkeypatch.setattr(
        train_router,
        "get_container",
        lambda _id: SimpleNamespace(logs=lambda tail=50: logs.encode()),
    )
    return container, updated


def test_score_falls_back_to_lightning_table(monkeypatch):
    container, updated = _score_setup(monkeypatch, LIGHTNING_TABLE, r"val_acc=([0-9.]+)")
    train_router.update_train_container_score(1, db=_Db())
    assert container.score == 0.987 and updated


def test_score_regexp_of_the_template_wins(monkeypatch):
    container, _ = _score_setup(
        monkeypatch, "val_acc=0.5\n" + LIGHTNING_TABLE, r"val_acc=([0-9.]+)"
    )
    train_router.update_train_container_score(1, db=_Db())
    assert container.score == 0.5


def test_score_still_404_without_any_match(monkeypatch):
    _score_setup(monkeypatch, "nothing here", r"val_acc=([0-9.]+)")
    with pytest.raises(HTTPException) as exc:
        train_router.update_train_container_score(1, db=_Db())
    assert exc.value.status_code == 404


def test_fallback_regexp_takes_the_last_match():
    text = "BinaryAccuracy 0.5\nBinaryAccuracy 0.75"
    assert train_router.FALLBACK_SCORE_REGEXP.findall(text) == ["0.5", "0.75"]


def _tc(score):
    return SimpleNamespace(architecture="EfficientNet", provider="Torchvision", score=score, score_name="Acc.")


def test_model_name_with_score():
    name = train_router.model_name_for_training(_tc(1.0), SimpleNamespace(name="mnist"))
    assert name == "EfficientNet (Torchvision) [mnist, 1.0 Acc.]"


def test_model_name_without_score_has_no_none():
    name = train_router.model_name_for_training(_tc(None), SimpleNamespace(name="mnist"))
    assert name == "EfficientNet (Torchvision) [mnist]" and "None" not in name

