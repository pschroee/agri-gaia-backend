# SPDX-FileCopyrightText: 2026 Philipp Schröer
#
# SPDX-License-Identifier: MIT

"""B18/T1: the build container's copy of a training image is removed after a
successful push, only when asked for; failures behave as specified.

No infrastructure is needed: Docker clients are replaced before the backend is
imported. Run with `--noconftest`.
"""

import inspect
import os
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

from agri_gaia_backend.routers import train
from agri_gaia_backend.services.docker import image_builder as ib

REPO, TAG = "registry.local/user/torchvision-efficientnet", "2026-01-01T00-00-00-train"


@pytest.fixture
def fake_docker(monkeypatch):
    fake = MagicMock()
    fake.buildx.build.return_value = iter(["#1 [1/2] FROM base\n", "#2 pushing\n"])
    monkeypatch.setattr(ib, "docker", fake)
    return fake


def _build(statuses=None, **kwargs):
    return ib._build_and_push_image(
        context_directory="/ctx",
        repository_url=REPO,
        image_tag=TAG,
        status_callback=(lambda kind, info: statuses.append(kind))
        if statuses is not None
        else (lambda *_: None),
        **kwargs,
    )


def test_removes_image_once_after_last_log(fake_docker):
    order = []

    def logs():
        for chunk in ("a", "b"):
            order.append("log")
            yield chunk

    fake_docker.buildx.build.return_value = logs()
    fake_docker.image.remove.side_effect = lambda *_: order.append("remove")
    _build(remove_after_push=True)
    fake_docker.image.remove.assert_called_once_with(f"{REPO}:{TAG}")
    assert order == ["log", "log", "remove"]


def test_default_does_not_remove(fake_docker):
    _build()
    fake_docker.image.remove.assert_not_called()


def test_explicit_false_does_not_remove(fake_docker):
    _build(remove_after_push=False)
    fake_docker.image.remove.assert_not_called()


def test_build_failure_does_not_remove_and_raises(fake_docker):
    def failing():
        yield "a"
        raise python_on_whales.exceptions.DockerException(["docker", "buildx"], 1)

    fake_docker.buildx.build.return_value = failing()
    statuses = []
    with pytest.raises(python_on_whales.exceptions.DockerException):
        _build(statuses, remove_after_push=True)
    fake_docker.image.remove.assert_not_called()
    assert statuses[-1] == "failed"


def test_remove_failure_keeps_build_successful_and_warns(fake_docker, caplog):
    fake_docker.image.remove.side_effect = RuntimeError("no such image")
    statuses = []
    with caplog.at_level("WARNING", logger="api-logger"):
        _build(statuses, remove_after_push=True)  # must not raise
    assert "failed" not in statuses
    assert any(f"{REPO}:{TAG}" in r.getMessage() for r in caplog.records)


def test_flag_is_not_passed_to_buildx(fake_docker):
    _build(remove_after_push=True)
    assert "remove_after_push" not in fake_docker.buildx.build.call_args.kwargs


def test_only_the_training_build_requests_removal():
    assert "remove_after_push=True" in inspect.getsource(train.build_train_image)
    source = inspect.getsource(ib)
    assert source.count("remove_after_push=True") == 0
