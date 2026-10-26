# SPDX-FileCopyrightText: 2026 Philipp Schröer
#
# SPDX-License-Identifier: MIT

"""Tests for #234: background tasks open their own database session and get
only ids, and a failed task reports its reason instead of "See backend logs".

No infrastructure is needed: Docker clients are replaced before the backend is
imported, sessions and SQL access are fakes. Run with `--noconftest`.
"""

import asyncio
import json
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
from sqlalchemy.orm import Session

from agri_gaia_backend.db.models import TaskStatus
from agri_gaia_backend.routers import common
from agri_gaia_backend.routers import container_images as ci
from agri_gaia_backend.routers import train as train_router
from agri_gaia_backend.routers import triton as triton_router
from agri_gaia_backend.routers.common import task_error_message
from agri_gaia_backend.schemas.container_image import ContainerImageBuildConfig


class _RequestDb:
    """The session of the request: unusable once the response is sent."""

    def __init__(self):
        object.__setattr__(self, "_closed", False)

    def close(self):
        object.__setattr__(self, "_closed", True)

    def __getattr__(self, name):
        raise AssertionError("request session used after the response")


class _TaskDb:
    """The session a task opens for itself."""

    instances = []

    def __init__(self):
        self.entered = False
        self.closed = False
        _TaskDb.instances.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.closed = True

    def close(self):
        self.closed = True

    def commit(self):
        pass

    def refresh(self, *_):
        pass

    def add(self, *_):
        pass


class _CapturingTaskCreator:
    def __init__(self):
        self.func = None
        self.kwargs = None
        self.title = None

    def create_background_task(self, func, task_title, *args, **kwargs):
        self.func = func
        self.title = task_title
        self.kwargs = kwargs
        return None, "loc", None


def _run(func, kwargs):
    """Runs a captured task the way TaskCreator does; returns collected messages."""
    errors, messages = [], []
    extra = {"on_message": messages.append} if "on_message" in func.__code__.co_varnames else {}
    func(
        on_error=errors.append,
        on_progress_change=lambda _: None,
        **extra,
        **kwargs,
    )
    return errors, messages


@pytest.fixture(autouse=True)
def _reset_sessions(monkeypatch):
    _TaskDb.instances = []
    for module in (train_router, triton_router, ci):
        monkeypatch.setattr(module, "SessionLocal", _TaskDb)


def _request(user=None):
    return SimpleNamespace(user=user or SimpleNamespace(username="u", access_token="t", minio_token="m"))


def _no_request_session(db):
    assert not isinstance(db, _RequestDb), "request session handed to a task"
    assert isinstance(db, _TaskDb)


class _FakeTrainContainer:
    def __init__(self, container_id=None):
        self.id = 1
        self.dataset_id = 5
        self.container_id = container_id
        self.image_id = "reg/img:tag"
        self.provider = "p"
        self.architecture = "a"
        self.model_filepath = "m.pt"
        self.config_override = None
        self.score = 1.0
        self.task_db = None

    @property
    def dataset(self):
        # lazy relationship: only available while the task session is open
        if self.task_db is None or self.task_db.closed:
            raise AssertionError("lazy load on a closed or foreign session")
        return SimpleNamespace(bucket_name="bucket")


def _setup_run(monkeypatch, container):
    state = {"running": False}
    seen = []
    fake_docker = MagicMock()
    fake_docker.containers.create.return_value = SimpleNamespace(
        id="new", start=MagicMock()
    )

    def get_tc(db, _id):
        if isinstance(db, _TaskDb):
            container.task_db = db
        seen.append(db)
        return container

    monkeypatch.setattr(train_router.sql_api, "get_train_container", get_tc)
    monkeypatch.setattr(
        train_router.sql_api, "update_train_container", lambda db, tc: seen.append(db)
    )
    monkeypatch.setattr(
        train_router.dataset_sql_api,
        "get_dataset",
        lambda db, _id: (seen.append(db), SimpleNamespace(id=5, annotation_task_id=None))[1],
    )
    monkeypatch.setattr(
        train_router, "container_exists", lambda _id: state["running"] or False
    )
    monkeypatch.setattr(
        train_router,
        "get_container",
        lambda _id: SimpleNamespace(status="running" if state["running"] else "exited"),
    )
    monkeypatch.setattr(train_router, "pull_image_if_missing", lambda _i: None)
    monkeypatch.setattr(train_router, "gpu_available", lambda: False)
    monkeypatch.setattr(train_router, "docker_host_client", fake_docker)
    monkeypatch.setattr(train_router, "write_train_container_files", lambda *_: None)
    monkeypatch.setattr(train_router, "read_train_container_files", lambda **_: {})
    monkeypatch.setattr(train_router, "remove_container", lambda *_a, **_k: None)
    return state, seen, fake_docker


def _call_run_route(creator, request_db):
    asyncio.run(
        train_router.run_train_container(
            _request(), 1, db=request_db, task_creator=creator
        )
    )
    request_db.close()


def test_run_train_task_gets_only_ids(monkeypatch):
    _setup_run(monkeypatch, _FakeTrainContainer())
    creator = _CapturingTaskCreator()
    _call_run_route(creator, _RequestDb())
    assert set(creator.kwargs) == {"user", "train_container_id"}
    assert not any(isinstance(v, Session) for v in creator.kwargs.values())


def test_run_train_task_uses_own_session(monkeypatch):
    container = _FakeTrainContainer()
    _, seen, fake_docker = _setup_run(monkeypatch, container)
    creator = _CapturingTaskCreator()
    request_db = _RequestDb()
    _call_run_route(creator, request_db)
    seen.clear()
    errors, _ = _run(creator.func, creator.kwargs)
    assert errors == []
    assert seen and all(isinstance(db, _TaskDb) for db in seen)
    fake_docker.containers.create.assert_called_once()
    assert container.container_id == "new"
    assert _TaskDb.instances[-1].closed


def test_run_train_task_lazy_dataset_after_commit(monkeypatch):
    container = _FakeTrainContainer(container_id="old")
    _setup_run(monkeypatch, container)
    creator = _CapturingTaskCreator()
    _call_run_route(creator, _RequestDb())
    errors, _ = _run(creator.func, creator.kwargs)
    assert errors == []
    assert container.container_id == "new"


def test_run_train_task_reports_http_detail(monkeypatch):
    container = _FakeTrainContainer(container_id="c")
    state, _, _ = _setup_run(monkeypatch, container)
    creator = _CapturingTaskCreator()
    _call_run_route(creator, _RequestDb())
    state["running"] = True
    errors, _ = _run(creator.func, creator.kwargs)
    assert errors == ["Train Container #1 is already running."]


def test_run_train_task_reports_exception_type(monkeypatch):
    _setup_run(monkeypatch, _FakeTrainContainer())

    def boom(_image):
        raise RuntimeError("pull failed")

    monkeypatch.setattr(train_router, "pull_image_if_missing", boom)
    creator = _CapturingTaskCreator()
    _call_run_route(creator, _RequestDb())
    errors, _ = _run(creator.func, creator.kwargs)
    assert errors == ["RuntimeError: pull failed"]


def test_stop_train_task_uses_own_session(monkeypatch):
    container = _FakeTrainContainer(container_id="c")
    stopped = MagicMock()
    dbs = []
    monkeypatch.setattr(
        train_router.sql_api,
        "get_train_container",
        lambda db, _id: (dbs.append(db), container)[1],
    )
    monkeypatch.setattr(
        train_router, "get_container", lambda _id: SimpleNamespace(stop=stopped)
    )
    creator = _CapturingTaskCreator()
    train_router.stop_train_container(1, task_creator=creator)
    assert set(creator.kwargs) == {"train_container_id"}
    errors, _ = _run(creator.func, creator.kwargs)
    assert errors == []
    stopped.assert_called_once()
    assert all(isinstance(db, _TaskDb) for db in dbs) and dbs


def test_build_train_image_uses_own_session(monkeypatch, tmp_path):
    for name in ("custom.json", "presets.json"):
        content = (
            {"model_filepath": "m.pt", "score_name": "s", "score_regexp": "r"}
            if name == "custom.json"
            else {}
        )
        (tmp_path / name).write_text(json.dumps(content))
    dbs = []
    monkeypatch.setattr(
        train_router.dataset_sql_api,
        "get_dataset",
        lambda db, _id: (dbs.append(db), SimpleNamespace(id=5))[1],
    )
    created = {}

    def create(db, **kw):
        dbs.append(db)
        created.update(kw)
        return SimpleNamespace(id=7)

    monkeypatch.setattr(train_router.sql_api, "create_train_container", create)
    monkeypatch.setattr(train_router, "copy_tree", lambda *_: None)
    monkeypatch.setattr(train_router, "copy", lambda *_: None)
    monkeypatch.setattr(train_router, "get_dockerfile_filepath", lambda *_: "/x/Dockerfile")
    monkeypatch.setattr(train_router, "get_torch_cuda_index", lambda: None)
    monkeypatch.setattr(train_router, "pull_image_if_missing", lambda _i: None)
    monkeypatch.setattr(train_router.image_builder, "_build_and_push_image", lambda **_: None)
    monkeypatch.setattr(
        train_router.image_builder, "create_repository_url", lambda owner: f"reg/{owner}"
    )
    kwargs = dict(
        provider="p",
        category="c",
        architecture="a b",
        dataset_id=5,
        train_config={},
        export_config=None,
        train_configs_path=str(tmp_path),
        owner="u",
    )
    errors, messages = _run(train_router.build_train_image, kwargs)
    assert errors == []
    assert dbs and all(isinstance(db, _TaskDb) for db in dbs)
    assert messages == ["Train Container #7 created."]
    assert _TaskDb.instances[-1].closed


def test_process_train_config_passes_no_session(monkeypatch):
    monkeypatch.setattr(
        train_router, "get_config_filepath", lambda *_: ("/t/config.json", None)
    )

    class _Req:
        user = SimpleNamespace(username="u")

        async def json(self):
            return {
                "provider": "p",
                "category": "c",
                "architecture": "a",
                "dataset_id": 5,
                "train_config": {},
                "export_config": {},
            }

    creator = _CapturingTaskCreator()
    asyncio.run(train_router.process_train_config(_Req(), task_creator=creator))
    assert "db" not in creator.kwargs
    assert not any(isinstance(v, Session) for v in creator.kwargs.values())


def test_inference_task_passes_no_session():
    creator = _CapturingTaskCreator()
    triton_router.get_tritonInfo(_request(), [], [], "triton:8000", task_creator=creator)
    assert set(creator.kwargs) == {"user", "models", "datasets", "url"}
    errors, _ = _run(creator.func, creator.kwargs)
    assert errors == []
    assert len(_TaskDb.instances) == 1
    assert _TaskDb.instances[0].entered and _TaskDb.instances[0].closed


def test_build_container_task_gets_ids(monkeypatch):
    model, device, template = object(), object(), object()
    dbs = []

    def pick(value):
        def getter(db, _id):
            dbs.append(db)
            return value

        return getter

    monkeypatch.setattr(ci.model_sql_api, "get_model", pick(model))
    monkeypatch.setattr(ci.edge_sql_api, "get_edge_device", pick(device))
    monkeypatch.setattr(ci.container_template_sql_api, "get_container_template", pick(template))
    monkeypatch.setattr(ci.image_builder, "validate_generator_input", lambda *_: None)
    built = []
    monkeypatch.setattr(
        ci.image_builder, "build_and_push_image", lambda **kw: built.append(kw)
    )
    request = _request(SimpleNamespace(username="u"))

    def call(config):
        creator = _CapturingTaskCreator()
        request_db = _RequestDb()
        # the route validates on the request session, which is still open here
        monkeypatch.setattr(
            ci.model_sql_api, "get_model", lambda db, _id: model
        )
        monkeypatch.setattr(
            ci.edge_sql_api, "get_edge_device", lambda db, _id: device
        )
        monkeypatch.setattr(
            ci.container_template_sql_api, "get_container_template", lambda db, _id: template
        )
        ci.build_container(request, config, db=request_db, task_creator=creator)
        request_db.close()
        # the task reloads everything over its own session
        monkeypatch.setattr(ci.model_sql_api, "get_model", pick(model))
        monkeypatch.setattr(ci.edge_sql_api, "get_edge_device", pick(device))
        monkeypatch.setattr(
            ci.container_template_sql_api, "get_container_template", pick(template)
        )
        return creator

    creator = call(
        ContainerImageBuildConfig(
            container_template_id=3, repository="u/x", tag="t", model_id=1, edge_device_id=2
        )
    )
    assert creator.kwargs == {
        "model_id": 1,
        "edge_device_id": 2,
        "architecture": None,
        "container_template_id": 3,
    }
    errors, _ = _run(creator.func, creator.kwargs)
    assert errors == []
    assert built[-1]["model"] is model
    assert built[-1]["edge_info"] is device
    assert built[-1]["inference_container_template"] is template
    assert dbs and all(isinstance(db, _TaskDb) for db in dbs)

    creator = call(
        ContainerImageBuildConfig(
            container_template_id=3, repository="u/x", tag="t", model_id=1, architecture="linux/amd64"
        )
    )
    _run(creator.func, creator.kwargs)
    assert built[-1]["edge_info"] == {"device_type": "generic", "architecture": "linux/amd64"}


def test_task_error_message_rules():
    assert task_error_message(HTTPException(409, "x")) == "x"
    assert task_error_message(ValueError("bad")) == "ValueError: bad"
    assert task_error_message(Exception()) == "Exception"
    long = task_error_message(RuntimeError("y" * 2000))
    assert len(long) == 1000 and long.endswith("…")


class _FakeTask:
    id = 1
    status = None
    message = None
    completion_percentage = 0.0


def _run_task_creator(monkeypatch, func):
    task = _FakeTask()
    task.status = TaskStatus.created
    task.message = None
    monkeypatch.setattr(common, "SessionLocal", _TaskDb)
    monkeypatch.setattr(common.tasks_api, "create_task", lambda db, **_: task)
    monkeypatch.setattr(common.tasks_api, "update_task", lambda db, t: t)
    creator = common.TaskCreator("u")
    _, _, future = creator.create_background_task(func, task_title="t")
    with pytest.raises(Exception):
        future.result(timeout=10)
    return task


def test_task_creator_reports_uncaught_exception(monkeypatch):
    def boom(on_error, on_progress_change):
        raise RuntimeError("boom")

    task = _run_task_creator(monkeypatch, boom)
    assert task.status == TaskStatus.failed
    assert task.message == "RuntimeError: boom"


def test_task_creator_keeps_on_error_text(monkeypatch):
    def partial(on_error, on_progress_change):
        on_error("no container")
        raise RuntimeError("later")

    task = _run_task_creator(monkeypatch, partial)
    assert task.status == TaskStatus.failed
    assert task.message == "no container"
