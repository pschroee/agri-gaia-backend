# SPDX-FileCopyrightText: 2026 Philipp Schröer
#
# SPDX-License-Identifier: MIT

"""Tests for #231: deploy/undeploy tasks run in their own session, a failed start
removes the container and reports the reason, and the containers of a device can be listed.

The tests need no database: sessions, SQL access and Portainer are replaced by fakes.
"""

import pytest
import requests
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.orm.exc import DetachedInstanceError

from agri_gaia_backend.db.models import ContainerDeploymentStatus
from agri_gaia_backend.routers import container_deployments as cd
from agri_gaia_backend.routers import edge_devices as ed
from agri_gaia_backend.services.portainer import portainer_api
from agri_gaia_backend.services.portainer.portainer_api import (
    DeploymentStartError,
    PortainerAPI,
)


class FakeSession:
    def __init__(self, name):
        self.name = name
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.closed = True


class FakeDeployment:
    """Behaves like an ORM object: port_bindings is lazy and needs an open session."""

    def __init__(self, session):
        self._session = session
        self.id = 9
        self.name = "agw-x"
        self.container_image_id = 1
        self.edge_device_id = 2
        self.status = ContainerDeploymentStatus.created
        self.docker_container_id = None

    @property
    def port_bindings(self):
        if self._session.closed:
            raise DetachedInstanceError("session is closed")
        return []


class FakeTaskCreator:
    def create_background_task(self, func, task_title, **kwargs):
        self.func = func
        self.kwargs = kwargs
        return None, "/tasks/1", None

    def run(self):
        errors = []
        self.func(on_error=errors.append, **self.kwargs)
        return errors


@pytest.fixture
def env(monkeypatch):
    request_db = FakeSession("request")
    task_dbs = []
    updated = []

    def session_local():
        s = FakeSession("task")
        task_dbs.append(s)
        return s

    def get_deployment(db, _id):
        return FakeDeployment(db)

    monkeypatch.setattr(cd, "SessionLocal", session_local)
    monkeypatch.setattr(cd.sql_api, "get_container_deployment", get_deployment)
    monkeypatch.setattr(cd.sql_api, "update_container_deployment", lambda db, d: updated.append(d) or d)
    monkeypatch.setattr(cd.sql_container_api, "get_container_image", lambda db, i: object())
    monkeypatch.setattr(cd.sql_edge_device_api, "get_edge_device", lambda db, i: object())
    return request_db, task_dbs, updated


def _deploy(request_db):
    creator = FakeTaskCreator()
    cd.deploy_container(9, db=request_db, task_creator=creator)
    return creator


def test_deploy_task_survives_closed_request_session(env, monkeypatch):
    request_db, task_dbs, updated = env

    def fake_deploy(edge_device, image, deployment):
        request_db.close()  # the response was sent, the request session is closed
        assert deployment.port_bindings == []  # would raise DetachedInstanceError
        return "cid"

    monkeypatch.setattr(cd.portainer, "deploy_container_to_edge_device", fake_deploy)
    errors = _deploy(request_db).run()
    assert errors == []
    assert updated[-1].status == ContainerDeploymentStatus.deployed
    assert updated[-1].docker_container_id == "cid"
    assert task_dbs and all(s.closed for s in task_dbs)


def test_deploy_task_reports_start_error(env, monkeypatch):
    request_db, _, updated = env

    def fake_deploy(*args):
        raise DeploymentStartError("Conflict. port is already allocated; the container was removed from the device again")

    monkeypatch.setattr(cd.portainer, "deploy_container_to_edge_device", fake_deploy)
    errors = _deploy(request_db).run()
    assert errors == ["Conflict. port is already allocated; the container was removed from the device again"]
    assert updated == []  # status stays "created", nothing stored


def test_deploy_task_keeps_left_container_id(env, monkeypatch):
    request_db, _, updated = env

    def fake_deploy(*args):
        raise DeploymentStartError("boom; stays on the device", container_id="abc")

    monkeypatch.setattr(cd.portainer, "deploy_container_to_edge_device", fake_deploy)
    errors = _deploy(request_db).run()
    assert errors == ["boom; stays on the device"]
    assert updated[-1].docker_container_id == "abc"
    assert updated[-1].status == ContainerDeploymentStatus.created


def test_deploy_task_reports_http_exception(env, monkeypatch):
    request_db, _, _ = env

    def fake_deploy(*args):
        raise HTTPException(400, "Conflict. The container name is already in use")

    monkeypatch.setattr(cd.portainer, "deploy_container_to_edge_device", fake_deploy)
    assert _deploy(request_db).run() == ["Conflict. The container name is already in use"]


def test_undeploy_task_uses_own_session(env, monkeypatch):
    request_db, task_dbs, updated = env

    def fake_undeploy(edge_device, deployment):
        request_db.close()
        assert deployment.port_bindings == []

    monkeypatch.setattr(cd.portainer, "undeploy_container_from_edge_device", fake_undeploy)
    creator = FakeTaskCreator()
    cd.undeploy_container(9, db=request_db, task_creator=creator)
    assert creator.run() == []
    assert updated[-1].status == ContainerDeploymentStatus.undeployed
    assert all(s.closed for s in task_dbs)


# --- PortainerAPI ---------------------------------------------------------------


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self.ok = status < 400
        self._body = body if body is not None else {}
        self.text = str(self._body)
        self.content = b"x" if body is not None else b""

    def json(self):
        return self._body

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(self.status_code)


@pytest.fixture
def api(monkeypatch):
    a = PortainerAPI.__new__(PortainerAPI)
    a.team_id = 1
    monkeypatch.setattr(a, "_login_if_needed", lambda: None)
    monkeypatch.setattr(a, "_get_auth_header", lambda: {"Authorization": "x"})
    return a


class Device:
    portainer_id = 5


class Image:
    repository = "r"
    tag = "t"


class Deployment:
    name = "agw-x"
    port_bindings = []


def _mock_requests(monkeypatch, start, delete):
    calls = []
    create = {
        "Id": "c" * 64,
        "Portainer": {"ResourceControl": {"Id": 1, "UserAccesses": [], "TeamAccesses": []}},
    }

    def post(url, **kw):
        calls.append(("POST", url, kw))
        if url.endswith("/images/create"):
            return Resp(200, {})
        if url.endswith("/containers/create"):
            return Resp(201, create)
        return start

    def put(url, **kw):
        calls.append(("PUT", url, kw))
        return Resp(200, {})

    def delete_(url, **kw):
        calls.append(("DELETE", url, kw))
        return delete

    monkeypatch.setattr(portainer_api.requests, "post", post)
    monkeypatch.setattr(portainer_api.requests, "put", put)
    monkeypatch.setattr(portainer_api.requests, "delete", delete_)
    return calls


def test_start_failure_removes_container(api, monkeypatch):
    calls = _mock_requests(monkeypatch, Resp(500, {"message": "port is already allocated"}), Resp(204, None))
    with pytest.raises(DeploymentStartError) as e:
        api.deploy_container_to_edge_device(Device(), Image(), Deployment())
    deletes = [c for c in calls if c[0] == "DELETE"]
    assert len(deletes) == 1
    assert deletes[0][2]["params"]["force"] is True
    assert str(e.value.message).endswith("removed from the device again")
    assert "port is already allocated" in e.value.message
    assert e.value.container_id is None


def test_start_failure_cleanup_fails(api, monkeypatch):
    _mock_requests(monkeypatch, Resp(500, {"message": "port is already allocated"}), Resp(500, {"message": "nope"}))
    with pytest.raises(DeploymentStartError) as e:
        api.deploy_container_to_edge_device(Device(), Image(), Deployment())
    assert e.value.container_id == "c" * 64
    assert "stays on the device" in e.value.message


def test_create_failure_raises_http_exception_without_cleanup(api, monkeypatch):
    calls = []

    def post(url, **kw):
        calls.append(url)
        if url.endswith("/containers/create"):
            return Resp(409, {"message": "Conflict. The container name is already in use"})
        return Resp(200, {})

    monkeypatch.setattr(portainer_api.requests, "post", post)
    with pytest.raises(HTTPException) as e:
        api.deploy_container_to_edge_device(Device(), Image(), Deployment())
    assert e.value.status_code == 400


def test_list_containers_maps_names_and_ports(api, monkeypatch):
    body = [
        {
            "Names": ["/agw-x"],
            "State": "running",
            "Ports": [
                {"IP": "0.0.0.0", "PrivatePort": 8000, "PublicPort": 18101, "Type": "tcp"},
                {"IP": "::", "PrivatePort": 8000, "PublicPort": 18101, "Type": "tcp"},
            ],
        },
        {"Names": ["/other"], "State": "exited", "Ports": [{"PrivatePort": 80, "Type": "tcp"}]},
    ]
    monkeypatch.setattr(portainer_api.requests, "get", lambda url, **kw: Resp(200, body))
    result = api.list_containers(5)
    assert result == [
        {"name": "agw-x", "state": "running", "host_ports": [18101]},
        {"name": "other", "state": "exited", "host_ports": []},
    ]


# --- route ----------------------------------------------------------------------


class Dev:
    def __init__(self, registered):
        self.registered = registered
        self.portainer_id = 5


def _call_route(monkeypatch, device, list_containers):
    monkeypatch.setattr(ed.sql_api, "get_edge_device", lambda db, i: device)
    monkeypatch.setattr(ed.portainer, "list_containers", list_containers)
    return ed.get_containers_on_edge_device(2, db=None)


def test_containers_route_unregistered_409(monkeypatch):
    with pytest.raises(HTTPException) as e:
        _call_route(monkeypatch, Dev(False), lambda pid: [])
    assert e.value.status_code == 409


def test_containers_route_portainer_down_502(monkeypatch):
    def down(pid):
        raise requests.ConnectionError("refused")

    with pytest.raises(HTTPException) as e:
        _call_route(monkeypatch, Dev(True), down)
    assert e.value.status_code == 502
    assert e.value.detail.startswith("Portainer: ")


def test_containers_route_unknown_device_404(monkeypatch):
    with pytest.raises(HTTPException) as e:
        _call_route(monkeypatch, None, lambda pid: [])
    assert e.value.status_code == 404
