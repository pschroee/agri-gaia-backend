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

import logging
import datetime

from typing import List
import requests
from requests.exceptions import HTTPError
from fastapi import APIRouter, Depends, Response, status
from fastapi.exceptions import HTTPException
from sqlalchemy.orm import Session
from agri_gaia_backend.schemas.edge_device import EdgeContainer, EdgeDevice, EdgeDeviceCreate
from agri_gaia_backend.schemas.container_deployment import ContainerDeployment
from agri_gaia_backend.routers.common import check_exists, get_db
from agri_gaia_backend.db import (
    models,
    edge_device_api as sql_api,
)
from agri_gaia_backend.routers.common import check_exists
from agri_gaia_backend.services.portainer.portainer_api import portainer

logger = logging.getLogger("api-logger")

ROOT_PATH = "/edge-devices"
router = APIRouter(prefix=ROOT_PATH)


def _update_portainer_info(edge_devices: List[EdgeDevice], db: Session):
    # query the last heartbeat and add to each device, if available
    try:
        endpoints = portainer.get_all_endpoints()
    except:
        endpoints = dict()

    try:
        tags = portainer.get_tags()
    except:
        tags = []

    endpoint_keys = endpoints.keys()
    tag_dict = dict()
    for t in tags:
        tag_dict[t["ID"]] = t

    for e in edge_devices:
        e.tags = []

        if e.name in endpoint_keys:
            endpoint = endpoints[e.name]

            # set last heartbeat
            last_heartbeat = endpoint["LastCheckInDate"]
            if last_heartbeat != 0:
                e.last_heartbeat = datetime.datetime.utcfromtimestamp(last_heartbeat)
                sql_api.update_edge_device(db, e)

            # set tags
            for tag_id in endpoint["TagIds"]:
                e.tags.append(tag_dict[tag_id]["Name"])


@router.get("", response_model=List[EdgeDevice])
def get_all_edge_devices(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    edge_devices = sql_api.get_edge_devices(db, skip=skip, limit=limit)
    _update_portainer_info(edge_devices, db)

    return edge_devices


@router.get("/{edge_device_id}", response_model=EdgeDevice)
def get_edge_device(edge_device_id: int, db: Session = Depends(get_db)):
    device = check_exists(sql_api.get_edge_device(db, edge_device_id))
    _update_portainer_info([device], db)
    if not device.registered:
        _register_edge_device(device, db)

    return device


@router.get("/{edge_device_id}/deployments", response_model=List[ContainerDeployment])
def get_container_deployments_for_edge_device(
    edge_device_id: int, db: Session = Depends(get_db)
) -> List[ContainerDeployment]:
    edge_device: models.EdgeDevice = check_exists(
        sql_api.get_edge_device(db, edge_device_id)
    )
    return edge_device.container_deployments


@router.get("/{edge_device_id}/containers", response_model=List[EdgeContainer])
def get_containers_on_edge_device(
    edge_device_id: int, db: Session = Depends(get_db)
) -> List[EdgeContainer]:
    edge_device: models.EdgeDevice = check_exists(
        sql_api.get_edge_device(db, edge_device_id)
    )
    if not edge_device.registered:
        raise HTTPException(status_code=409, detail="Edge Device is not registered!")
    try:
        return portainer.list_containers(edge_device.portainer_id)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Portainer: {e}")


@router.post("", response_model=EdgeDevice, status_code=status.HTTP_201_CREATED)
def create_edge_device(
    edge_device_create: EdgeDeviceCreate, db: Session = Depends(get_db)
):
    tag_ids = portainer.get_ids_for_tag_names(
        edge_device_create.tags, allow_create=True
    )

    try:
        endpoint = portainer.create_new_endpoint(edge_device_create.name, tag_ids)
    except HTTPError as e:
        status_code = e.response.status_code
        if status_code == 409:
            detail = f"Endpoint with name '{edge_device_create.name}' already exists."
            logging.error(detail)
            raise HTTPException(409, detail)
        else:
            raise HTTPException(status_code, str(e))

    return sql_api.create_edge_device(
        db,
        edge_device_create.name,
        edge_key=endpoint["EdgeKey"],
        portainer_id=endpoint["Id"],
    )


@router.delete("/{edge_device_id}")
def delete_edge_device(edge_device_id: int, db: Session = Depends(get_db)):

    edge_device: models.EdgeDevice = check_exists(
        sql_api.get_edge_device(db, edge_device_id)
    )

    portainer.delete_endpoint(edge_device.portainer_id)

    sql_api.delete_edge_device(db, edge_device)
    return Response(status_code=204)


# called for every edge device queried for the first time after assosication
# TODO: do this somewhere in the background and not during a request
def _register_edge_device(edge_device: EdgeDevice, db: Session):

    if not edge_device.last_heartbeat:
        return
    try:
        endpoint_info = portainer.get_endpoint_docker_info(edge_device.portainer_id)

        edge_device.os = endpoint_info["OperatingSystem"]
        edge_device.cpu_count = endpoint_info["NCPU"]
        edge_device.arch = (
            str(endpoint_info["Architecture"])
            .replace("x86_64", "amd64")
            .replace("aarch64", "arm64")
        )
        edge_device.memory = int(endpoint_info["MemTotal"] / 1000000)

        edge_device.registered = True

        sql_api.update_edge_device(db, edge_device)
    except HTTPError as e:
        logger.warning(
            f"Failed to access edge device '{edge_device.name}' at portainer endpoint id '{edge_device.portainer_id}' using websocket tunnel on edge agent port."
        )
