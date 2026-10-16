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

from typing import List
import datetime
import requests
from fastapi import APIRouter, HTTPException, Depends, Response, status

from sqlalchemy.orm import Session

from agri_gaia_backend.db.models import ContainerDeploymentStatus

from agri_gaia_backend.schemas.container_image import ContainerImage
from agri_gaia_backend.schemas.edge_device import EdgeDevice

from agri_gaia_backend.schemas.container_deployment import (
    ContainerDeploymentCreate,
    ContainerDeployment,
)

from agri_gaia_backend.routers.common import (
    TaskCreator,
    check_exists,
    get_db,
    get_task_creator,
)
from agri_gaia_backend.db import container_deployment_api as sql_api
from agri_gaia_backend.db import container_api as sql_container_api
from agri_gaia_backend.db import edge_device_api as sql_edge_device_api

from agri_gaia_backend.services.portainer.portainer_api import portainer


import logging

logger = logging.getLogger("api-logger")


ROOT_PATH = "/container-deployments"
router = APIRouter(prefix=ROOT_PATH)


@router.get("", response_model=List[ContainerDeployment])
def get_all_container_deployments(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    return sql_api.get_container_deployments(db, skip=skip, limit=limit)


@router.get("/{container_deployment_id}", response_model=ContainerDeployment)
def get_container_deployment(
    container_deployment_id: int, db: Session = Depends(get_db)
):
    return check_exists(sql_api.get_container_deployment(db, container_deployment_id))


@router.post(
    "", response_model=ContainerDeployment, status_code=status.HTTP_201_CREATED
)
def create_container_deployment(
    container_deployment_create: ContainerDeploymentCreate,
    db: Session = Depends(get_db),
):
    edge_device: EdgeDevice = check_exists(
        sql_edge_device_api.get_edge_device(
            db, container_deployment_create.edge_device_id
        )
    )

    check_exists(
        sql_container_api.get_container_image(
            db, container_deployment_create.container_image_id
        )
    )

    if not edge_device.registered:
        raise HTTPException(status_code=409, detail="Edge Device is not registered!")

    if portainer.check_if_deployment_exists(
        edge_device.portainer_id, container_deployment_create.name
    ):
        raise HTTPException(
            status_code=409, detail="A Container with this name already exists!"
        )

    return sql_api.create_container_deployment(
        db, container_deployment_create, creation_date=datetime.datetime.now()
    )


@router.put("/{container_deployment_id}/deploy")
def deploy_container(
    container_deployment_id: int,
    db: Session = Depends(get_db),
    task_creator: TaskCreator = Depends(get_task_creator),
):
    container_deployment: ContainerDeployment = check_exists(
        sql_api.get_container_deployment(db, container_deployment_id)
    )

    check_exists(
        sql_container_api.get_container_image(
            db, container_deployment.container_image_id
        )
    )

    check_exists(
        sql_edge_device_api.get_edge_device(db, container_deployment.edge_device_id)
    )

    def deploy_task(container_deployment_id: int, on_error, **kwargs):
        container_deployment: ContainerDeployment = check_exists(
            sql_api.get_container_deployment(db, container_deployment_id)
        )

        container_image: ContainerImage = check_exists(
            sql_container_api.get_container_image(
                db, container_deployment.container_image_id
            )
        )

        edge_device: EdgeDevice = check_exists(
            sql_edge_device_api.get_edge_device(db, container_deployment.edge_device_id)
        )
        try:
            container_id = portainer.deploy_container_to_edge_device(
                edge_device, container_image, container_deployment
            )

            logger.debug("Container ID:")
            logger.debug(container_id)

            # update the deployment status in db
            container_deployment.status = ContainerDeploymentStatus.deployed
            container_deployment.docker_container_id = container_id
            container_deployment = sql_api.update_container_deployment(
                db, container_deployment
            )
        except requests.HTTPError as e:
            logger.error("Portainer Deploy Failed!")
            on_error(str(e))

    _, task_location_url, _ = task_creator.create_background_task(
        deploy_task,
        task_title=f"Container Deployment '{container_deployment.name}'",
        container_deployment_id=container_deployment_id,
    )

    headers = {"Location": task_location_url}
    return Response(status_code=status.HTTP_202_ACCEPTED, headers=headers)


@router.put("/{container_deployment_id}/undeploy")
def undeploy_container(
    container_deployment_id: int,
    db: Session = Depends(get_db),
    task_creator: TaskCreator = Depends(get_task_creator),
):
    container_deployment: ContainerDeployment = check_exists(
        sql_api.get_container_deployment(db, container_deployment_id)
    )

    check_exists(
        sql_edge_device_api.get_edge_device(db, container_deployment.edge_device_id)
    )

    def undeploy_task(container_deployment_id: int, on_error, **kwargs):
        container_deployment: ContainerDeployment = check_exists(
            sql_api.get_container_deployment(db, container_deployment_id)
        )

        edge_device: EdgeDevice = check_exists(
            sql_edge_device_api.get_edge_device(db, container_deployment.edge_device_id)
        )
        try:
            portainer.undeploy_container_from_edge_device(
                edge_device, container_deployment
            )
        except HTTPException as e:
            # keep the deployment, but tell the user why it could not be removed
            on_error(str(e.detail))
            raise

        # update the deployment status in db
        container_deployment.status = ContainerDeploymentStatus.undeployed
        container_deployment.docker_container_id = None
        container_deployment = sql_api.update_container_deployment(
            db, container_deployment
        )

    _, task_location_url, _ = task_creator.create_background_task(
        undeploy_task,
        task_title=f"Container Undeployment '{container_deployment.name}'",
        container_deployment_id=container_deployment_id,
    )

    headers = {"Location": task_location_url}
    return Response(status_code=status.HTTP_202_ACCEPTED, headers=headers)


@router.delete("/{container_deployment_id}")
def delete_container_deployment(
    container_deployment_id: int,
    db: Session = Depends(get_db),
):
    container_deployment: ContainerDeployment = check_exists(
        sql_api.get_container_deployment(db, container_deployment_id)
    )

    if container_deployment.status == ContainerDeploymentStatus.deployed:
        return Response(
            "Can not delete a currently deployed Container Deployment. Undeploy first!",
            409,
        )

    sql_api.delete_container_deployment(db, container_deployment)
    return Response(status_code=204)
