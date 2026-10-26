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

import json
from typing import List, Optional

from fastapi import (
    APIRouter,
    Request,
    Response,
    HTTPException,
    Depends,
    status,
)
from requests import HTTPError

from agri_gaia_backend.schemas.container_image import (
    ContainerImage,
    ContainerImageBuildConfig,
)
from agri_gaia_backend.schemas.keycloak_user import KeycloakUser
from sqlalchemy.orm import Session
from agri_gaia_backend.db import (
    container_api as sql_api,
    container_template_api as container_template_sql_api,
    model_api as model_sql_api,
    edge_device_api as edge_sql_api,
    container_deployment_api as container_deployment_sql_api,
)
from agri_gaia_backend.db import models
from agri_gaia_backend.db.database import SessionLocal
from agri_gaia_backend.services.docker import image_builder
from agri_gaia_backend.services.docker import docker_api
from agri_gaia_backend.routers.common import (
    TaskCreator,
    check_exists,
    get_db,
    get_task_creator,
)

import logging

logger = logging.getLogger("api-logger")

ROOT_PATH = "/container-images"
router = APIRouter(prefix=ROOT_PATH)


@router.get("", response_model=List[ContainerImage])
def get_all_container_images(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    return sql_api.get_container_images(db, skip, limit)


@router.get("/{repository_first}/{repository_second}:{image_tag}")
def get_container_image(
    repository_first: str,
    repository_second: str,
    image_tag: str,
    db: Session = Depends(get_db),
) -> List[ContainerImage]:
    repository_name = f"{repository_first}/{repository_second}"
    container_images = sql_api.get_container_images_by_repository_and_tag(
        db, repository=repository_name, tag=image_tag
    )
    if len(container_images) == 0:
        raise HTTPException(status_code=404)
    return container_images


@router.delete("/{repository_first}/{repository_second}:{image_tag}")
def delete_container_image(
    repository_first: str,
    repository_second: str,
    image_tag: str,
    db: Session = Depends(get_db),
):
    repository_name = f"{repository_first}/{repository_second}"
    container_images = sql_api.get_container_images_by_repository_and_tag(
        db, repository_name, image_tag
    )
    if len(container_images) == 0:
        raise HTTPError(status=404)

    # check if this image is currently deployed for any platform on any device
    # if yes, don't delete and return an error
    for container_image in container_images:
        deployments = (
            container_deployment_sql_api.get_container_deployments_for_container_image(
                db, container_image
            )
        )
        if len(deployments):
            msg = "The Container Image is currently deployed. Can not delete!"
            return Response(msg, status_code=409)

    # delete the images from the DB
    for container_image in container_images:
        sql_api.delete_container_image(db, container_image)

    return Response(status_code=204)


@router.post("/download/{repository}/{tag}/{arch}")
async def download_container_image(
    request: Request,
    repository: str,
    tag: str,
    arch: str,
    target_repository: Optional[str] = None,
    target_tag: Optional[str] = None,
    db: Session = Depends(get_db),
    task_creator: TaskCreator = Depends(get_task_creator),
):
    user: KeycloakUser = request.user
    owner = user.username

    repository = repository.replace("___", "/")
    os_arch = arch.replace("_", "/")

    external_image_name = repository.split("/")[-1]
    platform_repo_last = target_repository or external_image_name
    platform_repository = f"{owner}/{platform_repo_last}"
    platform_tag = target_tag or tag

    container_images = sql_api.get_container_images_by_repository_and_tag(
        db, platform_repository, platform_tag
    )

    if len(container_images) > 0:
        return Response(status_code=status.HTTP_409_CONFLICT)

    def download_task(on_progress_change, on_error):
        (
            repo_after_download,
            tag_after_download,
        ) = docker_api.download_image_into_platform_registry(
            repository, tag, os_arch, platform_repository, platform_tag
        )
        if repo_after_download is None or tag_after_download is None:
            error_msg = f"Image download for image '{repository}:{tag}' for architecture '{os_arch}' failed."
            on_error(error_msg)
            return

    _, task_location, _ = task_creator.create_background_task(
        download_task,
        task_title=f"Download Container Image {repository}/{tag}/{os_arch}",
    )

    headers = {"Location": task_location}
    return Response(status_code=status.HTTP_202_ACCEPTED, headers=headers)


@router.post("/build", response_model=ContainerImage)
def build_container(
    request: Request,
    config: ContainerImageBuildConfig,
    db: Session = Depends(get_db),
    task_creator: TaskCreator = Depends(get_task_creator),
) -> models.ContainerImage:
    user: KeycloakUser = request.user
    # check if user is allowed to push into repo with repository_name
    # for now a user is only allowed to push into the repository with the username as prefix
    if len(config.repository.split("/")) != 2:
        raise HTTPException(status_code=400, detail="invalid repository name")
    if config.repository.split("/")[0] != user.username:
        raise HTTPException(
            status_code=400,
            detail=f"user is not allowed to push into repository '{config.repository}'",
        )

    # Get Model Metadata
    model = check_exists(
        model_sql_api.get_model(db, config.model_id),
        detail=f"Model with id '{config.model_id}' does not exist",
    )
    if config.edge_device_id:
        edge_info = check_exists(
            edge_sql_api.get_edge_device(db, config.edge_device_id),
            detail=f"Edge Device with id '{config.edge_device_id}' does not exist",
        )
    else:
        edge_info = {"device_type": "generic", "architecture": config.architecture}

    container_template = check_exists(
        container_template_sql_api.get_container_template(
            db, config.container_template_id
        )
    )

    image_builder.validate_generator_input(model, edge_info)

    def build_and_push_image(
        on_error,
        on_progress_change,
        model_id,
        edge_device_id,
        architecture,
        container_template_id,
    ):
        # The task runs in another thread than the request. It therefore opens
        # its own session: the request session is closed as soon as the
        # response is sent.
        with SessionLocal() as task_db:
            task_model = check_exists(
                model_sql_api.get_model(task_db, model_id),
                detail=f"Model with id '{model_id}' does not exist",
            )
            if edge_device_id is not None:
                task_edge_info = check_exists(
                    edge_sql_api.get_edge_device(task_db, edge_device_id),
                    detail=f"Edge Device with id '{edge_device_id}' does not exist",
                )
            else:
                task_edge_info = {"device_type": "generic", "architecture": architecture}
            task_template = check_exists(
                container_template_sql_api.get_container_template(
                    task_db, container_template_id
                )
            )
            image_builder.build_and_push_image(
                inference_container_template=task_template,
                repository_name=config.repository,
                image_tag=config.tag,
                model=task_model,
                edge_info=task_edge_info,
                status_callback=lambda type, info: logger.debug(
                    f"{type}: {json.dumps(info)}"
                ),
            )

    _, task_location_url, _ = task_creator.create_background_task(
        build_and_push_image,
        task_title=f"Container Buildjob '{config.repository}/{config.tag}'",
        model_id=config.model_id,
        edge_device_id=config.edge_device_id,
        architecture=config.architecture,
        container_template_id=config.container_template_id,
    )

    headers = {"Location": task_location_url}
    return Response(status_code=status.HTTP_202_ACCEPTED, headers=headers)
