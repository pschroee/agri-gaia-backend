#!/usr/bin/env python

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

# -*- coding: utf-8 -*-

import io
import logging
import os
import tempfile
import time
import zipfile
import xmltodict
from base64 import b64encode
from typing import Dict, Optional, Union

import docker
from agri_gaia_backend.schemas.dataset import Dataset
from agri_gaia_backend.schemas.keycloak_user import KeycloakUser
from agri_gaia_backend.services import minio_api
from agri_gaia_backend.services.cvat.cvat_client import CvatClient
from agri_gaia_backend.util.common import distinct_colors, exists_in_dict
from agri_gaia_backend.util.env import bool_from_env
from fastapi import HTTPException
from requests.exceptions import HTTPError
from pathlib import Path


logger = logging.getLogger("api-logger")

PROJECT_BASE_URL = os.environ.get("PROJECT_BASE_URL")
VERIFY_SSL = bool_from_env("BACKEND_VERIFY_SSL")

CVAT_ANNOTATIONS_FILENAME = "annotations.xml"
CVAT_ANNOTATIONS_EXPORT_FORMAT = "CVAT for images 1.1"
CVAT_ANNOTATIONS_IMPORT_FORMAT = "CVAT 1.1"

cvatClient = CvatClient(
    protocol="https", host=f"cvat.{PROJECT_BASE_URL}", port=None, verify_ssl=VERIFY_SSL
)


# TODO: Move this to docker_api (for some reason, listing all containers does not work using docker_api)
def get_docker_container_fuzzy(fuzzy_name: str):
    client = docker.from_env()
    containers = list(filter(lambda c: fuzzy_name in c.name, client.containers.list()))
    if len(containers) == 1:
        return containers[0]


def rest_login(username: str, password: str) -> Dict:
    return dict(cvatClient.login(username, password))


def rest_logout(auth: Dict) -> Dict:
    return dict(cvatClient.logout(auth))


def rest_user_exists(username: str) -> Dict:
    users = cvatClient.get_users()["results"]
    return {
        "username": username,
        "exists": any([user["username"] == username for user in users]),
    }


def rest_user_create(
    username: str, email: str, password: str, first_name: str, last_name: str
) -> Dict:
    cvat_server = get_docker_container_fuzzy("cvat_server")
    if cvat_server is not None:
        cvat_server.exec_run(
            f"/bin/bash -c '/create-user.sh {username} {email} {password} {first_name} {last_name}'"
        )
        return {
            "username": username,
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
        }
    else:
        raise HTTPException(status_code=404, detail="CVAT server container not found.")


def update_sample_paths(annotations: str, user: KeycloakUser, dataset: Dataset) -> str:
    xml: Dict = xmltodict.parse(annotations)

    if not exists_in_dict(xml, ["annotations", "image"]):
        return None

    images = xml["annotations"]["image"]
    if type(images) is not list:
        images = [images]

    for image in images:
        # we have to update the name attribute of each image tag in the annotation file
        # so that the dataset id fits the newly uploaded dataset
        # also, if this Dataset was labeled by an external CVAT instance
        # there are no information about user / dataset at all, so we add them here.
        parts = list(Path(image["@name"]).parts)
        part_count = len(parts)

        if part_count == 1:
            # there is no bucket information, create it
            filename = parts[0]
            parts = [user.username, str(dataset.id), filename]
        elif part_count >= 2:
            # there are information, so we override them
            parts[0] = user.username
            parts[1] = str(dataset.id)
        else:
            raise Exception(
                "Invalid Annotation format: Please check image name attribute"
            )

        image["@name"] = str(Path(*parts))
    return xmltodict.unparse(xml, pretty=True)


def _task_exists(auth_data: Dict, task_id: int) -> bool:
    try:
        cvatClient.get_task(task_id, auth_data)
    except HTTPError as e:
        if e.response.status_code == 404:
            return False
    return True


def create_task_for_dataset(
    auth_data: Dict,
    dataset: Dataset,
    user: KeycloakUser,
    segmentSize: int = 25,
    imageQuality: int = 70,
) -> Dict:
    task_id = dataset.annotation_task_id

    if task_id is None or not _task_exists(auth_data, task_id):
        if dataset.annotation_labels:
            label_colors = distinct_colors(len(dataset.annotation_labels))
            labels = [
                {
                    "name": label_name,
                    "attributes": [],
                    "color": label_colors[i],
                    "type": "any",
                }
                for i, label_name in enumerate(dataset.annotation_labels)
            ]
        else:
            labels = [{"name": "object", "attributes": [], "type": "any"}]

        try:
            task = cvatClient.create_task(
                task={
                    "project_id": None,
                    "name": dataset.name,
                    "labels": labels,
                    "segment_size": str(segmentSize),
                    "source_storage": {"location": "local"},
                    "target_storage": {"location": "local"},
                },
                auth_data=auth_data,
            )
            task_id = task["id"]

            try:
                cvatClient.add_data_to_task(
                    task_id=task_id,
                    data={
                        "server_files": [f"/{dataset.bucket_name}/{dataset.id}/"],
                        "image_quality": imageQuality,
                        "use_zip_chunks": True,
                        "use_cache": True,
                        "sorting_method": "lexicographical",
                    },
                    auth_data=auth_data,
                )

                _wait_for_task(auth_data, task_id)

                if _annotation_file_exists(dataset, user):
                    annotations = _get_dataset_annotations_from_file(
                        dataset, user, update_paths=True
                    )
                    _add_annotations_from_file_to_task(auth_data, task_id, annotations)
            except Exception as e:
                cvatClient.delete_task(task_id=task_id, auth_data=auth_data)
                raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"id": task_id, "next": f"/tasks/{task_id}"}


def remove_task(
    task_id: int, auth: Optional[Dict] = None, owner: Optional[str] = None
) -> None:
    """Delete a CVAT task.

    The platform deletes with the CVAT superuser when it is configured, so callers
    without a CVAT session (for example the agent gateway, which sends placeholders)
    can clean up as well. The caller's session is only a fallback.

    If ``owner`` is given, the task is only deleted when the CVAT owner matches. That
    protects the cleanup path that runs after the database entry is already gone and
    therefore has no dataset owner left to check against.

    A task that is already gone (HTTP 404) counts as deleted.
    """
    superuser_configured = bool(
        os.getenv("CVAT_SUPERUSER") and os.getenv("CVAT_SUPERUSER_PASSWORD")
    )
    if superuser_configured:
        auth_data = dict(cvatClient.login_superuser())
    elif auth is not None:
        auth_data = auth
    else:
        raise ValueError("No CVAT credentials available for task deletion.")

    try:
        if owner is not None:
            try:
                task = cvatClient.get_task(task_id=task_id, auth_data=auth_data)
            except HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    logger.info(f"CVAT task {task_id} is already gone.")
                    return
                raise
            task_owner = (task.get("owner") or {}).get("username")
            if task_owner != owner:
                raise PermissionError(
                    f"CVAT task {task_id} belongs to '{task_owner}', not to '{owner}'."
                )

        try:
            cvatClient.delete_task(task_id=task_id, auth_data=auth_data)
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.info(f"CVAT task {task_id} is already gone.")
                return
            raise
    finally:
        if superuser_configured:
            try:
                cvatClient.logout(auth_data)
            except Exception as e:
                logger.warning(f"Logging out of CVAT failed: {e}")


def get_task_annotations(task_id: int) -> Union[str, None]:
    while True:
        response = cvatClient.get_task_annotations(
            task_id,
            filename=os.path.splitext(CVAT_ANNOTATIONS_FILENAME)[0] + ".zip",
            format=CVAT_ANNOTATIONS_EXPORT_FORMAT,
        )

        if response.status_code == 202:
            time.sleep(1)
        elif response.status_code == 201:
            response = cvatClient.get_task_annotations(
                task_id,
                filename=os.path.splitext(CVAT_ANNOTATIONS_FILENAME)[0] + ".zip",
                format=CVAT_ANNOTATIONS_EXPORT_FORMAT,
                action="download",
            )
            with tempfile.TemporaryDirectory() as tempDir:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zipFile:
                    zipFile.extractall(tempDir)
                with open(os.path.join(tempDir, CVAT_ANNOTATIONS_FILENAME), "r") as fh:
                    return fh.read()
        else:
            break


def _wait_for_task(auth: Dict, task_id: int) -> None:
    while True:
        response = cvatClient.get_task_status(task_id, auth_data=auth)
        logging.info(f"CVAT task status: {response}")
        if response["state"] == "Finished":
            break
        elif response["state"] == "Failed":
            raise HTTPException(status_code=500, detail=response["message"])
        else:
            time.sleep(1)


def _add_annotations_from_file_to_task(
    auth: Dict, task_id: int, annotations: str
) -> None:
    response = cvatClient.add_task_annotations(
        task_id,
        filename=CVAT_ANNOTATIONS_FILENAME,
        format=CVAT_ANNOTATIONS_IMPORT_FORMAT,
        auth_data=auth,
        extra_headers={"upload-start": "true"},
    )

    if response.status_code == 202:
        extra_headers = {
            "upload-length": str(len(bytes(annotations, "utf-8"))),
            "upload-metadata": f"filename {_b64encode(CVAT_ANNOTATIONS_FILENAME)},filetype {_b64encode('text/xml')}",
        }
        response = cvatClient.add_task_annotations(
            task_id, auth_data=auth, extra_headers=extra_headers
        )

        if response.status_code == 201:
            file_id = os.path.basename(response.headers.get("Location"))
            cvatClient.update_task_annotations_from_file(
                task_id,
                file_id,
                annotations,
                auth_data=auth,
                extra_headers={
                    "content-type": "application/offset+octet-stream",
                    "content-length": extra_headers["upload-length"],
                    "upload-offset": "0",
                },
            )

            response = cvatClient.add_task_annotations(
                task_id,
                filename=CVAT_ANNOTATIONS_FILENAME,
                format=CVAT_ANNOTATIONS_IMPORT_FORMAT,
                auth_data=auth,
                extra_headers={"upload-finish": "true"},
            )

            if response.status_code == 202:
                while True:
                    response = cvatClient.update_task_annotations(
                        task_id,
                        filename=CVAT_ANNOTATIONS_FILENAME,
                        format=CVAT_ANNOTATIONS_IMPORT_FORMAT,
                        auth_data=auth,
                    )
                    if response.status_code == 202:
                        time.sleep(1)
                    else:
                        break


def _annotation_file_exists(dataset: Dataset, user: KeycloakUser) -> bool:
    return minio_api.exists(
        bucket=user.username,
        object_name=_annotation_file_object_name(dataset),
        token=user.minio_token,
    )


def _get_dataset_annotations_from_file(
    dataset: Dataset, user: KeycloakUser, update_paths: bool = False
) -> str:
    annotations = minio_api.get_object(
        bucket=user.username,
        object_name=_annotation_file_object_name(dataset),
        token=user.minio_token,
    ).data.decode("utf-8")

    if update_paths:
        annotations = update_sample_paths(annotations, user, dataset)

    return annotations


def _annotation_file_object_name(dataset):
    return f"datasets/{dataset.id}/annotations/{CVAT_ANNOTATIONS_FILENAME}"


def _b64encode(string: str) -> str:
    return str(b64encode(string.encode("utf-8")), "utf-8")
