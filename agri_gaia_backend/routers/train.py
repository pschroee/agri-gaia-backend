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
import os
import re
import glob
import json
import types
import logging
import tempfile
import zipfile
from datetime import datetime
from distutils.dir_util import copy_tree
from operator import itemgetter
from shutil import copy
from typing import Dict, List, Optional
from itertools import chain

from agri_gaia_backend.db import dataset_api as dataset_sql_api
from agri_gaia_backend.db.database import SessionLocal
from agri_gaia_backend.db import train_api as sql_api
from agri_gaia_backend.db.models import Dataset
from agri_gaia_backend.schemas.model import Model
from agri_gaia_backend.routers.common import (
    TaskCreator,
    check_exists,
    create_single_file_response,
    get_db,
    get_task_creator,
    task_error_message,
)
from agri_gaia_backend.util.common import gpu_available, get_torch_cuda_index
from agri_gaia_backend.util.train_metrics import (
    metrics_supported,
    parse_gpu_csv,
    parse_metrics_jsonl,
)
from agri_gaia_backend.routers.datasets import update_annotations_from_cvat
from agri_gaia_backend.routers.models import persist_model, persist_model_artifact
from agri_gaia_backend.schemas.keycloak_user import KeycloakUser
from agri_gaia_backend.schemas.train_container import TrainContainer
from agri_gaia_backend.services.docker import image_builder
from agri_gaia_backend.services.docker.client import host_client as docker_host_client
from agri_gaia_backend.util.train import (
    container_exists,
    get_config_filepath,
    get_config_filepaths,
    get_container,
    get_container_status,
    pull_image_if_missing,
    get_dockerfile_filepath,
    jsonfile2dict,
    remove_container,
    remove_image,
    is_float,
    infer_model_format,
    validate_template,
    install_template,
    uninstall_template,
    get_onnx_model_filepath,
    read_from_container,
    read_class_order,
    read_train_container_files,
    write_train_container_files,
    get_config_with_schema,
    get_export_schema_filepath,
    update_train_container_config,
    update_train_container_dataset,
    read_train_container_config,
    merge_train_config,
    get_config_container_filepath,
    TRAIN_CONFIG_FILENAME,
)
from docker.types import DeviceRequest, Ulimit
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status, Form
from fastapi.datastructures import UploadFile
from fastapi.param_functions import File
from sqlalchemy.orm import Session
from jsonschema import validate

ROOT_PATH = "/train"

MODEL_TRAINING_PATH = os.path.abspath("./model-training")
TEMPLATES_PATH = os.path.join(MODEL_TRAINING_PATH, "templates")
EXPORT_PATH = os.path.join(MODEL_TRAINING_PATH, "export")
COMMON_PATH = os.path.join(MODEL_TRAINING_PATH, "common")
DATASETS_PATH = os.path.join(MODEL_TRAINING_PATH, "datasets")

VERIFY_SSL = False

logger = logging.getLogger("api-logger")
router = APIRouter(prefix=ROOT_PATH)


def train_container_created_message(train_container_id: int) -> str:
    return f"Train Container #{train_container_id} created."


FALLBACK_SCORE_REGEXP = re.compile(
    r"(?:Multiclass|Binary)?Accuracy\W+([0-9]*\.?[0-9]+)"
)

GPU_NOT_VISIBLE_MESSAGE = (
    "GPU configured, but the backend cannot see it (NVML); "
    "the training container still requests it."
)


# The NVIDIA device files a GPU container gets explicitly besides its device request (#247): after a
# `systemctl daemon-reload` a container with only `--gpus` loses the GPU (NVML "Unknown Error", systemd cgroup
# driver); with the files given as devices it keeps it (shown on the instance, 27.10.2026). GPU_DEVICE_FILES
# overrides the list (comma separated); only files that exist are passed.
DEFAULT_GPU_DEVICE_FILES = (
    "/dev/nvidia0",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-modeset",
)


def gpu_device_files() -> List[str]:
    """The device mappings ("path:path:rwm") of the NVIDIA device files that exist."""
    raw = os.getenv("GPU_DEVICE_FILES", "")
    paths = [p.strip() for p in raw.split(",") if p.strip()] if raw else list(DEFAULT_GPU_DEVICE_FILES)
    return [f"{p}:{p}:rwm" for p in paths if os.path.exists(p)]


def gpu_device_request_for_run(on_message=None) -> Optional[DeviceRequest]:
    """Return the GPU request for a training container, or None for the CPU.

    If GPUS is set, the GPU is always requested: the container is created at the
    host daemon, which hands out the GPU afresh, so the backend does not need to
    see it itself. After a systemd reload the backend loses its own NVML view
    (the /dev/char links vanish) while the host GPU is fine; asking nvidia-smi in
    the backend then used to put the training silently on the CPU. If the host
    really has no GPU, creating the container fails loudly. Without GPUS the
    backend's own view decides, as before.
    """
    gpu_device_ids = os.getenv("GPUS")
    if gpu_device_ids:
        if not gpu_available():
            logger.error(
                "GPUS is set, but the backend cannot see the GPU (NVML); "
                "the training container still requests it."
            )
            if on_message:
                on_message(GPU_NOT_VISIBLE_MESSAGE)
    elif not gpu_available():
        return None

    # See: https://github.com/docker/docker-py/blob/82cf559b5a641f53e9035b44b91f829f3b4cca80/docker/types/containers.py
    if (
        gpu_device_ids
        and gpu_device_ids != "all"
        and all(map(lambda x: x.isdigit(), gpu_device_ids.split(",")))
    ):
        return DeviceRequest(
            name="nvidia", device_ids=[gpu_device_ids], capabilities=[["gpu"]]
        )
    return DeviceRequest(name="nvidia", count=-1, capabilities=[["gpu"]])


def read_score_from_logs(train_container) -> Optional[float]:
    """Score of a finished training read from its container logs.

    Tries the regexp of the template first, then the Lightning fallback.
    Returns None if the logs hold no score; raises if a match is not a number.
    """
    logs = get_container(train_container.container_id).logs(tail=50).decode("utf-8")

    score_regexp = re.compile(train_container.score_regexp)
    score_matches = re.findall(score_regexp, logs)
    if not score_matches:
        # Lightning prints its test table as "MulticlassAccuracy  0.98"; the
        # regexp of the template does not always match it.
        score_matches = re.findall(FALLBACK_SCORE_REGEXP, logs)

    if not score_matches:
        return None

    if type(score_matches[0]) is not str:
        score_matches = chain.from_iterable(score_matches)

    return round(
        float(
            list(
                filter(
                    lambda s: is_float(s),
                    map(lambda s: s.strip(), score_matches),
                )
            )[-1]
        ),
        3,
    )


def model_name_for_training(train_container, dataset) -> str:
    """Name of the model made from a training; the score is left out if unknown."""
    detail = dataset.name
    if train_container.score is not None:
        detail += f", {train_container.score} {train_container.score_name}"
    return f"{train_container.architecture} ({train_container.provider}) [{detail}]"


def build_train_image(
    on_error,
    on_progress_change,
    provider: str,
    category: str,
    architecture: str,
    dataset_id: int,
    train_config: Dict,
    export_config: Optional[Dict],
    train_configs_path: str,
    owner: str,
    on_message=None,
) -> None:
    try:
        # The task runs in another thread than the request. It therefore opens
        # its own session: the request session is closed as soon as the
        # response is sent.
        with SessionLocal() as task_db:
            dataset = check_exists(
                dataset_sql_api.get_dataset(task_db, dataset_id),
                detail=f"Dataset with id '{dataset_id}' does not exist.",
            )

            with tempfile.TemporaryDirectory() as context_directory:
                copy_tree(COMMON_PATH, context_directory)
                copy_tree(DATASETS_PATH, context_directory)

                dockerfile_filepath = get_dockerfile_filepath(
                    TEMPLATES_PATH, provider, architecture
                )
                copy_tree(os.path.dirname(dockerfile_filepath), context_directory)
                for config_filename in ("custom.json", "presets.json"):
                    copy(
                        os.path.join(train_configs_path, config_filename), context_directory
                    )
                copy(os.path.join(EXPORT_PATH, "export.py"), context_directory)

                with open(
                    os.path.join(context_directory, "train_config.json"),
                    "w",
                ) as fh:
                    json.dump(train_config, fh)

                with open(
                    os.path.join(context_directory, "export_config.json"),
                    "w",
                ) as fh:
                    if not export_config:
                        export_config = {}
                    json.dump(export_config, fh)

                image_name = f"{provider}-{architecture.replace(' ', '_')}".lower()
                repository_url = (
                    image_builder.create_repository_url(owner) + f"/{image_name}"
                )
                image_tag = datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + "-train"

                torch_cuda_index = get_torch_cuda_index()
                build_kwargs = (
                    {"build_args": {"TORCH_CUDA_INDEX": torch_cuda_index}}
                    if torch_cuda_index
                    else {}
                )

                def on_phase(text: str, fraction: float) -> None:
                    if on_message:
                        on_message(text)
                    on_progress_change(fraction)

                image_builder._build_and_push_image(
                    context_directory=context_directory,
                    repository_url=repository_url,
                    image_tag=image_tag,
                    platforms=["linux/amd64"],
                    status_callback=lambda type, info: logger.info(
                        f"{type}: {json.dumps(info)}"
                    ),
                    on_phase=on_phase,
                    remove_after_push=True,
                    **build_kwargs,
                )

            if on_message:
                on_message(image_builder.PHASE_CREATING_CONTAINER)
            on_progress_change(0.95)

            image_id = f"{repository_url}:{image_tag}"

            custom_config_filepath = os.path.join(train_configs_path, "custom.json")
            with open(custom_config_filepath, "r") as fh:
                model_filepath, score_name, score_regexp = itemgetter(
                    "model_filepath", "score_name", "score_regexp"
                )(json.load(fh))

            train_container = sql_api.create_train_container(
                task_db,
                owner=owner,
                image_id=image_id,
                repository=f"{owner}/{image_name}",
                tag=image_tag,
                last_modified=datetime.now(),
                provider=provider,
                category=category,
                architecture=architecture,
                dataset_id=dataset_id,
                dataset=dataset,
                model_filepath=model_filepath,
                score_name=score_name,
                score_regexp=score_regexp,
            )
            if on_message:
                on_message(train_container_created_message(train_container.id))

            # Warm the image on the host while no user token is involved. The run
            # task only has the user's access token, which lives 300 s, and a cold
            # pull takes minutes; pulling here keeps the token valid until the
            # container starts. A failure must not fail the build: the run task
            # pulls as a fallback.
            try:
                pull_image_if_missing(image_id)
            except Exception as e:
                logger.error(f"Unable to pull train image '{image_id}' to the host: {e}")
    except Exception as e:
        logger.exception("Build of train image failed")
        on_error(task_error_message(e))


def _clone_config_files(container_id: str, train_container) -> Dict[str, bytes]:
    config_filepath = str(
        get_config_container_filepath(
            config_filename=TRAIN_CONFIG_FILENAME,
            templates_path=TEMPLATES_PATH,
            provider=train_container.provider,
            architecture=train_container.architecture,
        )
    )
    base_data = read_from_container(container_id, config_filepath)
    base = json.loads(base_data.decode("utf-8")) if base_data else {}
    return {config_filepath: merge_train_config(base, train_container.config_override)}


@router.post("/containers/{train_container_id}/run")
async def run_train_container(
    request: Request,
    train_container_id: int,
    db: Session = Depends(get_db),
    task_creator: TaskCreator = Depends(get_task_creator),
) -> Response:
    user: KeycloakUser = request.user

    train_container = check_exists(
        sql_api.get_train_container(db, train_container_id)
    )
    if (
        train_container.container_id is not None
        and container_exists(train_container.container_id)
        and get_container(train_container.container_id).status == "running"
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Train Container #{train_container_id} is already running.",
        )

    def _run_train_container(
        on_error,
        on_progress_change,
        user: KeycloakUser,
        train_container_id: int,
        on_message=None,
    ) -> None:
        try:
            # The task runs in another thread than the request. It therefore opens
            # its own session: the request session is closed as soon as the
            # response is sent.
            with SessionLocal() as task_db:
                train_container: TrainContainer = check_exists(
                    sql_api.get_train_container(task_db, train_container_id)
                )

                dataset: Dataset = check_exists(
                    dataset_sql_api.get_dataset(task_db, train_container.dataset_id)
                )

                if (
                    train_container.container_id is not None
                    and container_exists(train_container.container_id)
                    and get_container(train_container.container_id).status == "running"
                ):
                    raise HTTPException(
                        status_code=409,
                        detail=f"Train Container #{train_container_id} is already running.",
                    )

                # The access token is written into the container environment at
                # creation time and Docker cannot change the environment of an
                # existing container. Pulling a cold image can take longer than the
                # token lives (300 s), so pull first and only then create a fresh
                # container with the token from this request. A container left over
                # from an earlier run still carries the token of that run.
                pull_image_if_missing(train_container.image_id)

                # Configuration edits and trained models live only in the old
                # container's filesystem. Keep them across the recreation.
                preserved_files = {}
                first_run = train_container.container_id is None
                if train_container.container_id is not None:
                    preserved_files = read_train_container_files(
                        container_id=train_container.container_id,
                        templates_path=TEMPLATES_PATH,
                        provider=train_container.provider,
                        architecture=train_container.architecture,
                        extra_filepaths=(
                            train_container.model_filepath,
                            get_onnx_model_filepath(train_container.model_filepath),
                        ),
                    )
                    remove_container(train_container.container_id, force=True)
                    train_container.container_id = None
                    sql_api.update_train_container(task_db, train_container)

                run_parameters = {
                    "image": train_container.image_id,
                    "environment": {
                        "OID_ACCESS_TOKEN": user.access_token,
                        "S3_HOST": "http://minio:9000",
                        "S3_BUCKET_NAME": train_container.dataset.bucket_name,
                        "DATASET_ID": dataset.id,
                    },
                    "network": f"{os.environ.get('PROJECT_NAME')}_network",
                    "privileged": True,
                    "cap_add": ["SYS_ADMIN"],
                    "devices": ["/dev/fuse:/dev/fuse"],
                    "detach": True,
                    "ipc_mode": "host",
                    "ulimits": [
                        Ulimit(name="memlock", soft=-1, hard=-1),
                        Ulimit(name="stack", soft=67108864, hard=67108864),
                    ],
                }

                gpu_device_request = gpu_device_request_for_run(on_message)
                if gpu_device_request is not None:
                    run_parameters["device_requests"] = [gpu_device_request]
                    run_parameters["environment"]["NVIDIA_VISIBLE_DEVICES"] = "all"
                    device_files = gpu_device_files()
                    if device_files:
                        run_parameters["devices"] = device_files

                container = docker_host_client.containers.create(**run_parameters)
                if first_run and train_container.config_override:
                    # A clone has no container yet: its train_config is the one baked into
                    # the image, overridden by the values given at clone time.
                    preserved_files = {
                        **preserved_files,
                        **_clone_config_files(container.id, train_container),
                    }
                write_train_container_files(container.id, preserved_files)
                container.start()
                train_container.container_id = container.id

                train_container.score = None
                sql_api.update_train_container(task_db, train_container)

                if dataset.annotation_task_id is not None:
                    update_annotations_from_cvat(dataset, user.minio_token)
        except Exception as e:
            logger.exception("Run of train container #%s failed", train_container_id)
            on_error(task_error_message(e))

    _, task_location_url, _ = task_creator.create_background_task(
        _run_train_container,
        task_title=f"Run Train Container #{train_container_id}.",
        user=user,
        train_container_id=train_container_id,
    )

    headers = {"Location": task_location_url}
    return Response(status_code=status.HTTP_202_ACCEPTED, headers=headers)


@router.patch("/containers/{train_container_id}/score")
def update_train_container_score(
    train_container_id: int, db: Session = Depends(get_db)
):
    train_container = check_exists(sql_api.get_train_container(db, train_container_id))
    train_container_status = get_container_status(train_container.container_id)
    if train_container_status != "exited":
        raise HTTPException(
            status_code=400,
            detail=f"Train Container #{train_container_id} has not exited.",
        )

    if train_container.score is not None:
        return train_container

    try:
        score = read_score_from_logs(train_container)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to extract score for Train Container #{train_container_id}: {str(e)}",
        )

    if score is None:
        raise HTTPException(
            status_code=404,
            detail=f"No score for Train Container #{train_container_id} found in logs.",
        )

    train_container.score = score
    sql_api.update_train_container(db, train_container)
    return train_container


@router.post("/containers/{train_container_id}/stop")
def stop_train_container(
    train_container_id: int,
    task_creator: TaskCreator = Depends(get_task_creator),
) -> Response:
    def _stop_train_container(
        on_error,
        on_progress_change,
        train_container_id: int,
    ) -> None:
        try:
            # The task runs in another thread than the request. It therefore opens
            # its own session: the request session is closed as soon as the
            # response is sent.
            with SessionLocal() as task_db:
                train_container: TrainContainer = check_exists(
                    sql_api.get_train_container(task_db, train_container_id)
                )
                get_container(train_container.container_id).stop()
        except Exception as e:
            logger.exception("Stop of train container #%s failed", train_container_id)
            on_error(task_error_message(e))

    _, task_location_url, _ = task_creator.create_background_task(
        _stop_train_container,
        task_title=f"Stop Train Container #{train_container_id}.",
        train_container_id=train_container_id,
    )

    headers = {"Location": task_location_url}
    return Response(status_code=status.HTTP_202_ACCEPTED, headers=headers)


@router.get("/containers", response_model=List[TrainContainer])
def get_train_containers(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    train_containers = sql_api.get_train_containers(db, skip=skip, limit=limit)
    for train_container in train_containers[:]:
        try:
            train_container.status = get_container_status(train_container.container_id)
            sql_api.update_train_container(db, train_container)
        # if get_container_status raises a "not found" exception, the container was removed from the host
        # consequently, it needs to be deleted from the database because it cannot be started again.
        except HTTPException as e: 
            sql_api.delete_train_container(db, train_container)
            train_containers.remove(train_container)

    return train_containers


@router.get("/containers/{train_container_id}")
def get_train_container(train_container_id: int, db: Session = Depends(get_db)) -> Dict:
    return check_exists(sql_api.get_train_container(db, train_container_id))


@router.get("/containers/{train_container_id}/metrics")
def get_train_container_metrics(
    train_container_id: int, db: Session = Depends(get_db)
) -> Dict:
    """Epoch figures (metrics.jsonl) and GPU samples (gpu.csv) of a training container.

    Readable while the container runs and after it exited. Containers built from images
    without the label agrigaia.metrics answer with supported=false and no rows.
    """
    train_container = check_exists(sql_api.get_train_container(db, train_container_id))

    container = None
    labels = None
    if train_container.container_id is not None and container_exists(
        train_container.container_id
    ):
        container = get_container(train_container.container_id)
        try:
            labels = container.image.labels
        except Exception:
            labels = None
    if labels is None:
        try:
            labels = docker_host_client.images.get(train_container.image_id).labels
        except Exception:
            labels = None

    result = {
        "supported": metrics_supported(labels),
        "epochs": [],
        "gpu": [],
        "gpu_available": False,
    }
    if container is None or not result["supported"]:
        return result

    result["epochs"] = parse_metrics_jsonl(
        read_from_container(container.id, "/train/metrics.jsonl")
    )
    result["gpu"] = parse_gpu_csv(read_from_container(container.id, "/train/gpu.csv"))
    result["gpu_available"] = bool(result["gpu"])
    return result


@router.post(
    "/containers/{train_container_id}/clone",
    status_code=status.HTTP_201_CREATED,
    response_model=TrainContainer,
)
async def clone_train_container(
    request: Request, train_container_id: int, db: Session = Depends(get_db)
) -> TrainContainer:
    """Creates a copy of a built training container with another train_config.

    The copy uses the same image (no build). It gets its own container on the first run.
    Body: {"train_config": {...}}, the complete configuration; nothing else is accepted.
    """
    user: KeycloakUser = request.user
    base = check_exists(sql_api.get_train_container(db, train_container_id))

    try:
        data = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Body is not valid JSON.")
    if (
        not isinstance(data, dict)
        or set(data) != {"train_config"}
        or not isinstance(data["train_config"], dict)
    ):
        raise HTTPException(
            status_code=400,
            detail="Body must be a JSON object with the single key 'train_config'.",
        )
    train_config = data["train_config"]

    config_filepath, ext = get_config_filepath(
        TEMPLATES_PATH, base.provider, base.architecture
    )
    if ext == ".jsonschema":
        try:
            validate(train_config, jsonfile2dict(config_filepath))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return sql_api.create_train_container(
        db,
        owner=user.username,
        image_id=base.image_id,
        repository=base.repository,
        tag=base.tag,
        last_modified=datetime.now(),
        provider=base.provider,
        category=base.category,
        architecture=base.architecture,
        dataset_id=base.dataset_id,
        dataset=base.dataset,
        model_filepath=base.model_filepath,
        score_name=base.score_name,
        score_regexp=base.score_regexp,
        config_override=train_config,
    )


@router.get("/containers/{train_container_id}/model")
async def get_train_container_model(
    request: Request, train_container_id: int, db: Session = Depends(get_db)
) -> Model:
    try:
        user: KeycloakUser = request.user

        train_container = check_exists(
            sql_api.get_train_container(db, train_container_id)
        )
        dataset = check_exists(
            dataset_sql_api.get_dataset(db, train_container.dataset_id),
        )

        if train_container.score is None:
            # Taking over a model is the moment its name is made, so read the
            # score now if nobody has asked for it yet. A failure leaves the
            # name without a value.
            try:
                if get_container_status(train_container.container_id) == "exited":
                    score = read_score_from_logs(train_container)
                    if score is not None:
                        train_container.score = score
                        sql_api.update_train_container(db, train_container)
            except Exception as e:
                logger.warning(
                    "score of train container %s not read: %s",
                    train_container_id,
                    e,
                )

        generic_model_filepath = train_container.model_filepath
        onnx_model_filepath = get_onnx_model_filepath(generic_model_filepath)

        trained_models = {
            candidate_model_filepath: read_from_container(
                train_container.container_id, candidate_model_filepath
            )
            for candidate_model_filepath in (
                onnx_model_filepath,
                generic_model_filepath,
            )
        }

        if not any(trained_models.values()):
            raise HTTPException(
                status_code=404,
                detail=f"Trained model file not found in Train Container at {' or '.join(trained_models.keys())}.",
            )

        trained_model_data, trained_model_filepath = (
            (trained_models[onnx_model_filepath], onnx_model_filepath)
            if trained_models[onnx_model_filepath] is not None
            else (trained_models[generic_model_filepath], generic_model_filepath)
        )

        trained_model_filename = os.path.basename(trained_model_filepath)
        train_container_logs = get_container(train_container.container_id).logs()

        model: Model = persist_model(
            db=db,
            user=user,
            name=model_name_for_training(train_container, dataset),
            filename=trained_model_filename,
            model_file=io.BytesIO(trained_model_data),
            format=infer_model_format(trained_model_filename),
        )

        persist_model_artifact(
            user=user,
            model=model,
            filename="train.log",
            data=train_container_logs,
        )

        if train_container.category == "Classification":
            try:
                labels = read_class_order(train_container.container_id)
                shape = model.output_shape
                if labels and (not shape or shape[-1] == len(labels)):
                    model.output_labels = labels
                    db.commit()
                    db.refresh(model)
                elif labels:
                    logger.warning(
                        "class order of train container %s not set: %d labels, output shape %s",
                        train_container_id,
                        len(labels),
                        shape,
                    )
            except Exception as e:
                logger.warning(
                    "class order of train container %s not set: %s",
                    train_container_id,
                    e,
                )

        return model
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/containers/{train_container_id}/status")
def get_train_container_status(
    train_container_id: int, db: Session = Depends(get_db)
) -> Dict:
    train_container = check_exists(sql_api.get_train_container(db, train_container_id))
    return {
        "status": get_container_status(train_container.container_id),
        "score": train_container.score,
    }


@router.delete("/containers/{train_container_id}")
def delete_train_container(
    train_container_id: int, db: Session = Depends(get_db)
) -> Response:
    train_container = check_exists(sql_api.get_train_container(db, train_container_id))
    if train_container.container_id is not None:
        remove_container(train_container.container_id, force=True)
    # Clones share the image of their base container: keep it while another row uses it.
    image_shared = (
        sql_api.count_train_containers_with_image(
            db, train_container.image_id, exclude_id=train_container.id
        )
        > 0
    )
    if not image_shared:
        # Do not force remove images as they could be used by multiple containers
        remove_image(train_container.image_id, force=False)
    sql_api.delete_train_container(db, train_container)
    return Response(status_code=204)


@router.patch("/containers/{train_container_id}", response_model=TrainContainer)
def update_train_container(
    train_container_id: int,
    train_container_update: TrainContainer,
    db: Session = Depends(get_db),
) -> TrainContainer:
    stored_train_container = check_exists(
        sql_api.get_train_container(db, train_container_id)
    )
    update_data = train_container_update.dict(exclude_unset=True)
    updated_train_container = stored_train_container.copy(update=update_data)
    sql_api.update_train_container(db, updated_train_container)

    return updated_train_container


@router.get("/containers/{train_container_id}/logs")
def get_train_container_logs(
    train_container_id: int,
    tail: int = None,
    max_length: int = 20000,
    db: Session = Depends(get_db),
) -> Dict:
    train_container = check_exists(sql_api.get_train_container(db, train_container_id))

    if train_container.container_id is None:
        raise HTTPException(status_code=404, detail="No logs found.")

    train_container = get_container(train_container.container_id)
    logs = (
        train_container.logs()
        if (tail is None or tail < 0)
        else train_container.logs(tail=tail)
    ).decode("utf-8")

    if tail is not None and len(logs) > max_length:
        logs = logs[-max_length:]

    return {"logs": logs}


@router.get("/containers/{train_container_id}/config")
def get_train_container_config(
    train_container_id: int,
    db: Session = Depends(get_db),
) -> Dict:
    train_container = check_exists(sql_api.get_train_container(db, train_container_id))

    # A container that was built but never run (and every clone) has no Docker container yet: its
    # settings are the ones baked into the image. Read them from a short-lived stopped container.
    scratch = None
    reader = train_container
    if train_container.container_id is None:
        try:
            scratch = docker_host_client.containers.create(image=train_container.image_id)
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"No Train Container found and the image could not be opened: {e}",
            )
        reader = types.SimpleNamespace(
            container_id=scratch.id,
            provider=train_container.provider,
            architecture=train_container.architecture,
        )

    try:
        config_obj = _read_config_obj(train_container, reader)
    finally:
        if scratch is not None:
            try:
                scratch.remove(force=True)
            except Exception as e:
                logger.error(f"Unable to remove scratch container {scratch.id}: {e}")

    return {
        "container_id": train_container_id,
        "config": config_obj,
        "provider": train_container.provider,
        "architecture": {
            "category": train_container.category,
            "name": train_container.architecture,
        },
        "dataset_id": train_container.dataset_id,
    }


def _read_config_obj(train_container, reader) -> Dict:
    config_obj = {}
    for config_filename, config_key in [
        ("train_config.json", "train"),
        ("export_config.json", "export"),
    ]:
        config_json = read_train_container_config(
            config_filename=config_filename,
            train_container=reader,
            templates_path=TEMPLATES_PATH,
        )
        if config_key == "train" and train_container.config_override:
            config_json = {**config_json, **train_container.config_override}

        if config_key == "train":
            schema_with_values = get_train_config_for_architecture(
                train_container.provider, train_container.architecture
            )
        elif config_key == "export":
            schema_with_values = get_export_config()
        schema, config = itemgetter("schema", "values")(schema_with_values)

        config_obj[config_key] = {
            "schema": schema,
            "values": {k: v for k, v in config_json.items() if k in config},
            "empty": not config_json,
        }

    return config_obj


@router.get("/providers")
def get_providers() -> List[str]:
    return [
        os.path.basename(os.path.dirname(d))
        for d in glob.glob(os.path.join(TEMPLATES_PATH, "*", ""))
    ]


@router.get("/architectures/{provider}")
def get_architectures(provider: str) -> List[Dict]:
    config_filepaths = get_config_filepaths(TEMPLATES_PATH, provider, provider)
    return [
        {
            "category": config_filepath.parent.parent.parent.name,
            "name": config_filepath.parent.parent.name,
        }
        for config_filepath in config_filepaths
    ]


@router.post("/config")
async def process_train_config(
    request: Request,
    task_creator: TaskCreator = Depends(get_task_creator),
) -> Response:
    user: KeycloakUser = request.user

    data = await request.json()

    (
        provider,
        category,
        architecture,
        dataset_id,
        train_config,
        export_config,
    ) = itemgetter(
        "provider",
        "category",
        "architecture",
        "dataset_id",
        "train_config",
        "export_config",
    )(
        data
    )

    config_filepath, _ = get_config_filepath(TEMPLATES_PATH, provider, architecture)
    train_configs_path = os.path.dirname(config_filepath)

    _, task_location_url, _ = task_creator.create_background_task(
        build_train_image,
        task_title=f"Train Container Buildjob: {architecture} ({provider})",
        provider=provider,
        category=category,
        architecture=architecture,
        dataset_id=dataset_id,
        train_config=train_config,
        export_config=export_config,
        train_configs_path=train_configs_path,
        owner=user.username,
    )

    headers = {"Location": task_location_url}
    return Response(status_code=status.HTTP_202_ACCEPTED, headers=headers)


@router.put("/config")
async def update_train_config(
    request: Request,
    db: Session = Depends(get_db),
) -> Dict:
    data = await request.json()

    (
        train_config,
        export_config,
        dataset_id,
        train_container_id,
    ) = itemgetter(
        "train_config", "export_config", "dataset_id", "container_id"
    )(data)

    train_container = check_exists(sql_api.get_train_container(db, train_container_id))

    if train_container.container_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"No Train Container '{train_container.container_id}' found.",
        )

    for config, config_filename in [
        (train_config, "train_config.json"),
        (export_config, "export_config.json"),
    ]:
        update_train_container_config(
            config=config,
            config_filename=config_filename,
            train_container=train_container,
            templates_path=TEMPLATES_PATH,
        )

    update_train_container_dataset(
        db=db,
        dataset_id=dataset_id,
        train_container=train_container,
        templates_path=TEMPLATES_PATH,
    )

    return {
        "train_config": train_config,
        "export_config": export_config,
        "container": train_container,
    }


@router.get("/config/{provider}/{architecture}")
def get_train_config_for_architecture(provider: str, architecture: str) -> Dict:
    config_filepath, _ = get_config_filepath(TEMPLATES_PATH, provider, architecture)
    train_json_schema, train_config_values = get_config_with_schema(config_filepath)

    return {
        "schema": train_json_schema,
        "values": train_config_values,
    }


@router.post("/config/download")
async def download_train_config(request: Request):
    data = await request.json()

    provider, architecture, train_config = itemgetter(
        "provider", "architecture", "train_config"
    )(data)

    _, ext = get_config_filepath(TEMPLATES_PATH, provider, architecture)

    train_config = json.dumps(train_config, indent=4)

    if ext == ".jsonschema":
        ext = ".json"

    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return create_single_file_response(
        bytes(train_config, "utf-8"),
        filename=f"{provider.lower()}-{architecture.lower()}_{now}-train{ext}",
        content_type="application/json",
    )


@router.post("/config/upload")
def upload_train_config(
    provider: str = Form(...),
    architecture: str = Form(...),
    config_file: UploadFile = File(...),
) -> Dict:
    try:
        config_filepath, ext = get_config_filepath(
            TEMPLATES_PATH, provider, architecture
        )
        if ext != ".jsonschema":
            raise RuntimeError
    except:
        raise HTTPException(
            status_code=500,
            detail="Unable to verify schema of configuration file.",
        )

    try:
        train_config_values = json.loads(config_file.file.read().decode("utf-8"))
        train_config_schema = jsonfile2dict(config_filepath)
        validate(train_config_values, train_config_schema)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return train_config_values


@router.get("/config/export")
def get_export_config() -> Dict:
    export_schema_filepath = get_export_schema_filepath(EXPORT_PATH)
    export_json_schema, export_config_values = get_config_with_schema(
        export_schema_filepath
    )

    return {
        "schema": export_json_schema,
        "values": export_config_values,
    }


@router.post("/template/upload", status_code=status.HTTP_201_CREATED)
def upload_train_container_template(
    template_zip: UploadFile = File(...),
) -> Dict:
    try:
        with tempfile.TemporaryDirectory() as template_directory:
            zip_bytes = template_zip.file.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(template_directory)

            template_directory_entries = os.listdir(template_directory)
            if len(template_directory_entries) != 1 or not os.path.isdir(
                os.path.join(template_directory, template_directory_entries[0])
            ):
                raise RuntimeError("Archive did not extract as a single directory.")

            template_dirname = template_directory_entries[0]
            template_root_dir = os.path.join(template_directory, template_dirname)
            provider, category, architecture = validate_template(
                MODEL_TRAINING_PATH, template_root_dir
            )
            install_template(
                TEMPLATES_PATH, template_root_dir, provider, category, architecture
            )
            return {
                "provider": provider,
                "architecture": {"category": category, "name": architecture},
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/template")
async def delete_train_container_template(request: Request) -> Response:
    data = await request.json()
    provider, architecture = itemgetter("provider", "architecture")(data)
    uninstall_template(TEMPLATES_PATH, provider, architecture)
    return Response(status_code=204)
