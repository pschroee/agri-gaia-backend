# SPDX-FileCopyrightText: 2024 Osnabrück University of Applied Sciences
# SPDX-FileContributor: Andreas Schliebitz
# SPDX-FileContributor: Henri Graf
# SPDX-FileContributor: Jonas Tüpker
# SPDX-FileContributor: Lukas Hesse
# SPDX-FileContributor: Maik Fruhner
# SPDX-FileContributor: Prof. Dr.-Ing. Heiko Tapken
# SPDX-FileContributor: Tobias Wamhof
#
# SPDX-License-Identifier: MIT

import re
import io
import logging
import os
import tarfile
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import python_on_whales


from agri_gaia_backend.db.models import EdgeDevice, InferenceContainerTemplate, Model
from agri_gaia_backend.services.container_template.definitions import (
    INFERENCE_CONTAINER_TEMPLATES_DIR,
)
from agri_gaia_backend.services.docker.client import build_container_client as docker
from agri_gaia_backend.services.docker import image_util
from agri_gaia_backend.services.minio_api.client import (
    MINIO_HOST,
    MINIO_IP_ADDRESS,
)

logger = logging.getLogger("api-logger")

REGISTRY_URL = os.environ.get("REGISTRY_URL")


class MissingInputDataException(Exception):
    def __init__(
        self, model_properties: List[str] = [], device_properties: List[str] = []
    ):
        self.model_properties = model_properties
        self.device_properties = device_properties


def _get_device_metadata(edge_info: Union[EdgeDevice, Dict]):
    if isinstance(edge_info, Dict):
        device_metadata = edge_info
    else:
        device_metadata = {"device_type": "generic", "architecture": edge_info.arch}

    return device_metadata


def _validate_device_metadata(edge_info: Union[EdgeDevice, Dict]) -> List[str]:
    if not edge_info["architecture"]:
        return ["architecture"]
    return []


def _validate_model_properties(model: Model) -> List[str]:
    required_properties = [
        "input_shape",
        "input_datatype",
        "input_semantics",
        "output_datatype",
        "output_shape",
        "format",
    ]
    empty_required_properties = [
        propertyname
        for propertyname, propertyvalue in model.__dict__.items()
        if propertyname in required_properties and propertyvalue is None
    ]
    return empty_required_properties


def _create_fs_build_context(build_context: bytes, directory: str):
    memoryfile = io.BytesIO(build_context)
    with tarfile.open(fileobj=memoryfile, mode="r:gz") as tar:
        tar.extractall(directory)


def create_repository_url(repository_name: str) -> str:
    return f"{REGISTRY_URL}/{repository_name}"


def validate_generator_input(model: Model, edge_info: Union[EdgeDevice, Dict]) -> None:
    invalid_model_props = _validate_model_properties(model)
    device_metadata = _get_device_metadata(edge_info)
    invalid_device_props = _validate_device_metadata(device_metadata)

    if invalid_model_props or invalid_device_props:
        raise MissingInputDataException(invalid_model_props, invalid_device_props)


def build_and_push_image(
    inference_container_template: InferenceContainerTemplate,
    repository_name: str,
    image_tag: str,
    model: Model,
    edge_info: Union[EdgeDevice, Dict],
    status_callback: Callable[[str, dict], None] = lambda x, y: None,
    on_phase: Optional[Callable[[str, float], None]] = None,
) -> str:
    """
    Either device metadata or edge_device must be given
    """
    repository_url = create_repository_url(repository_name)
    device_metadata = _get_device_metadata(edge_info)

    invalid_model_props = _validate_model_properties(model)
    invalid_device_props = _validate_device_metadata(device_metadata)

    if invalid_model_props or invalid_device_props:
        raise MissingInputDataException(invalid_model_props, invalid_device_props)

    container_template_dir = (
        INFERENCE_CONTAINER_TEMPLATES_DIR / inference_container_template.dirname
    )

    build_context = image_util.create_tar_build_context(container_template_dir, model)

    with tempfile.TemporaryDirectory() as tempdirname:
        _create_fs_build_context(build_context, tempdirname)
        _build_and_push_image(
            tempdirname,
            repository_url=repository_url,
            image_tag=image_tag,
            platforms=[device_metadata["architecture"]],
            status_callback=status_callback,
            on_phase=on_phase,
            add_hosts={MINIO_HOST: MINIO_IP_ADDRESS},
        )

    return f"{repository_url}:{image_tag}"


PHASE_PREPARING = "Preparing the build"
PHASE_PUSHING = "Pushing the image to the registry"
PHASE_CREATING_CONTAINER = "Creating the training container"

# Share of the progress bar: building takes up to 85 %, the push 90 %, and the
# caller reports what it does afterwards (e.g. creating the container: 95 %).
BUILD_FRACTION = 0.85
PUSH_FRACTION = 0.9

_STEP_RE = re.compile(r"^#\d+\s+\[(?P<stage>[^\]]*?)\s*(?P<i>\d+)/(?P<m>\d+)\]\s*(?P<cmd>.*)$")
_PUSH_RE = re.compile(r"^#\d+\s+(exporting to image|pushing layers|pushing manifest|exporting layers|exporting manifest)")
_MAX_COMMAND_LENGTH = 60


class BuildProgress:
    """Turns BuildKit plain-progress lines into (phase text, fraction) updates.

    Only the lines `#<n> [<stage> <i>/<m>] <command>` (a build step) and the
    export/push lines are of interest. Updates are throttled to one per
    `min_interval` seconds so that a build with hundreds of lines does not hit
    the database for each of them; the latest state is never lost: `finish()`
    returns it if it was held back.
    """

    def __init__(
        self,
        on_phase: Callable[[str, float], None],
        min_interval: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._on_phase = on_phase
        self._min_interval = min_interval
        self._clock = clock
        self._last_emit: Optional[float] = None
        self._pending: Optional[Tuple[str, float]] = None
        self._fraction = 0.0
        self._pushing = False
        self._emit((PHASE_PREPARING, 0.01), force=True)

    def _emit(self, update: Tuple[str, float], force: bool = False) -> None:
        now = self._clock()
        if (
            force
            or self._last_emit is None
            or now - self._last_emit >= self._min_interval
        ):
            self._last_emit = now
            self._pending = None
            self._on_phase(*update)
        else:
            self._pending = update

    def feed(self, line: str) -> None:
        line = line.strip()
        if not line.startswith("#"):
            return
        if _PUSH_RE.match(line):
            if not self._pushing:
                self._pushing = True
                self._emit((PHASE_PUSHING, PUSH_FRACTION), force=True)
            return
        if self._pushing:
            return
        match = _STEP_RE.match(line)
        if not match:
            return
        i, m = int(match["i"]), int(match["m"])
        if m <= 0 or i > m:
            return
        # Steps of several stages interleave; the bar must not run backwards.
        self._fraction = max(self._fraction, round(i / m * BUILD_FRACTION, 3))
        command = " ".join(match["cmd"].split())
        if len(command) > _MAX_COMMAND_LENGTH:
            command = command[: _MAX_COMMAND_LENGTH - 1] + "…"
        text = f"Building the image · step {i}/{m}"
        if command:
            text += f" · {command}"
        self._emit((text, max(self._fraction, 0.01)))

    def finish(self) -> None:
        if self._pending is not None:
            update, self._pending = self._pending, None
            self._on_phase(*update)


def _build_and_push_image(
    context_directory: str,
    repository_url: str,
    image_tag: str,
    status_callback: Callable[[str, dict], None],
    on_phase: Optional[Callable[[str, float], None]] = None,
    **kwargs,
):
    """Builds an image from the given dockerfile.

    Args:
        context_directory   : the directory in the filesystem or the build context
        repository_url      : the repository url that the image will be tagged with.
        image_tag           : the tag that the image will get (only the part after the colon).
        status_callback     : a callback function for statusupdates during the build
        on_phase            : optional, called with (phase text, fraction between 0 and 1)
            while the image is built and pushed, at most every 2 seconds
        buildargs           : additional arguments that can be used in build phase
        network             : the network the build container shall connect to. Set this to 'host'
            if the build container needs to connect to the platform services in RUN statements.
            This should only be used for static, non user provided dockerfiles!
    """
    repository_url_with_tag = f"{repository_url}:{image_tag}"
    progress = BuildProgress(on_phase) if on_phase else None
    try:
        logs_generator = docker.buildx.build(
            context_path=context_directory,
            tags=repository_url_with_tag,
            pull=True,
            push=True,
            stream_logs=True,
            **kwargs,
        )

        build_successful = True
        for log in logs_generator:
            status_callback("update", log)
            if progress:
                progress.feed(log)
        if progress:
            progress.finish()

    except python_on_whales.exceptions.DockerException as e:
        status_callback("failed", {"reason": str(e)})
        raise e
