#!/usr/bin/env python

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

# -*- coding: utf-8 -*-

import os
import json
import tarfile
import docker
import tempfile
import subprocess
import logging


from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Set
from agri_gaia_backend.db import train_api as sql_api
from agri_gaia_backend.services.docker.client import (
    build_container_client as docker_platform_client,
)
from agri_gaia_backend.services.docker.client import host_client as docker_host_client
from agri_gaia_backend.schemas.train_container import TrainContainer
from agri_gaia_backend.util.common import (
    delete_keys_from_dict,
    mv,
    rm,
    mkdir,
    is_valid_json,
)
from agri_gaia_backend.util.jsonschema import JSONSchemaDefault
from docker.models.containers import Container
from fastapi import HTTPException
from genson import SchemaBuilder
from dockerfile_parse import DockerfileParser
from dotenv import dotenv_values, set_key
from sqlalchemy.orm import Session

DEFAULT_CONTAINER_WORKSPACE = "/workspace"
ENV_FILENAME = ".env"
TRAIN_CONFIG_FILENAME = "train_config.json"
EXPORT_CONFIG_FILENAME = "export_config.json"

logger = logging.getLogger("api-logger")


def dict2jsonschema(json_dict: Dict, keys_to_remove=["required"]) -> Dict:
    builder = SchemaBuilder()
    builder.add_object(json_dict)
    schema = builder.to_schema()
    delete_keys_from_dict(schema, keys_to_remove)
    return schema


def jsonschema2defaults(json_schema: Dict) -> Dict:
    return JSONSchemaDefault(json_schema).get_default_values()


def jsonfile2dict(json_filepath: str) -> Dict:
    with open(json_filepath, "r") as fh:
        return json.load(fh)


def get_filepaths(
    path: str,
    recursive: bool = True,
    hidden: bool = False,
    filenames_only: bool = False,
) -> List[Union[Path, str]]:
    p = Path(path)
    filepaths = filter(
        lambda p: p.is_file(), p.rglob("*") if recursive else p.glob("*")
    )
    if not hidden:
        filepaths = filter(
            lambda p: not any((part for part in p.parts if part.startswith("."))),
            filepaths,
        )

    if filenames_only:
        filepaths = map(lambda p: p.name, filepaths)

    return list(filepaths)


def get_directory_paths(path: str) -> List[Path]:
    return list(filter(lambda p: p.is_dir(), Path(path).rglob("*")))


def get_config_filepaths(
    templates_path: str, provider: str, architecture: str
) -> List[Path]:
    config_path = os.path.join(templates_path, provider)
    return list(
        filter(
            lambda p: architecture in str(p) and p.stem == "config",
            get_filepaths(config_path),
        )
    )


def get_dockerfile_filepath(templates_path, provider: str, architecture: str) -> str:
    config_path = os.path.join(templates_path, provider)
    dockerfile_filepath = list(
        filter(
            lambda p: architecture in str(p) and p.name == "Dockerfile",
            get_filepaths(config_path),
        )
    )

    if len(dockerfile_filepath) != 1:
        raise HTTPException(
            status_code=404,
            detail=f"Dockerfile for '{architecture}' ('{provider}') not found.",
        )

    return dockerfile_filepath[0]


def get_config_filepath(
    templates_path: str, provider: str, architecture: str
) -> Tuple[str, str]:
    config_filepaths = get_config_filepaths(templates_path, provider, architecture)

    if not config_filepaths:
        raise HTTPException(
            status_code=404,
            detail=f"Configuration file for '{architecture}' ('{provider}') not found.",
        )

    config_filepath = config_filepaths[0]
    _, ext = os.path.splitext(config_filepath)
    return config_filepath, ext


def get_container(container_id: str) -> Container:
    try:
        return docker_host_client.containers.get(container_id)
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail="Container not found.")


def container_exists(container_id: str) -> bool:
    containers = docker_host_client.containers.list(
        all=True, filters={"id": container_id}
    )
    return len(containers) != 0


def image_exists(image_name: str, where: str = "platform") -> Optional[bool]:
    if where not in {"platform", "host"}:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot check existence of image '{image_name}' on '{where}'.",
        )

    images = (
        [
            image_name == tag
            for image in docker_platform_client.image.list()
            for tag in image.repo_tags
        ]
        if where == "platform"
        else [
            image_name == tag
            for image in docker_host_client.images.list()
            for tag in image.tags
        ]
    )

    return any(images)


def pull_image_if_missing(image_name: str) -> None:
    """Pulls a train container image to the host if it is not present yet.

    Pulling a cold image can take several minutes. Values written into the
    container environment at creation time, such as an OIDC access token, must
    therefore be gathered only after this call returned.
    """
    try:
        docker_host_client.images.get(image_name)
    except docker.errors.ImageNotFound:
        print("Pulling train container image to host:", image_name)
        docker_host_client.images.pull(image_name)


def remove_container(container_id: str, force: bool = False) -> None:
    try:
        if container_exists(container_id):
            print("Removing train container:", container_id)
            container = get_container(container_id)
            container.remove(force=force)
        else:
            print("Container does not exist:", container_id)
    except docker.errors.APIError as e:
        raise HTTPException(status_code=500, detail=str(e))


def remove_image_from_platform(image_name: str, force: bool = False) -> None:
    if image_exists(image_name, where="platform"):
        try:
            docker_platform_client.image.remove(image_name, force=force)
            print("Removed train container image from platform registry:", image_name)
        except Exception as e:
            print(str(e))
    else:
        print("Image does not exist on platform:", image_name)


def remove_image_from_host(image_name: str, force: bool = False) -> None:
    if image_exists(image_name, where="host"):
        try:
            docker_host_client.images.remove(image_name, force=force)
            print("Removed train container image from host registry:", image_name)
        except Exception as e:
            print(str(e))
    else:
        print("Image does not exist on host:", image_name)


def remove_image(image_name: str, force: bool = False) -> None:
    remove_image_from_platform(image_name, force)
    remove_image_from_host(image_name, force)


def get_container_status(container_id: str) -> str:
    return get_container(container_id).status if container_id is not None else "created"


def is_float(string: str) -> bool:
    try:
        float(string)
        return True
    except:
        return False


def exists_in_container(container_id: str, path: str, is_file=False) -> bool:
    test_flag = "f" if is_file else "d"
    container = get_container(container_id)
    exit_code, _ = container.exec_run(f"test -{test_flag} {path}")
    return exit_code == 0


def get_container_contents(container_id: str, path: str) -> Tuple:
    container = get_container(container_id)
    try:
        data, stat = container.get_archive(path)
        return data, stat
    except docker.errors.APIError:
        return None, None


def get_onnx_model_filepath(model_filepath: str) -> str:
    model_filepath = Path(model_filepath)
    return str(Path(model_filepath.parent, model_filepath.stem + ".onnx"))


def infer_model_format(filename: str) -> Union[str, None]:
    _, ext = os.path.splitext(filename)
    if ext in {".pt", ".pth", ".torchscript"}:
        return "pytorch"
    if ext == ".onnx":
        return "onnx"
    if ext == ".plan":
        return "tensorrt"
    if ext in {".graphdef", ".tf"}:
        return "tensorflow"
    return None


def validate_template(
    model_training_path: str, template_root_dir: str
) -> Tuple[str, str, str]:
    def load_directory_schemas(model_training_path: str) -> Dict[str, Dict]:
        # Each directory schema file contains the output of 'tree -J'.
        directory_schema_filepaths = Path(model_training_path, "validation").rglob(
            "*.json"
        )
        directory_schemas = {}
        for directory_schema_filepath in directory_schema_filepaths:
            with open(directory_schema_filepath, "r") as fh:
                directory_schemas[directory_schema_filepath.stem] = json.load(fh)
        return directory_schemas

    def traverse_check(root_dir, path: Path, directory_schema: Dict):
        contents = directory_schema["contents"]
        for content in contents:
            entry = path.joinpath(content["name"])
            if not entry.exists():
                raise RuntimeError(
                    f"Validation error: Entry '{entry.relative_to(root_dir)}' not found."
                )
            if entry.suffix in {".json", ".jsonschema"}:
                valid_json, error_msg = is_valid_json(entry)
                if not valid_json:
                    raise RuntimeError(
                        f"Invalid JSON file '{entry.relative_to(root_dir)}': {error_msg}"
                    )
            if content["type"] == "directory":
                traverse_check(root_dir, entry, content)

    def get_template_dir(template_root_dir: Path) -> Path:
        config_dir = list(
            filter(
                lambda path: any(part == "config" for part in path.parts),
                get_directory_paths(template_root_dir),
            )
        )
        if len(config_dir) != 1:
            raise RuntimeError(
                "None or multiple configuration directories 'config/' found."
            )
        template_dir = config_dir[0].parent
        return template_dir

    def validate_directory_structure(
        template_root_dir: str, directory_schemas: Dict[str, Dict]
    ) -> Path:
        template_root_dir = Path(template_root_dir)
        template_dir = get_template_dir(template_root_dir)

        template_dir_rel = template_dir.relative_to(template_root_dir)
        if len(template_dir_rel.parts) != 3:
            raise RuntimeError(
                "Either provider, category or architecture are missing or there are additional parent directories relative to 'config/'."
            )

        traverse_check(
            template_root_dir, template_root_dir, directory_schemas["common"]
        )
        traverse_check(template_root_dir, template_dir, directory_schemas["template"])
        return template_dir_rel

    directory_schemas = load_directory_schemas(model_training_path)
    template_dir_rel = validate_directory_structure(
        template_root_dir, directory_schemas
    )

    return template_dir_rel.parts


def install_template(
    templates_path: str,
    template_root_dir: str,
    provider: str,
    category: str,
    architecture: str,
) -> None:
    source_path = os.path.join(template_root_dir, provider, category, architecture)
    target_path = os.path.normpath(
        os.path.join(templates_path, provider, category, architecture)
    )
    if os.path.exists(target_path):
        rm(target_path)
    mkdir(os.path.dirname(target_path))
    mv(source_path, os.path.dirname(target_path))


def uninstall_template(templates_path: str, provider: str, architecture: str) -> None:
    template_root_dir = Path(
        get_config_filepath(templates_path, provider, architecture)[0]
    ).parent.parent

    if Path(templates_path) not in template_root_dir.parents:
        raise HTTPException(
            status_code=500, detail="Train Container Template cannot be removed safely."
        )

    rm(template_root_dir)

    path = template_root_dir.parent
    while path != Path(templates_path) and not any(path.iterdir()):
        rm(path)
        path = path.parent


def read_from_container(container_id: str, container_filepath: str) -> bytes:
    tar_data, _ = get_container_contents(container_id, container_filepath)
    if tar_data is None:
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_filepath = os.path.join(tmp_dir, "archive.tar")
        with open(tar_filepath, "wb") as fh:
            for tar_chunk in tar_data:
                fh.write(tar_chunk)

        with tarfile.open(tar_filepath) as tar:
            tar.extractall(tmp_dir)

        filename = os.path.basename(container_filepath)
        filepath = os.path.join(tmp_dir, filename)
        with open(filepath, "rb") as fh:
            return fh.read()


def tar_compress(src_path: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_filepath = os.path.join(tmp_dir, "archive.tar")
        with tarfile.open(output_filepath, "w") as tar:
            tar.add(src_path, arcname=os.path.basename(src_path))
        with open(output_filepath, "rb") as fh:
            return fh.read()


def write_to_container(container_id: str, container_path: str, src: str) -> None:
    docker_cp_cmd = ["docker", "cp", src, f"{container_id}:{container_path}"]
    logger.info(" ".join(docker_cp_cmd))
    subprocess.check_output(docker_cp_cmd, universal_newlines=True)


def parse_dockerfile(dockerfile_filepath: str) -> Dict:
    dfp = DockerfileParser(path=str(dockerfile_filepath))
    return dfp.structure


def get_container_filepath(
    filename: str, templates_path: str, provider: str, architecture: str
) -> Path:
    dockerfile_filepath = get_dockerfile_filepath(
        templates_path, provider, architecture
    )
    df = parse_dockerfile(dockerfile_filepath)

    train_config_copy_instruction = next(
        filter(
            lambda line: (
                line["instruction"].upper() == "COPY" and filename in line["value"]
            ),
            df,
        )
    )

    copy_dst = Path(train_config_copy_instruction["value"].split(" ")[-1])
    if copy_dst.name != filename:
        copy_dst = copy_dst.joinpath(filename)

    if not copy_dst.is_absolute():
        train_config_copy_startline = train_config_copy_instruction["startline"]

        workdir_instructions = list(
            filter(
                lambda instruction: instruction["instruction"].upper() == "WORKDIR"
                and instruction["endline"] < train_config_copy_startline,
                df,
            )
        )

        if workdir_instructions:
            copy_dst = Path(workdir_instructions[-1]["value"], copy_dst)
        else:
            raise RuntimeError(f"Unable to find WORKDIR for file '{copy_dst}'")
    return copy_dst


def get_config_container_filepath(
    config_filename: str, templates_path: str, provider: str, architecture: str
) -> Path:
    try:
        assert config_filename in {TRAIN_CONFIG_FILENAME, EXPORT_CONFIG_FILENAME}
        return get_container_filepath(
            config_filename, templates_path, provider, architecture
        )
    except Exception as e:
        logger.warning(str(e))
        return default_container_filepath(config_filename)


def get_env_container_filepath(
    templates_path: str, provider: str, architecture: str
) -> Path:
    try:
        return get_container_filepath(
            ENV_FILENAME, templates_path, provider, architecture
        )
    except Exception as e:
        logger.warning(str(e))
        return default_container_filepath(ENV_FILENAME)


def default_container_filepath(filename: str) -> Path:
    return Path(DEFAULT_CONTAINER_WORKSPACE, filename)


def get_config_with_schema(config_filepath: str) -> Tuple[Dict, Dict]:
    _, ext = os.path.splitext(config_filepath)
    if ext == ".json":
        config_values = jsonfile2dict(config_filepath)
        json_schema = dict2jsonschema(config_values)
    elif ext == ".jsonschema":
        json_schema = jsonfile2dict(config_filepath)
        config_values = jsonschema2defaults(json_schema)
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported config file extension '{ext}'."
        )
    return json_schema, config_values


def get_export_schema_filepath(export_path: str) -> str:
    export_schema_filepath = os.path.join(export_path, "export.jsonschema")

    if not os.path.isfile(export_schema_filepath):
        raise HTTPException(
            status_code=404,
            detail=f"JSON schema file for model export not found.",
        )

    return export_schema_filepath


def update_train_container_config(
    config: Dict,
    config_filename: str,
    train_container: TrainContainer,
    templates_path: str,
) -> None:
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_filepath = os.path.join(tmp_dir, config_filename)
            with open(config_filepath, "w") as fh:
                if not config:
                    config = {}
                json.dump(config, fh, indent=4)

            config_container_filepath = get_config_container_filepath(
                config_filename=TRAIN_CONFIG_FILENAME,
                templates_path=templates_path,
                provider=train_container.provider,
                architecture=train_container.architecture,
            )

            write_to_container(
                container_id=train_container.container_id,
                container_path=config_container_filepath.parent,
                src=config_filepath,
            )
    except subprocess.CalledProcessError as e:
        error_msg = str(e.output)
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


def update_train_container_dataset(
    db: Session,
    dataset_id: int,
    train_container: TrainContainer,
    templates_path: str,
) -> None:
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            if train_container.dataset_id != dataset_id:
                env_container_filepath = get_env_container_filepath(
                    templates_path=templates_path,
                    provider=train_container.provider,
                    architecture=train_container.architecture,
                )

                env_data = read_from_container(
                    train_container.container_id,
                    container_filepath=env_container_filepath,
                )

                container_env = {}
                env_filepath = os.path.join(tmp_dir, ".env")
                if env_data:
                    env_str = env_data.decode("utf-8")
                    with open(env_filepath, "w") as fh:
                        fh.write(env_str)
                    container_env = dotenv_values(env_filepath)

                container_env["DATASET_ID"] = str(dataset_id)

                for env_name, env_value in container_env.items():
                    set_key(
                        dotenv_path=env_filepath,
                        key_to_set=env_name,
                        value_to_set=env_value,
                    )

                write_to_container(
                    container_id=train_container.container_id,
                    container_path=env_container_filepath.parent,
                    src=env_filepath,
                )

                train_container.dataset_id = dataset_id
                sql_api.update_train_container(db, train_container)
    except subprocess.CalledProcessError as e:
        error_msg = str(e.output)
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


def read_train_container_config(
    train_container: TrainContainer, config_filename: str, templates_path: str
) -> Dict:
    train_config_container_filepath = get_config_container_filepath(
        config_filename=config_filename,
        templates_path=templates_path,
        provider=train_container.provider,
        architecture=train_container.architecture,
    )

    train_config_data = read_from_container(
        train_container.container_id, container_filepath=train_config_container_filepath
    )

    if train_config_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Train Configuration '{train_config_container_filepath}' not found in Train Container '{train_container.container_id}'.",
        )

    train_config_json = json.loads(train_config_data.decode("utf-8"))
    return train_config_json
