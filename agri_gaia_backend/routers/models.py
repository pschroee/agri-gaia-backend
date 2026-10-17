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

import base64
import io
import datetime
import json
import logging
import struct
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List, Optional, Union, Tuple

from agri_gaia_backend.db import container_api as container_sql_api
from agri_gaia_backend.db import model_api as sql_api
from agri_gaia_backend.db import dataset_api as dataset_sql_api
from agri_gaia_backend.db import models
from agri_gaia_backend.routers import common
from agri_gaia_backend.routers.common import check_exists, get_db
from agri_gaia_backend.schemas.keycloak_user import KeycloakUser
from agri_gaia_backend.schemas.metadata import MetadataDocument
from agri_gaia_backend.schemas.model import Model, ModelPatch
from agri_gaia_backend.services import minio_api
from agri_gaia_backend.services.edc.connector import (
    create_catalog_entry_model,
    delete_catalog_entry_model,
)
from agri_gaia_backend.services.graph.sparql_operations import (
    models as sparql_models_api,
)
from agri_gaia_backend.services.graph.sparql_operations import util as sparql_util
from agri_gaia_backend.services.minio_api import MINIO_ENDPOINT
from agri_gaia_backend.services.model import model_metadata
from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.datastructures import UploadFile
from fastapi.param_functions import File
from sqlalchemy.orm import Session

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException, triton_to_np_dtype
import tritonclient.grpc.model_config_pb2 as mc

logger = logging.getLogger("api-logger")

ROOT_PATH = "/models"

router = APIRouter(prefix=ROOT_PATH)

"""
Only needed as long as EDC has no persistent catalogue storage.
Initially fills the catalogue with previous published datasets.
"""


@router.on_event("startup")
async def startup():
    logger.info("Creating Model EDC Catalog Entries...")
    # Does not work with Depends() in function call
    db = next(get_db())
    models = sql_api.get_published_models(db, 0, 1000)
    for model in models:
        _create_catalog_entry(model)
    logger.info("Creating Model EDC Catalog Entries Done!")


@router.get("", response_model=List[Model])
def get_all_models(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return sql_api.get_models(db, skip=skip, limit=limit)


@router.get("/keyword")
def get_all_data_for_keyword(
    uri: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    possible_concepts = sparql_util.query_narrower_concepts(uri)
    result = sparql_models_api.query_for_concepts(possible_concepts)

    uris = []
    for res in result["results"]["bindings"]:
        uris.append(res["iri"]["value"])

    models = sql_api.get_models_by_metadata_uri(db, skip=skip, limit=limit, uris=uris)

    return models


@router.get("/{model_id}", response_model=Model)
def get_model(model_id: int, db: Session = Depends(get_db)):
    return check_exists(sql_api.get_model(db, model_id))


@router.get("/{model_id}/metadata", response_model=MetadataDocument)
def get_model_metadata(
    request: Request, model_id: int, db: Session = Depends(get_db)
):
    """Returns the description of a model from the triple store (X4, #146).

    Models carry no uploader metadata; the field stays empty so that the answer has the
    same shape as the dataset's. Only the owner may read the description.
    """
    model = check_exists(sql_api.get_model(db, model_id))
    if model.owner != request.user.username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the owner can read the metadata of this model.",
        )
    description = ""
    if model.metadata_uri:
        description = sparql_util.get_description_for_uri(model.metadata_uri)
    return MetadataDocument(description=description, metadata={})


# Downloads all data from a passed dataset in a passed bucket, using the passed token for authentication and authorization
#
# dataset_name: the dataset name passed as a string
# bucket_name:  the bucket name passed as a string
@router.get("/{model_id}/download")
def download_model(request: Request, model_id: int, db: Session = Depends(get_db)):
    user: KeycloakUser = request.user
    model = check_exists(sql_api.get_model(db, model_id))

    bucket_name = model.bucket_name
    token = user.minio_token

    _validate_parameters(bucket_name, token)

    model_prefix = f"models/{model.id}"
    model_filepath = f"{model_prefix}/{model.file_name}"

    model_objects = minio_api.get_all_objects(
        bucket_name, prefix=model_prefix, token=token
    )

    if not len(model_objects):
        raise HTTPException(
            status_code=404,
            detail=f"No model objects found in bucket '{model_prefix}'.",
        )

    if len(model_objects) == 1 and model_filepath == model_objects[0].object_name:
        file_bytes = minio_api.get_object(
            bucket_name, object_name=model_filepath, token=token
        ).read()
        return common.create_single_file_response(file_bytes, model.file_name)

    downloaded_files = {
        item.object_name: minio_api.download_file(bucket_name, token, item).read()
        for item in model_objects
        if not item.is_dir
    }

    return common.create_zip_file_response(
        downloaded_files, filename=f"{model.name}.zip"
    )


def validate_name(name: str):
    if (
        name.lower() == "shapes"
        or name.lower() == "ontologies"
        or name.lower() == "agrovoc"
    ):
        raise ValueError("Cannot be named 'shapes', 'ontologies' or 'agrovoc'")


# TODO: Do we still need a list of files here?
@router.post("", response_model=Model, status_code=status.HTTP_201_CREATED)
def create_model(
    request: Request,
    name: str = Form(...),
    modelfile: UploadFile = File(...),
    labels: Optional[List[str]] = Form(None),
    description: str = Form(...),
    format: str = Form(...),
    db: Session = Depends(get_db),
):
    validate_name(name)
    user: KeycloakUser = request.user

    filename: str = modelfile.filename

    model: Model = persist_model(
        db=db,
        user=user,
        name=name,
        filename=filename,
        model_file=modelfile.file._file,
        format=format,
        labels=labels,
        description=description,
    )

    return model


def persist_model(
    db: Session,
    user: KeycloakUser,
    name: str,
    filename: str,
    model_file: io.BytesIO,
    format: str,
    labels: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> Model:
    created_model = create_initial_entry_postgres(
        db=db, user=user, name=name, filename=filename, format=format
    )
    fuseki_id: Optional[str] = None
    if description is not None:
        fuseki_id = save_model_metadata_to_fuseki(
            db=db, model=created_model, labels=labels, description=description
        )
    model_prefix = upload_model_to_minio(
        db=db,
        model=created_model,
        token=user.minio_token,
        filename=filename,
        model_file=model_file,
        fuseki_id=fuseki_id,
    )
    try:
        if created_model.format == models.ModelFormat.onnx:
            model_file.seek(0)
            model_bytes = model_file.read()
            model_metadata.set_onnx_input_output_metadata(
                created_model, io.BytesIO(model_bytes)
            )
    except Exception as ex:
        logger.error("Automatic metadata retrieval of onnx model failed")
        logger.exception(ex)

    model = save_model_to_postgres(
        db=db,
        model=created_model,
        token=user.minio_token,
        prefix=model_prefix,
        fuseki_id=fuseki_id,
    )
    return model


def persist_model_artifact(
    user: KeycloakUser, model: Model, filename: str, data: Union[bytes, io.BytesIO]
) -> str:
    try:
        _validate_parameters(model.bucket_name, user.minio_token)

        prefix = f"models/{model.id}"
        minio_api.upload_data(
            model.bucket_name,
            prefix=prefix,
            token=user.minio_token,
            data=data,
            objectname=filename,
        )

        return f"{prefix}/{filename}"
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Uploading model artifact '{filename}' to Minio failed.",
        )


def create_initial_entry_postgres(
    db: Session, user: KeycloakUser, name: str, filename: str, format: str
):
    try:
        # Replace model name by name(2), (3)... if name already exist
        if sql_api.get_model_by_name(db=db, name=name):
            count = 2
            while sql_api.get_model_by_name(db=db, name=f"{name}({count})"):
                count += 1
            name = f"{name}({count})"

        created_model = sql_api.create_model(
            db,
            name=name,
            format=format,
            owner=user.username,
            last_modified=datetime.datetime.now(),
            bucket_name=user.minio_bucket_name,
            file_size=None,
            file_name=filename,
        )

        return created_model
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(
            status_code=500,
            detail="Initial Insertion into Database failed. Please try again.",
        )


def save_model_metadata_to_fuseki(
    model: Model, labels: List[str], description: str, db: Session
):
    try:
        sparql_util.createFusekiDataset("model-" + str(model.id))

        uris = []
        if labels is not None:
            for label in labels:
                uris.append(sparql_util.convert_to_URI(label))

        graph, fuseki_id = sparql_models_api.create_graph(
            uris, MINIO_ENDPOINT, model.bucket_name, model.name, model.id, description
        )

        sparql_util.store_graph(graph, "model-" + str(model.id))

        shape_information = sparql_util.get_shapes()
        report = sparql_util.shacl_validate("model-" + str(model.id), shape_information)

        if not report["sh:conforms"]:
            raise HTTPException(
                status_code=400,
                detail="Data does not match required data for triple storage.",
            )

        sparql_util.store_graph(graph)
        sparql_util.delete_graph("model-" + str(model.id))

        return fuseki_id
    except Exception as ex:
        logger.error(ex)
        delete_postgres_entry(model.id, db)
        raise HTTPException(
            status_code=500,
            detail="Creation of Metdata for Model failed. Please try again.",
        )


def upload_model_to_minio(
    db: Session,
    token: str,
    model: Model,
    filename: str,
    model_file: Union[io.BytesIO, bytes],
    fuseki_id: Union[str, None],
):
    try:
        _validate_parameters(model.bucket_name, token)
        model_prefix = f"models/{model.id}"

        minio_api.upload_data(
            model.bucket_name,
            prefix=model_prefix,
            token=token,
            data=model_file,
            objectname=filename,
        )

        return model_prefix
    except Exception as ex:
        logger.exception(ex)
        delete_postgres_entry(model.id, db)
        delete_fuseki_entry(fuseki_id, model.id)
        raise HTTPException(
            status_code=500, detail="Uploading Files to Minio failed. Please try again."
        )


def save_model_to_postgres(
    db: Session, model: Model, token: str, prefix: str, fuseki_id: Union[str, None]
):
    try:
        object = minio_api.stat_object(
            bucket=model.bucket_name,
            object_name=f"{prefix}/{model.file_name}",
            token=token,
        )
        model.file_size = object.size
        model.last_modified = datetime.datetime.now()
        model.metadata_uri = fuseki_id

        return sql_api.update_model(db, model)
    except Exception as ex:
        logger.error(ex)
        delete_postgres_entry(model.id, db)
        delete_fuseki_entry(fuseki_id, model.id)
        remove_files_from_minio(model.bucket_name, f"models/{model.id}/", token)
        raise HTTPException(
            status_code=500,
            detail="Saving of Model into Database failed. Please try again.",
        )


def delete_postgres_entry(id, db):
    dataset = check_exists(sql_api.get_model(db, id))
    sql_api.delete_model(db, dataset)


def delete_fuseki_entry(metadata_uri: Union[str, None], id: int):
    if metadata_uri is not None:
        sparql_models_api.delete_model(metadata_uri)
        sparql_util.delete_graph("model-" + str(id))


def remove_files_from_minio(bucket_name: str, model_prefix: str, token):
    minio_api.delete_all_objects(bucket_name, prefix=model_prefix, token=token)


@router.patch("/{model_id}", response_model=Model)
def update_model(
    request: Request,
    model_id: int,
    model_patch: ModelPatch,
    db: Session = Depends(get_db),
):
    model: models.Model = check_exists(sql_api.get_model(db, model_id))

    model.name = model_patch.name
    model.format = model_patch.format
    model.input_name = model_patch.input_name
    model.input_datatype = model_patch.input_datatype
    model.input_shape = model_patch.input_shape
    model.input_semantics = model_patch.input_semantics
    model.output_name = model_patch.output_name
    model.output_datatype = model_patch.output_datatype
    model.output_shape = model_patch.output_shape
    model.output_labels = model_patch.output_labels

    sql_api.update_model(db, model)

    return Response(status_code=204)


@router.patch("/{model_id}/toggle-public")
async def update_dataset(
    request: Request, model_id: int, db: Session = Depends(get_db)
):
    model = check_exists(sql_api.get_model(db, model_id))
    model.public = not model.public

    if model.public:
        _create_zip(model, request.user.minio_token)
        _create_catalog_entry(model)
    else:
        _remove_zip(model, request.user.minio_token)
        _remove_catalog_entry(model)

    return sql_api.update_model(db, model)


def _create_zip(model: Model, token):
    downloaded_files = {}
    for item in minio_api.get_all_objects(
        model.bucket_name, prefix=f"models/{model.id}/", token=token
    ):
        if item.is_dir is False:
            downloaded_files[item.object_name] = minio_api.download_file(
                model.bucket_name, token, item
            ).read()
    zip = common.create_zip_file(downloaded_files)
    minio_api.upload_data(
        bucket=model.bucket_name,
        prefix=f"models/{model.id}/edc",
        token=token,
        data=zip.getvalue(),
        objectname=f"{model.name}.zip",
    )


def _create_catalog_entry(model: Model):
    labels = sparql_util.get_labels_for_uri(model.metadata_uri)
    description = sparql_util.get_description_for_uri(model.metadata_uri)
    metadata = sparql_util.get_metadata_information_for_uri(model.metadata_uri)
    create_catalog_entry_model(model, labels, description, metadata)


def _remove_zip(model: Model, token):
    minio_api.delete_object(
        bucket=model.bucket_name,
        object_name=f"models/{model.id}/edc/{model.name}.zip",
        token=token,
    )


def _remove_catalog_entry(model: Model):
    delete_catalog_entry_model(model)


# Deletes dataset from bucket on minio and all information in Fuseki
#
# dataset_name:  the dataset name passed as a string
# bucket_name:   the bucket name passed as a string
@router.delete("/{model_id}")
def delete_model(request: Request, model_id: int, db: Session = Depends(get_db)):
    user: KeycloakUser = request.user
    model = check_exists(sql_api.get_model(db, model_id))

    if user.username != model.owner:
        raise HTTPException(
            status_code=403, detail="Only the Owner can delete Datasets."
        )

    # check if this model is used in any container built on the platform
    # if yes, don't delete and return an error
    containers = container_sql_api.get_container_images_for_model(db, model)

    if len(containers):
        msg = "The Model is used in a Container. Can not delete!"
        return Response(msg, status_code=409)

    bucket_name = model.bucket_name
    token = user.minio_token
    prefix = f"models/{model.id}/"

    _validate_parameters(bucket_name, token)
    if model.metadata_uri is not None:
        sparql_models_api.delete_model(model.metadata_uri)

    minio_api.delete_all_objects(bucket_name, prefix, token)
    minio_api.delete_all_objects("triton", f"{model.name}/", token)
    sql_api.delete_model(db, model)
    sparql_util.delete_graph("model-" + str(model_id))

    return Response(status_code=204)


def _validate_parameters(bucket, token):
    try:
        minio_api.valid_params(bucket, token)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
