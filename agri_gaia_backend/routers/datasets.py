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

import copy
import datetime
import sys
import json
import logging
import os
import tempfile
import fiftyone as fo
import fiftyone.utils.labels as foul
import fiftyone.utils.cvat as cvat
import docker
import xmltodict
import inspect
import zipfile
import subprocess
from functools import reduce
from typing import Dict, List, Optional, Union
from pathlib import Path
import io
from PIL import Image

from agri_gaia_backend.db import dataset_api as sql_api
from agri_gaia_backend.routers import common
from agri_gaia_backend.routers.agrovoc import check_keyword
from agri_gaia_backend.routers.common import (
    check_exists,
    get_db,
    create_zip_file_response,
    create_single_file_response,
    extract_zip,
)
from agri_gaia_backend.schemas.cvat import CvatAuthDataSchema
from agri_gaia_backend.schemas.dataset import Dataset
from agri_gaia_backend.schemas.keycloak_user import KeycloakUser
from agri_gaia_backend.services import minio_api
from agri_gaia_backend.services.cvat.cvat_api import get_task_annotations, remove_task
from agri_gaia_backend.services.edc.connector import (
    create_catalog_entry_dataset,
    delete_catalog_entry_dataset,
    get_catalouge_information,
)
from agri_gaia_backend.services.graph.sparql_operations import (
    datasets as sparql_datasets_api,
)
from agri_gaia_backend.services.graph.sparql_operations import util as sparql_util
from agri_gaia_backend.services.minio_api import MINIO_ENDPOINT
from agri_gaia_backend.util.common import get_stacktrace, rm, mkdir, gpu_available
from agri_gaia_backend.util.datasets import (
    validate_name,
    is_cvat_annotation_xml,
    validate_dataresource_configuration_files,
)
from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.datastructures import UploadFile
from fastapi.param_functions import File
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from agri_gaia_backend.util.env import NUCLIO_CVAT_PROJECT_NAME
from agri_gaia_backend.routers.common import TaskCreator, get_task_creator

ROOT_PATH = "/datasets"

logger = logging.getLogger("api-logger")
router = APIRouter(prefix=ROOT_PATH)


@router.on_event("startup")
async def startup():
    """
    Only needed as long as EDC has no persistent catalogue storage.
    Initially fills the catalogue with previous published datasets.
    """
    logger.info("Creating Dataset EDC Catalog Entries...")
    # Does not work with Depends() in function call
    db = next(get_db())
    datasets = sql_api.get_published_datasets(db, 0, 1000)
    logger.info("Received Datasets from Database...")
    _create_catalog_entries(datasets)
    logger.info("Creating Dataset EDC Catalog Entries Done!")
    minio_client = minio_api.get_admin_client()
    if minio_client.bucket_exists("triton"):
        logger.info("Triton bucket already exists")
    else:
        minio_client.make_bucket("triton")
        logger.info("triton bucket created")


@router.get("", response_model=List[Dataset])
def get_all_datasets(skip: int = 0, limit: int = 10000, db: Session = Depends(get_db)):
    """
    Fetches all Datasets from the postgres database

    Args:
        skip: How many dataset entries shall be skipped. Defaults to 0.
        limit: What is the maximum number of datasets to be fetched? Defaults to 100.
        db: Database Session. Created automatically.

    Returns:
        A list of all datasets, which are stored by the plattform.
    """
    return sql_api.get_datasets(db, skip=skip, limit=limit)


@router.get("/keyword")
def get_all_datasets_for_keyword(
    uri: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    """
    Fetches datasets matching the given keyword.

    Searches for dataset URLs in the Apache Fuseki, which are annotated with the given keyword.
    Afterwards the Postgres database is queried for all datasets, which are represented by the URLs, returned from the Fuseki Storage.

    Args:
        uri: The URI of an Agrovoc Concept.
        skip: How many dataset entries shall be skipped. Defaults to 0.
        limit: What is the maximum number of datasets to be fetched? defaults to 100.
        db: Database Session. Created automatically.

    Returns:
        A list of all datasets, which are stored by the plattform and are annotated with the given keyword.
    """
    possible_concepts = sparql_util.query_narrower_concepts(uri)
    uris = sparql_datasets_api.query_for_concepts(possible_concepts)

    datasets = sql_api.get_datasets_by_metadata_uri(
        db, skip=skip, limit=limit, uris=uris
    )

    return datasets


@router.get("/catalogue")
async def get_catalogue(request: Request):
    """
    Returns assets, policies, contractdefinitions and contractnegotiations stored in the EDC

    Return:
        The queried information in a json format.
    """
    return get_catalouge_information()


@router.get("/{dataset_id}", response_model=Dataset)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Fetches the dataset matching the given ID.

    Args:
        dataset_id: The ID of the searched dataset
        db: Database Session. Created automatically.

    Returns:
        One Dataset object that is identified by the given ID.

    Raises:
        HTTPException: No dataset is found for the given ID.
    """
    return check_exists(sql_api.get_dataset(db, dataset_id))


@router.get("/{dataset_id}/download")
def download_dataset(
    request: Request,
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """
    Downloads the Dataset which is identified by the given ID.

    Searches the Dataset for the given ID in the Postgres database.
    Afterwards all files stored in the folder of the found dataset will be downloaded from MinIO.
    Those files are converted to a Zip file, which is encapsulated in a response with the required header information.

    Args:
        request: The request object containing information on the user.
            (like authentication token, his bucket name in MinIO, ...)
        dataset_id: The ID of the searched dataset
        db: Database Session. Created automatically.

    Returns:
        A response containing all files found in the MinIO storage for the dataset with the given ID as a Zip file.

    Raises:
        HTTPException: No dataset is found for the given ID.
    """
    user: KeycloakUser = request.user
    dataset = check_exists(sql_api.get_dataset(db, dataset_id))

    bucket_name = dataset.bucket_name
    token = user.minio_token
    dataset_prefix = f"datasets/{dataset.id}"

    _validate_parameters(bucket_name, token)

    if dataset.annotation_task_id is not None:
        update_annotations_from_cvat(dataset, token)

    downloaded_files = {}
    for item in minio_api.get_all_objects(
        bucket_name, prefix=dataset_prefix, token=token
    ):
        if item.is_dir is False:
            downloaded_files[item.object_name] = minio_api.download_file(
                bucket_name, token, item
            ).read()
    return common.create_zip_file_response(
        downloaded_files, filename=f"{dataset.name}.zip"
    )


@router.post("/import", response_model=Dataset, status_code=status.HTTP_201_CREATED)
def create_dataset(
    request: Request,
    db: Session = Depends(get_db),
    files: Optional[List[UploadFile]] = File(None),
):
    """
    Creates a dataset instance.

    Initially creates an entry in the Postgres database to retreive the autogenerated ID of the dataset.
    Afterwards the metadata will be uploaded to the Fuseki Triple Storage.
    After putting the metadata into the triple storage, all files passed to this function will be uploaded to MinIO.
    Last the database entry in the Postgres database will be updated by information like filesize, the graph URI, etc.

    Args:
        request: The request object containing information on the user.
            (like authentication token, his bucket name in MinIO, ...)
        db: Database Session. Created automatically.
        files: A list of files, which shall be uploaded to the MinIO storage. First entry must be a zip archive containing all relevant files.

    Returns:
        The created dataset instance saved in the Postgres database.

    Raises:
        HTTPException: If one of the steps failes.
    """

    zip_content = extract_zip(files[0].file._file)
    return _import_dataset_from_zip(zip_content, db, request)


def _import_dataset_from_zip(zip_content, db, request):
    if "metadata.json" not in zip_content:
        raise HTTPException(
            status_code=500,
            detail="Zip has to contain json ld conform Metadata in a metadata.json file.",
        )

    meta = json.loads(zip_content["metadata.json"])

    dataset = sql_api.create_dataset(
        db,
        name=meta["dcat:title"],
        owner=request.user.username,
        filecount=0,
        total_filesize=0,
        last_modified=datetime.datetime.now(),
        bucket_name=request.user.minio_bucket_name,
        dataset_type=meta["@type"].split(":")[1],
        annotation_labels=None,
    )

    sparql_util.store_json(zip_content["metadata.json"])

    for name in zip_content:
        minio_api.upload_data(
            dataset.bucket_name,
            prefix=f"datasets/{dataset.id}",
            token=request.user.minio_token,
            data=zip_content[name],
            objectname=name,
        )

    _save_dataset_to_postgres(
        dataset=dataset,
        token=request.user.minio_token,
        minio_location=f"datasets/{dataset.id}",
        db=db,
        fuseki_id=meta["@id"],
    )

    return dataset


@router.post("", response_model=Dataset, status_code=status.HTTP_201_CREATED)
def create_dataset(
    request: Request,
    name: str = Form(...),
    db: Session = Depends(get_db),
    files: Optional[List[UploadFile]] = File(None),
    filenames: Optional[List[str]] = Form(None),
    semantic_labels: Optional[List[str]] = Form(None),
    locations: Optional[List[str]] = Form(None),
    metadata: Optional[str] = Form(None),
    dataset_type: Optional[str] = Form(None),
    annotation_labels: Optional[List[str]] = Form(None),
    description: str = Form(...),
    includes_annotation_file: bool = Form(...),
    is_classification_dataset: bool = Form(...),
):
    """
    Creates a dataset instance.

    Initially creates an entry in the Postgres database to retreive the autogenerated ID of the dataset.
    Afterwards the metadata will be uploaded to the Fuseki Triple Storage.
    After putting the metadata into the triple storage, all files passed to this function will be uploaded to MinIO.
    Last the database entry in the Postgres database will be updated by information in filesize, ...

    Args:
        request: The request object containing information on the user.
            (like authentication token, his bucket name in MinIO, ...)
        name: The name of the dataset to be created.
        db: Database Session. Created automatically.
        files: A list of files, which shall be uploaded to the MinIO storage.
        filenames: TODO
        semantic_labels: A list of (semantic) label URIs, which are used to annotate the dataset with keywords.
        locations: A list of (semantic) label URIs, which are used to annotate the dataset with a location.
        metadata: All optional metadata on the dataset given as a dict object.
        dataset_type: The Dataset type of the created Dataset (for example AgriImageDataResource).
        annotation_labels: A list of user defined labels, which are used for dataset annotation.
        description: A short description of the dataset and what to find inside.
        includes_annotation_file: Flag indicating the last element of files is an annotation file.
        is_classification_dataset: Flag, if the given datasets is a classification dataset.

    Returns:
        The created dataset instance saved in the Postgres database.

    Raises:
        HTTPException: If one of the steps failes.
    """
    validate_name(name)
    user: KeycloakUser = request.user

    logger.debug(f"[create_dataset] dataset metadata: {metadata}")
    logger.debug(f"[create_dataset] dataset type: {dataset_type}")

    created_dataset = _create_initial_entry_postgres(
        db=db,
        user=user,
        dataset_name=name,
        annotation_labels=annotation_labels,
        dataset_type=dataset_type,
    )

    fuseki_id = _save_dataset_metadata_to_fuseki(
        dataset=created_dataset,
        labels=semantic_labels,
        description=description,
        db=db,
        metadata=metadata,
        locations=locations,
        dataset_type=dataset_type,
        files=files,
        annotation_labels=annotation_labels,
    )

    dataset_prefix, labels = _upload_dataset_to_minio(
        user=user,
        dataset=created_dataset,
        token=user.minio_token,
        files=files,
        filenames=filenames,
        includes_annotation_file=includes_annotation_file,
        is_classification_dataset=is_classification_dataset,
        dataset_type=dataset_type,
        fuseki_id=fuseki_id,
        db=db,
    )

    if is_classification_dataset:
        created_dataset.annotation_labels = labels

    dataset = _save_dataset_to_postgres(
        dataset=created_dataset,
        token=user.minio_token,
        minio_location=dataset_prefix,
        db=db,
        fuseki_id=fuseki_id,
    )

    return dataset


def _create_initial_entry_postgres(
    db: Session,
    user: KeycloakUser,
    dataset_name: str,
    annotation_labels: List[str],
    dataset_type: str,
):
    """
    Created an initial entry for the dataset in the Postgres database.

    Args:
        db: Database Session.
        user: Information on the user, who wants to create the dataset.
        dataset_name: The name of the dataset.
        annotation_labels: Labels used for dataset annotation.
        dataset_type: The Dataset type of the created Dataset (for example AgriImageDataResource).

    Returns:
        The instance of the created dataset.

    Raises:
        HTTPException: If there are problems during creation of the dataset.
    """
    try:
        # Replace dataset name by name(2), (3)... if name already exist
        if sql_api.get_dataset_by_name(db=db, name=dataset_name):
            count = 2
            while sql_api.get_dataset_by_name(db=db, name=f"{dataset_name}({count})"):
                count += 1
            dataset_name = f"{dataset_name}({count})"

        created_dataset = sql_api.create_dataset(
            db,
            name=dataset_name,
            owner=user.username,
            filecount=0,
            total_filesize=0,
            last_modified=datetime.datetime.now(),
            bucket_name=user.minio_bucket_name,
            annotation_labels=annotation_labels,
            dataset_type=dataset_type,
        )

        return created_dataset
    except Exception as e:
        logger.error(
            "Saving into Postgres failed. Stacktrace:\n" + get_stacktrace(e))
        raise HTTPException(
            status_code=500,
            detail="Initial Creation of Dataset failed. Please try again.",
        )


def _save_dataset_metadata_to_fuseki(
    dataset: Dataset,
    labels: Optional[List[str]],
    description: str,
    db: Session,
    metadata: str,
    locations: List[str],
    dataset_type: str,
    files,
    annotation_labels: Optional[List[str]],
):
    """
    Saves dataset metadata to the Fuseki Triple storage.

    First this function creates a temporary Fuseki dataset.
    Then the given metadata will be used, to build a graph representation.
    This representation is sent to the temporary dataset and will be validated using the shacl shapes stored in the shapes dataset.
    If the validation does not fail, tha graph will be also saved in the Fuseki dataset ds.
    After the uploading of the metadata the temporary graph gets deleted.

    Args:
        dataset: The dataset instance, that was put into Postgres.
        labels: A list of labels, which are used to describe the dataset.
        description: The description of the dataset.
        db: Database Session. Only used to delete the Postgres entry, if the uploading to Fuseki fails.
        metadata: All optional metadata on the dataset given as a dict object.
        locations: A list of (semantic) label URIs, which are used to annotate the dataset with a location.
        dataset_type: The Dataset type of the created Dataset (for example AgriImageDataResource).
        config_files: the configuration files, needed to extract metadata for the dataressource type.

    Returns:
        The URL used as an ID in the Fuseki Storage.

    Raises:
        HTTPException: If uploading metadata to Fuseki failes.
    """
    try:
        config_files = validate_dataresource_configuration_files(
            dataset_type, files)
        temporary_fuseki_dataset = "dataset-" + str(dataset.id)
        if labels is None:
            labels = []
        if annotation_labels is not None:
            for annotation_label in annotation_labels:
                result = check_keyword(annotation_label)
                if result["concept"] is not None and result["concept"] not in labels:
                    labels.append(result["concept"])
                    annotation_labels.remove(annotation_label)

        sparql_util.createFusekiDataset(temporary_fuseki_dataset)
        label_uris = []
        if labels:
            for label in labels:
                label_uris.append(sparql_util.convert_to_URI(label))

        location_uris = []
        if locations is not None:
            for location in locations:
                location_uris.append(sparql_util.convert_to_URI(location))

        graph, fuseki_id = sparql_datasets_api.create_graph(
            label_uris=label_uris,
            location_uris=location_uris,
            minio_server=MINIO_ENDPOINT,
            bucket=dataset.bucket_name,
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            description=description,
            metadata=metadata,
            dataset_type=dataset_type,
            config_files=config_files,
            annotation_labels=annotation_labels,
        )

        sparql_util.store_graph(graph, temporary_fuseki_dataset)

        shape_information = sparql_util.get_shapes()
        report = sparql_util.shacl_validate(
            temporary_fuseki_dataset, shape_information)

        if not report["sh:conforms"]:
            raise HTTPException(
                status_code=400,
                detail="Data does not match required data for triple storage.",
            )

        sparql_util.store_graph(graph)
        sparql_util.delete_graph(temporary_fuseki_dataset)

        return fuseki_id
    except Exception as e:
        logger.error(
            "Uploading Metadata to Fuseki failed. Stacktrace:\n" +
            get_stacktrace(e)
        )
        _delete_postgres_entry(dataset.id, db)
        raise HTTPException(
            status_code=500,
            detail="Creation of Metdata for Dataset failed. Please try again.",
        )


def _upload_dataset_to_minio(
    user: KeycloakUser,
    token: str,
    dataset: Dataset,
    files: List[UploadFile],
    filenames: List[str],
    db: Session,
    fuseki_id: str,
    includes_annotation_file: bool,
    is_classification_dataset: bool,
    dataset_type: str,
):
    """
    Uploads files to MinIO.

    Args:
        user: Information on the user, who wants to create the dataset.
        token: The authentication token of the uploading user.
        dataset: The created dataset in the Postgres database.
        files: The files to be uploaded.
        filenames: TODO
        db: Database Session. Only used to delete the Postgres entry, if the uploading to Fuseki fails.
        fuseki_id: ID of dataset in Fuseki storage. Only used to delete from Fuseki, if Uploading of the files fails.
        includes_annotation_file: Flag indicating the last element of files is an annotation file.
        is_classification_dataset: Flag, if the given datasets is a classification dataset.

    Returns:
        The directory, where files are uploaded into MinIO.

    Raises:
        HTTPException: If uploading of the files fails.
    """
    try:
        _validate_parameters(dataset.bucket_name, token)
        dataset_prefix = f"datasets/{dataset.id}"
        labels = []

        if includes_annotation_file:
            annotation_file = files[-1]

            if is_cvat_annotation_xml(annotation_file):
                annotation_file.filename = "annotations.xml"

            print("Annotation File:", annotation_file)

            minio_api.upload_file(
                dataset.bucket_name,
                prefix=dataset_prefix + "/annotations",
                token=token,
                file=annotation_file,
            )
            del files[-1]
        if is_classification_dataset and dataset_type is "AgriImageDataResource":
            iaw = cvat.CVATImageAnnotationWriter()
            images: List[cvat.CVATImage] = []
            id = 0

            zip = zipfile.ZipFile(file=files[0].file._file)
            for name in zip.namelist():
                if not zip.getinfo(name).is_dir():
                    with zip.open(name) as file:
                        tag = cvat.CVATImageTag(name.split("/")[1])
                        image = Image.open(file)
                        images.append(
                            cvat.CVATImage(
                                id=id,
                                name=f"{dataset.owner}/{dataset.id}/{name}",
                                width=image.width,
                                height=image.height,
                                tags=[tag],
                            )
                        )
                        file.seek(0)
                        minio_api.upload_data(
                            dataset.bucket_name,
                            prefix=dataset_prefix,
                            token=token,
                            data=file,
                            objectname=name,
                        )
            taskLabels = cvat.CVATTaskLabels.from_cvat_images(images)
            for label in taskLabels.labels:
                labels.append(label["name"])
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = os.path.join(tmp_dir, f"{dataset.id}.xml")
                iaw.write(taskLabels, images, path)
                xml = open(path, "rb")
                bio = io.BytesIO(xml.read())
                minio_api.upload_data(
                    dataset.bucket_name,
                    prefix=dataset_prefix + "/annotations",
                    token=token,
                    data=bio,
                    objectname="annotations.xml",
                )
        elif filenames != None:
            logger.info(filenames)
            for filename in filenames:
                data = minio_api.get_object(
                    user.minio_bucket_name,
                    object_name=filename,
                    token=user.minio_token,
                ).data
                minio_api.upload_data(
                    dataset.bucket_name,
                    prefix=dataset_prefix,
                    token=token,
                    data=data,
                    objectname="/".join(filename.split("/")[-2:]),
                )
        elif files != None:
            for file in files:
                minio_api.upload_file(
                    dataset.bucket_name, prefix=dataset_prefix, token=token, file=file
                )

        return dataset_prefix, labels
    except Exception as e:
        logger.error(
            "Uploading Files to Minio failed. Stacktrace:\n" +
            get_stacktrace(e)
        )
        _delete_postgres_entry(dataset.id, db)
        _delete_fuseki_entry(fuseki_id, dataset.id)
        raise HTTPException(
            status_code=500, detail="Uploading Files to Minio failed. Please try again."
        )


def _save_dataset_to_postgres(
    dataset: Dataset, token: str, minio_location: str, db: Session, fuseki_id: str
):
    """
    Updates the data saved in the Postgres database.

    Args:
        dataset: The already saved instance in the database.
        token: The authentication token of the uploading user.
        minio_location: The dataset location in MinIO.
        db: Database Session. Only used to delete the Postgres entry, if the uploading to Fuseki fails.
        fuseki_id: ID of dataset in Fuseki storage. Only used to delete from Fuseki, if Uploading of the files fails.

    Returns:
        The instance of the created dataset.

    Raises:
        HTTPException: If updating of the dataset failed.
    """
    try:
        dataset.last_modified = datetime.datetime.now()
        all_dataset_objects = minio_api.get_all_objects(
            dataset.bucket_name, minio_location, token
        )
        dataset.filecount = len(all_dataset_objects)
        dataset.total_filesize = _get_dataset_filesize(all_dataset_objects)
        dataset.metadata_uri = fuseki_id
        dataset.minio_location = minio_location

        return sql_api.update_dataset(db, dataset)
    except Exception as ex:
        logger.error(
            "Saving into Postgres failed. Stacktrace:\n" + get_stacktrace(ex))
        _delete_postgres_entry(dataset.id, db)
        _delete_fuseki_entry(fuseki_id, dataset.id)
        _remove_files_from_minio(
            dataset.bucket_name, f"datasets/{dataset.id}/", token)
        raise HTTPException(
            status_code=500,
            detail="Saving of Dataset into Database failed. Please try again.",
        )


@router.delete("/{dataset_id}")
def delete_dataset(
    request: Request,
    cvat_auth: CvatAuthDataSchema,
    dataset_id: int,
    annotation_task_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """
    Deletes a dataset and everything derived from it.

    The steps run one after another, each with its own error handling: database
    entry, metadata and graph in Fuseki, files in the CVAT server container, CVAT
    task and finally the files in MinIO. A failing cleanup step no longer aborts the
    remaining steps. If a cleanup step fails after the database entry is gone, the
    response is a named partial success (HTTP 207) instead of a server error.

    Deleting a dataset whose database entry is already gone is idempotent: the
    remaining files in the caller's bucket and, if ``annotation_task_id`` is given,
    the caller's CVAT task are cleaned up. That lets a caller finish a delete that
    failed during cleanup before the fix.

    Args:
        request: The request object containing information on the user.
            (like authentication token, his bucket name in MinIO, ...)
        cvat_auth: CVAT authentication data (sessionid, key, csrftoken). Used as a
            fallback when the platform's CVAT superuser is not configured.
        dataset_id: The ID of the dataset to be deleted.
        annotation_task_id: CVAT task to clean up when the database entry is already
            gone. Ignored for an existing dataset, which keeps its stored task ID.
        db: Database Session. Created automatically.

    Returns:
        204 if everything was removed. 207 with a named list of failed cleanup steps
        when the dataset entry was deleted but a cleanup step failed.
    """
    user: KeycloakUser = request.user
    dataset = sql_api.get_dataset(db, dataset_id)

    if dataset is None:
        return _remove_dataset_leftovers(
            user, dataset_id, annotation_task_id, dict(cvat_auth)
        )

    if user.username != dataset.owner:
        raise HTTPException(
            status_code=403, detail="Only the owner can delete datasets."
        )

    _validate_parameters(dataset.bucket_name, user.minio_token)

    try:
        sql_api.delete_dataset(db, dataset)
    except IntegrityError as e:
        logger.error(
            "Deleting the dataset entry failed. Stacktrace:\n" + get_stacktrace(e)
        )
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(
            "Deleting the dataset entry failed. Stacktrace:\n" + get_stacktrace(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Deleting the dataset entry failed. Nothing was removed.",
        )

    failed_steps = []

    if dataset.metadata_uri is not None:
        try:
            response = sparql_datasets_api.delete_dataset(dataset.metadata_uri)
            response.raise_for_status()
        except Exception as e:
            logger.error(
                "Deleting the dataset metadata from Fuseki failed. Stacktrace:\n"
                + get_stacktrace(e)
            )
            failed_steps.append(_failed_step("fuseki_metadata", e))

    try:
        response = sparql_util.delete_graph(f"dataset-{dataset_id}")
        # The graph dataset-<id> exists only temporarily: the upload creates it for
        # the SHACL validation and deletes it again afterwards. A 404 therefore means
        # that there is nothing left to delete and counts as a successful cleanup.
        if response.status_code != 404:
            response.raise_for_status()
    except Exception as e:
        logger.error(
            "Deleting the dataset graph from Fuseki failed. Stacktrace:\n"
            + get_stacktrace(e)
        )
        failed_steps.append(_failed_step("fuseki_graph", e))

    try:
        _remove_files_from_cvat_container(dataset)
    except Exception as e:
        logger.error(
            "Removing the dataset files from the CVAT server failed. Stacktrace:\n"
            + get_stacktrace(e)
        )
        failed_steps.append(_failed_step("cvat_files", e))

    if dataset.annotation_task_id is not None:
        try:
            remove_task(
                dataset.annotation_task_id,
                dict(cvat_auth),
                owner=user.username,
            )
        except Exception as e:
            logger.error(
                "Deleting the CVAT annotation task failed. Stacktrace:\n"
                + get_stacktrace(e)
            )
            failed_steps.append(
                _failed_step("cvat_task", e, task_id=dataset.annotation_task_id)
            )

    dataset_prefix = f"datasets/{dataset.id}/"
    try:
        minio_api.delete_all_objects(
            dataset.bucket_name, prefix=dataset_prefix, token=user.minio_token
        )
    except Exception as e:
        logger.error(
            "Deleting the dataset files from MinIO failed. Stacktrace:\n"
            + get_stacktrace(e)
        )
        failed_steps.append(_failed_step("minio_objects", e))

    if failed_steps:
        return _partial_failure_response(dataset_id, failed_steps)

    return Response(status_code=204)


@router.patch("/{dataset_id}")
async def update_dataset(
    request: Request, dataset_id: int, db: Session = Depends(get_db)
):
    """
    Updates a dataset.

    Args:
        request: the request, containing the updated information.
        dataset_id: the dataset ID of the dataset to be updated.
        db: Database Session. Created automatically.

    Returns:
        The updated dataset instance.
    """
    dataset = check_exists(sql_api.get_dataset(db, dataset_id))

    updated_dataset_values = await request.json()

    for attribute, value in updated_dataset_values.items():
        setattr(dataset, attribute, value)

    return sql_api.update_dataset(db, dataset)


@router.patch("/{dataset_id}/toggle-public")
def update_dataset_public(
    request: Request, dataset_id: int, db: Session = Depends(get_db)
):
    """
    Toggles the visibility of a dataset in the EDC catalogue.

    Args:
        request: the request, containing the updated information.
        dataset_id: the dataset ID of the dataset to be updated.
        db: Database Session. Created automatically.

    Returns:
        The updated dataset instance.
    """
    dataset = check_exists(sql_api.get_dataset(db, dataset_id))
    dataset.public = not dataset.public

    if dataset.public:
        _create_zip(dataset, request.user.minio_token)
        _create_catalog_entry(dataset)
    else:
        _remove_zip(dataset, request.user.minio_token)
        _remove_catalog_entry(dataset)

    return sql_api.update_dataset(db, dataset)


@router.post("/{dataset_id}/annotate")
def annotate_dataset(request: Request, dataset_id: int, db: Session = Depends(get_db)):
    """
    TODO
    """
    dataset = check_exists(sql_api.get_dataset(db, dataset_id))

    cvat_server = _get_docker_container_fuzzy("cvat_server")
    if cvat_server is None:
        raise HTTPException(
            status_code=500, detail="CVAT server is not running.")

    mount_bucket_cmd = f"/bin/bash -c '$HOME/mount-minio-bucket.sh {MINIO_ENDPOINT} {dataset.bucket_name} {dataset.id}'"
    print(f"Executing: docker exec -it {cvat_server.name}", mount_bucket_cmd)
    cvat_server.exec_run(mount_bucket_cmd)

    return dataset


@router.get("/convert/formats")
def get_conversion_label_formats() -> List[str]:
    """
    TODO
    """
    supported_conversions = {
        "COCODetectionDataset",
        "CVATImageDataset",
        "YOLOv5Dataset",
    }
    return [
        elem[0]
        for elem in inspect.getmembers(sys.modules["fiftyone.types"], inspect.isclass)
        if elem[0] in supported_conversions
    ]


@router.post("/convert/{input_format}/{output_format}")
def convert_labels(
    input_format: str,
    output_format: str,
    label_file: UploadFile = File(...),
) -> FileResponse:
    """
    TODO
    """

    def import_labels(
        tmp_dir: str, label_file: UploadFile, input_type: fo.types
    ) -> fo.Dataset:
        _, ext = os.path.splitext(label_file.filename)

        if input_type == fo.types.YOLOv5Dataset and ext != ".zip":
            raise RuntimeError(
                "YOLOv5 labels have to be uploaded with images as a ZIP archive!"
            )

        input_dir = os.path.join(tmp_dir, "input")
        mkdir(input_dir)

        labels_path = os.path.join(input_dir, label_file.filename)

        with open(labels_path, "wb") as fh:
            label_data = label_file.file.read()
            fh.write(label_data)

        if ext == ".zip":
            with zipfile.ZipFile(labels_path, "r") as zip:
                zip.extractall(input_dir)
                rm(labels_path)
            labels_path = input_dir
        else:
            labels = label_data.decode("utf-8")

        from_dir_args = {}
        if input_type == fo.types.YOLOv5Dataset:
            dataset_filepaths = list(
                filter(lambda p: p.is_file(), Path(labels_path).rglob("*.*"))
            )

            if not len(
                list(filter(lambda p: p.parent.name == "images", dataset_filepaths))
            ):
                raise RuntimeError(
                    "In order to determine image dimensions, YOLOv5 labels have to be uploaded with samples."
                )

            dataset_yaml_filepath = list(
                filter(
                    lambda p: p.name in {"dataset.yaml", "dataset.yml"},
                    dataset_filepaths,
                )
            )
            if len(dataset_yaml_filepath) != 1:
                raise RuntimeError("No dataset.yaml found for YOLOv5 labels.")
            dataset_yaml_filepath = dataset_yaml_filepath[0]

            from_dir_args["dataset_dir"] = dataset_yaml_filepath.parent
            labels_path = None
        else:
            image_paths = None
            if input_type == fo.types.COCODetectionDataset:
                labels = json.loads(labels)
                image_paths = [image["file_name"]
                               for image in labels["images"]]
                # With COCO, only labels of type "detections" are read by default.
                from_dir_args["label_types"] = ("detections", "segmentations")
            elif input_type == fo.types.CVATImageDataset:
                labels = xmltodict.parse(labels)
                images = labels["annotations"]["image"]
                image_paths = [image["@name"] for image in images]

            data_path = {
                image_path: os.path.join(
                    tmp_dir, "data", os.path.basename(image_path))
                for image_path in image_paths
            }
            from_dir_args["data_path"] = data_path

        # See: https://docs.voxel51.com/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.from_dir
        return fo.Dataset.from_dir(
            dataset_type=input_type,
            labels_path=labels_path,
            **from_dir_args,
        )

    def transform_labels(
        labels: fo.Dataset, input_type: fo.types, output_type: fo.types
    ) -> Union[str, List[str]]:

        # imported_exported_label_fields = [
        #    field_name
        #    for field_name in labels.get_field_schema().keys()
        #    if field_name in {"detections", "segmentations", "polylines", "keypoints"}
        # ]

        exported_label_fields = None
        if output_type == fo.types.YOLOv5Dataset:
            if labels.get_field("detections") is not None:
                exported_label_fields = "detections"
        else:
            if input_type == fo.types.COCODetectionDataset:
                if labels.get_field("detections") is not None:
                    exported_label_fields = ["detections"]
                # With COCO, "segmentations" have to be manually converted to polylines.
                if labels.get_field("segmentations") is not None:
                    foul.instances_to_polylines(
                        labels, "segmentations", "polylines", tolerance=0
                    )
                    exported_label_fields.append("polylines")
            elif input_type == fo.types.CVATImageDataset:
                if labels.get_field("polylines") is not None:
                    exported_label_fields = "polylines"
                else:
                    exported_label_fields = "detections"

        return exported_label_fields

    def get_label_filename(output_type: fo.types) -> str:
        if output_type == fo.types.CVATImageDataset:
            # See: https://docs.voxel51.com/user_guide/export_datasets.html#cvatimagedataset
            label_filename = "labels.xml"
        elif output_type == fo.types.COCODetectionDataset:
            label_filename = "labels.json"
        else:
            # For an input label file, multiple label output files are generated.
            # Example: CVAT -> YOLOv5
            label_filename = ""
        return label_filename

    def export_labels(output_type, label_output_path, label_fields) -> None:
        labels.export(
            dataset_type=output_type,
            labels_path=label_output_path,
            label_field=label_fields,
            export_media=False,
        )

    try:
        input_type = eval(f"fo.types.{input_format}")
        output_type = eval(f"fo.types.{output_format}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            labels = import_labels(tmp_dir, label_file, input_type)
            label_fields = transform_labels(labels, input_type, output_type)
            label_filename = get_label_filename(output_type)

            output_dir = os.path.join(tmp_dir, "output")

            label_output_path = os.path.join(output_dir, label_filename)
            export_labels(output_type, label_output_path, label_fields)

            if label_filename:
                if os.path.isfile(label_output_path):
                    with open(label_output_path, "rb") as fh:
                        return create_single_file_response(
                            file=fh.read(),
                            filename=os.path.basename(label_output_path),
                        )
                else:
                    raise RuntimeError(
                        "Converted label file was not generated.")

            output_label_filepaths = list(Path(output_dir).rglob("*.*"))
            if output_label_filepaths:
                output_label_files = {}
                for output_label_filepath in output_label_filepaths:
                    with open(output_label_filepath, "rb") as fh:
                        output_label_files[output_label_filepath.name] = fh.read(
                        )

                return create_zip_file_response(
                    files=output_label_files,
                    filename="labels.zip",
                )
            else:
                raise RuntimeError("Converted label files were not generated.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/auto-annotation", status_code=status.HTTP_201_CREATED)
def create_auto_annotation_model(
    auto_annotation_archive: UploadFile = File(...),
    task_creator: TaskCreator = Depends(get_task_creator),
) -> Response:
    def _create_auto_annotation_model(
        on_error,
        on_progress_change,
        auto_annotation_archive,
    ) -> None:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                auto_annotation_archive_filepath = os.path.join(
                    tmp_dir, auto_annotation_archive.filename
                )
                with open(auto_annotation_archive_filepath, "wb") as fh:
                    auto_annotation_data = auto_annotation_archive.file.read()
                    fh.write(auto_annotation_data)

                with zipfile.ZipFile(auto_annotation_archive_filepath, "r") as zip:
                    zip.extractall(tmp_dir)

                nuclio_create_project_cmd = (
                    f"nuctl create project {NUCLIO_CVAT_PROJECT_NAME}"
                )

                archive_dirname, _ = os.path.splitext(
                    auto_annotation_archive.filename)
                nuclio_deploy_cmd = (
                    f"nuctl deploy --project-name={NUCLIO_CVAT_PROJECT_NAME} "
                    f"--path={archive_dirname} --platform=local "
                    "--platform-config='{\"attributes\":{\"network\":\"agri_gaia_network\"}}'"
                )

                if gpu_available():
                    nuclio_deploy_cmd += ' --resource-limit="nvidia.com/gpu=1" --triggers=\'{"myHttpTrigger": {"maxWorkers": 1}}\''

                for nuclio_cmd in (nuclio_create_project_cmd, nuclio_deploy_cmd):
                    logger.info(f"Executing '{nuclio_deploy_cmd}'.")
                    subprocess.check_output(
                        nuclio_cmd, universal_newlines=True, cwd=tmp_dir, shell=True
                    )
        except subprocess.CalledProcessError as e:
            error_msg = str(e.output)
            logger.error(error_msg)
            on_error(error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.error(error_msg)
            on_error(error_msg)
        finally:
            auto_annotation_archive.file.close()

    _, task_location_url, _ = task_creator.create_background_task(
        _create_auto_annotation_model,
        task_title=f"Auto Annotation Model Deployment: {auto_annotation_archive.filename}",
        auto_annotation_archive=copy.deepcopy(auto_annotation_archive),
    )

    headers = {"Location": task_location_url}
    return Response(status_code=status.HTTP_202_ACCEPTED, headers=headers)


def _create_zip(dataset: Dataset, token):
    downloaded_files = {}
    metadata = sparql_util.get_metadata_information_for_uri(
        dataset.metadata_uri)
    downloaded_files["metadata.json"] = json.dumps(metadata).encode("utf-8")
    for item in minio_api.get_all_objects(
        dataset.bucket_name, prefix=dataset.minio_location, token=token
    ):
        if item.is_dir is False:
            downloaded_files[item.object_name] = minio_api.download_file(
                dataset.bucket_name, token, item
            ).read()
    zip = common.create_zip_file(downloaded_files)
    minio_api.upload_data(
        bucket=dataset.bucket_name,
        prefix=f"{dataset.minio_location}/edc",
        token=token,
        data=zip.getvalue(),
        objectname=f"{dataset.name}.zip",
    )


def _remove_zip(dataset: Dataset, token):
    minio_api.delete_object(
        bucket=dataset.bucket_name,
        object_name=f"{dataset.minio_location}/edc/{dataset.name}.zip",
        token=token,
    )


def _create_catalog_entry(dataset: Dataset):
    metadata = sparql_util.get_metadata_information_for_uri(
        dataset.metadata_uri)
    return build_catalogue_entry_from_metadata(dataset, metadata)


def _create_catalog_entries(datasets: List[Dataset]):
    uris = {}

    for dataset in datasets:
        uris[dataset.metadata_uri] = dataset

    metadata = sparql_util.get_metadata_information_for_uris(uris.keys())
    graph = []

    if "@graph" in metadata:
        if not isinstance(metadata["@graph"], list):
            graph = [metadata["@graph"]]
            metadata["@graph"] = graph
        for entry in metadata["@graph"]:
            build_catalogue_entry_from_metadata(uris[entry["@id"]], entry)


def build_catalogue_entry_from_metadata(dataset, metadata):
    labels = []
    if "dcat:keyword" in metadata:
        if not isinstance(metadata["dcat:keyword"], list):
            keywords = [metadata["dcat:keyword"]]
            metadata["dcat:keyword"] = keywords
        for label in metadata["dcat:keyword"]:
            if "@id" in label:
                labels.append(label["@id"])

    locations = []
    if "dct:spatial" in metadata:
        locations.append(metadata["dct:spatial"]["@id"])

    description = metadata["dcat:description"]

    return create_catalog_entry_dataset(
        dataset, labels, description, metadata, locations
    )


def _remove_catalog_entry(dataset: Dataset):
    delete_catalog_entry_dataset(dataset)


# TODO: Move this to docker_api (for some reason, listing all containers does not work using docker_api)
def _get_docker_container_fuzzy(fuzzy_name: str):
    client = docker.from_env()
    containers = list(
        filter(lambda c: fuzzy_name in c.name, client.containers.list()))
    if len(containers) == 1:
        return containers[0]


def _get_dataset_filesize(dataset_objects):
    return reduce(lambda x, y: x + y, map(lambda o: o.size, dataset_objects))


def _validate_parameters(bucket, token):
    try:
        minio_api.valid_params(bucket, token)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _delete_postgres_entry(id, db):
    dataset = check_exists(sql_api.get_dataset(db, id))
    sql_api.delete_dataset(db, dataset)


def _delete_fuseki_entry(metadata_uri: str, id: int):
    sparql_datasets_api.delete_dataset(metadata_uri)
    sparql_util.delete_graph("dataset-" + str(id))


def _remove_files_from_minio(bucket_name: str, dataset_prefix: str, token):
    minio_api.delete_all_objects(
        bucket_name, prefix=dataset_prefix, token=token)


def _remove_files_from_cvat_container(dataset: Dataset) -> None:
    """Remove the dataset's files from the shared volume of the CVAT server container."""
    cvat_server = _get_docker_container_fuzzy("cvat_server")
    if cvat_server is not None:
        cvat_server.exec_run(
            f"/bin/bash -c 'rm $HOME/share/{dataset.bucket_name}/{dataset.id}'"
        )


_CLEANUP_STEP_DETAILS = {
    "fuseki_metadata": "Deleting the dataset metadata from the triple store failed.",
    "fuseki_graph": "Deleting the dataset graph from the triple store failed.",
    "cvat_files": "Removing the dataset files from the CVAT server container failed.",
    "cvat_task": "Deleting the CVAT annotation task failed.",
    "minio_objects": "Deleting the dataset files from MinIO failed.",
}


def _failed_step(step: str, error: Exception, **extra) -> Dict:
    """Describe a failed cleanup step.

    The exception text stays in the server log. Only the fixed description and the
    exception type reach the response, so no token, key or signed URL can leak
    through an error message (#141).
    """
    return {
        "step": step,
        "error_type": type(error).__name__,
        "detail": _CLEANUP_STEP_DETAILS[step],
        **extra,
    }


def _partial_failure_response(dataset_id: int, failed_steps: List[Dict]) -> JSONResponse:
    """Report a delete whose database entry is gone but whose cleanup was incomplete."""
    step_names = ", ".join(step["step"] for step in failed_steps)
    logger.error(
        f"Dataset {dataset_id} was deleted, but cleanup failed for: {step_names}."
    )
    return JSONResponse(
        status_code=status.HTTP_207_MULTI_STATUS,
        content={
            "status": "partial",
            "dataset_id": dataset_id,
            "deleted": True,
            "message": (
                f"The dataset entry was deleted, but cleanup failed for: {step_names}. "
                "Repeat the DELETE to finish the cleanup; pass annotation_task_id "
                "for a CVAT task that is still left over."
            ),
            "failed_steps": failed_steps,
        },
    )


def _remove_dataset_leftovers(
    user: KeycloakUser,
    dataset_id: int,
    annotation_task_id: Optional[int],
    cvat_auth: Dict,
) -> Response:
    """Clean up after a delete whose database entry is already gone.

    Runs only against the caller's own bucket and a CVAT task the caller owns, so a
    repeated DELETE cannot touch foreign data.
    """
    failed_steps = []

    if annotation_task_id is not None:
        try:
            remove_task(annotation_task_id, cvat_auth, owner=user.username)
        except Exception as e:
            logger.error(
                "Deleting the CVAT annotation task failed. Stacktrace:\n"
                + get_stacktrace(e)
            )
            failed_steps.append(
                _failed_step("cvat_task", e, task_id=annotation_task_id)
            )

    dataset_prefix = f"datasets/{dataset_id}/"
    try:
        minio_api.delete_all_objects(
            user.minio_bucket_name, prefix=dataset_prefix, token=user.minio_token
        )
    except Exception as e:
        logger.error(
            "Deleting the dataset files from MinIO failed. Stacktrace:\n"
            + get_stacktrace(e)
        )
        failed_steps.append(_failed_step("minio_objects", e))

    if failed_steps:
        return _partial_failure_response(dataset_id, failed_steps)

    return Response(status_code=204)


def update_annotations_from_cvat(dataset: Dataset, minio_token: str) -> None:
    annotations_xml = get_task_annotations(task_id=dataset.annotation_task_id)
    annotations_prefix = f"datasets/{dataset.id}/annotations"
    minio_api.upload_data(
        bucket=dataset.bucket_name,
        prefix=annotations_prefix,
        token=minio_token,
        data=bytes(annotations_xml, "utf-8"),
        objectname="annotations.xml",
        content_type="application/xml",
    )
