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

import json
import pytest
import zipfile

from io import BytesIO
from starlette.status import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_204_NO_CONTENT,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
)
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from agri_gaia_backend import schemas
from agri_gaia_backend.db import dataset_api

from minio import Minio
from types import SimpleNamespace
from typing import Dict


def get_dataset_bucket_prefix(dataset: schemas.Dataset) -> str:
    return f"datasets/{dataset.id}/"


class TestGetDataset:
    def test_get_all_datasets(
        self, authenticated_client: TestClient, test_dataset: schemas.Dataset
    ):
        response = authenticated_client.get("/datasets")
        datasets = response.json()
        assert response.status_code == HTTP_200_OK, "Error getting datasets"
        filtered_datasets = [d for d in datasets if d["id"] == test_dataset.id]
        assert (
            len(filtered_datasets) == 1
        ), "Testdataset is not in the returned datasets (or too many with the same id)"
        returned_dataset = schemas.Dataset(**filtered_datasets[0])

        assert returned_dataset == test_dataset, "Returned dataset is not testdataset"

    def test_get_single_dataset(
        self, authenticated_client: TestClient, test_dataset: schemas.Dataset
    ):
        response = authenticated_client.get(f"/datasets/{test_dataset.id}")
        dataset = schemas.Dataset(**response.json())
        assert response.status_code == HTTP_200_OK, "Error getting dataset"
        assert dataset.name == test_dataset.name, "Dataset is not the test dataset"

    def test_get_dataset_incorrect_id(
        self, authenticated_client: TestClient, db: Session
    ):
        response = authenticated_client.get(f"/datasets/-10")
        assert (
            response.status_code == HTTP_404_NOT_FOUND
        ), "Dataset shouldn't have been returned"

    # suche nach kartoffeln
    # benutzt uri zur suche http://aims.fao.org/aos/agrovoc/c_13551
    # beim erzeugen auch schon übergeben
    # agrovoc hat nw route zum schecken agrovoc/keyword/check

    def test_get_dataset_by_keyword(
        self, authenticated_client: TestClient, test_dataset: schemas.Dataset
    ):
        response = authenticated_client.get(
            "/datasets/keyword?uri=http://aims.fao.org/aos/agrovoc/c_13551"
        )
        datasets = response.json()
        assert response.status_code == HTTP_200_OK, "Error getting dataset"
        filtered_datasets = [d for d in datasets if d["id"] == test_dataset.id]
        assert (
            len(filtered_datasets) == 1
        ), "Testdataset is not in the returned datasets (or too many with the same id)"
        returned_dataset = schemas.Dataset(**filtered_datasets[0])
        assert returned_dataset == test_dataset, "Returned dataset is not testdataset"


class TestUpdateDataset:
    def test_update_dataset(
        self,
        authenticated_client: TestClient,
        db: Session,
        test_dataset: schemas.Dataset,
    ):
        data = {"name": "Updated TestData"}
        response = authenticated_client.patch(f"/datasets/{test_dataset.id}", json=data)
        dataset = schemas.Dataset(**response.json())
        assert response.status_code == HTTP_200_OK, "Error updating dataset"
        assert dataset.name == "Updated TestData", "Dataset name was not updated"

        assert (
            dataset_api.get_dataset(db, dataset.id).name == "Updated TestData"
        ), "Dataset name does not equal test dataset name"

    def test_update_dataset_incorrect_id(
        self,
        authenticated_client: TestClient,
        db: Session,
        test_dataset: schemas.Dataset,
    ):
        data = {"name": "Updated TestData"}
        response = authenticated_client.patch(f"/datasets/-10", json=data)
        assert response.status_code == HTTP_404_NOT_FOUND, "Error updating dataset"


class TestDownloadDataset:
    def test_download_dataset(
        self,
        cvat_authentication_data: Dict,
        authenticated_client: TestClient,
        test_dataset: schemas.Dataset,
    ):
        response = authenticated_client.get(
            f"/datasets/{test_dataset.id}/download?cvat_auth={json.dumps(cvat_authentication_data)}",
        )

        assert response.status_code == HTTP_200_OK, "Error downloading dataset"

        assert "This is a test file." in response.text

        assert "datasets/" + str(test_dataset.id) + "/testfile.txt" in response.text

    def test_download_dataset_incorrect_id(
        self,
        cvat_authentication_data: Dict,
        authenticated_client: TestClient,
        test_dataset: schemas.Dataset,
    ):
        response = authenticated_client.get(
            f"/datasets/-10/download?cvat_auth={json.dumps(cvat_authentication_data)}",
        )

        assert response.status_code == HTTP_404_NOT_FOUND, "Error downloading dataset"


class TestCreateDataset:
    def test_create_empty_dataset_not_allowed(
        self, request, authenticated_client: TestClient, db: Session
    ):
        data = {
            "name": "Empty-Test-Dataset",
            "description": "This is a test dataset with no files.",
            "includes_annotation_file": False,
            "is_classification_dataset": False,
            "dataset_type": "AgriImageDataRessource",
            "metadata": "{}",
        }
        num_datasets_before = len(dataset_api.get_datasets(db))
        response = authenticated_client.post("/datasets", data=data)

        def dataset_delete(response):
            if response.is_success:
                response_dataset = schemas.Dataset(**response.json())
                deletion_response = authenticated_client.delete(
                    f"/datasets/{response_dataset.id}"
                )

        request.addfinalizer(lambda: dataset_delete(response=response))

        assert not response.is_success, "Creating an empty dataset shouldn't be allowed"
        assert (
            len(dataset_api.get_datasets(db)) == num_datasets_before
        ), "Dataset was created"

    def test_create_dataset_single_file(
        self,
        request,
        cvat_authentication_data,
        authenticated_client: TestClient,
        minio_client: Minio,
        db: Session,
    ):
        data = {
            "name": "Test-Dataset",
            "description": "This is a test dataset with no files.",
            "includes_annotation_file": "False",
            "is_classification_dataset": "False",
            "dataset_type": "AgriImageDataRessource",
            "metadata": "{}",
        }

        testfile_name = "testfile.txt"
        testfile_content = "This is a test file.".encode("utf-8")
        files = {
            "files": (testfile_name, BytesIO(testfile_content)),
        }

        response = authenticated_client.post("/datasets", data=data, files=files)
        assert response.status_code == HTTP_201_CREATED, "Error creating dataset"
        assert response.headers["Content-Type"] == "application/json"

        response_dataset = schemas.Dataset(**response.json())
        dataset_bucketname = response_dataset.bucket_name
        testfile_objectname = f"datasets/{response_dataset.id}/{testfile_name}"

        def dataset_delete(response):
            if response.is_success:
                response_dataset = schemas.Dataset(**response.json())
                response = authenticated_client.request(
                    "DELETE",
                    f"/datasets/{response_dataset.id}",
                    json=cvat_authentication_data,
                )

        request.addfinalizer(lambda: dataset_delete(response=response))

        assert (
            dataset_api.get_dataset(db, response_dataset.id).name
            == response_dataset.name
        ), "Dataset name does not equal test dataset name"

        assert minio_client.stat_object(
            bucket_name=dataset_bucketname, object_name=testfile_objectname
        ).size == len(
            testfile_content
        ), "File size in minio differs from size of given file"
        assert response_dataset.total_filesize == len(
            testfile_content
        ), "Total file size of dataset differs from total file size of given files"


class TestDatasetDelete:
    # filter the CleanupError
    @pytest.mark.filterwarnings("ignore:CleanupError")
    def test_delete_dataset(
        self,
        cvat_authentication_data: Dict,
        authenticated_client: TestClient,
        test_dataset: schemas.Dataset,
        db: Session,
        minio_client: Minio,
    ):
        assert (
            dataset_api.get_dataset(db, test_dataset.id) is not None
        ), "Precondition for delete dataset test failed, dataset is not in DB"

        dataset_prefix = get_dataset_bucket_prefix(test_dataset)
        minio_objects = list(
            minio_client.list_objects(
                test_dataset.bucket_name, dataset_prefix, recursive=True
            )
        )
        assert (
            len(minio_objects) > 0
        ), "Precondition for delete dataset test failed, dataset files are not in MinIO"
        # TODO test if metadata is in fuseki

        data = {
            "cvat_auth": cvat_authentication_data,
        }

        response = authenticated_client.request(
            "DELETE", f"/datasets/{test_dataset.id}", json=cvat_authentication_data
        )

        assert response.status_code == HTTP_204_NO_CONTENT, "Error deleting dataset"

        minio_objects = list(
            minio_client.list_objects(
                test_dataset.bucket_name, dataset_prefix, recursive=True
            )
        )
        assert len(minio_objects) == 0, "Files in MinIO were not deleted"
        # TODO check that metadata in fuseki is deleted

    def test_delete_dataset_incorrect_id(
        self,
        cvat_authentication_data: Dict,
        authenticated_client: TestClient,
        test_dataset: schemas.Dataset,
        db: Session,
        minio_client: Minio,
    ):
        assert (
            dataset_api.get_dataset(db, test_dataset.id) is not None
        ), "Precondition for delete dataset test failed, dataset is not in DB"

        dataset_prefix = get_dataset_bucket_prefix(test_dataset)
        minio_objects = list(
            minio_client.list_objects(
                test_dataset.bucket_name, dataset_prefix, recursive=True
            )
        )
        assert (
            len(minio_objects) > 0
        ), "Precondition for delete dataset test failed, dataset files are not in MinIO"
        # TODO test if metadata is in fuseki

        response = authenticated_client.request(
            "DELETE", f"/datasets/-10", json=cvat_authentication_data
        )

        assert (
            response.status_code == HTTP_404_NOT_FOUND
        ), "Dataset shouldn't have been deleted"

        minio_objects = list(
            minio_client.list_objects(
                test_dataset.bucket_name, dataset_prefix, recursive=True
            )
        )
        assert len(minio_objects) == 1, "Files shouldn't have been deleted"
        # TODO check that metadata in fuseki is deleted


class TestDatasetMetadata:
    """#146: the fixture verification reads description and metadata from the triple store."""

    def test_metadata_of_own_dataset(
        self, authenticated_client: TestClient, test_dataset: schemas.Dataset
    ):
        response = authenticated_client.get(
            f"/datasets/{test_dataset.id}/metadata"
        )
        assert response.status_code == HTTP_200_OK, "Error getting dataset metadata"
        body = response.json()
        assert (
            body["description"] == "This is a test dataset with a single file."
        ), "Description differs from the one the dataset was created with"
        assert body["metadata"] == {}, "The test dataset carries no additional metadata"

    def test_metadata_of_missing_dataset(self, authenticated_client: TestClient):
        response = authenticated_client.get("/datasets/-10/metadata")
        assert (
            response.status_code == HTTP_404_NOT_FOUND
        ), "Metadata of a missing dataset shouldn't have been returned"

    def test_metadata_of_foreign_dataset(
        self,
        authenticated_client: TestClient,
        test_dataset: schemas.Dataset,
        monkeypatch,
    ):
        foreign = SimpleNamespace(
            id=test_dataset.id,
            owner="someone-else",
            metadata_uri=test_dataset.metadata_uri,
        )
        monkeypatch.setattr(
            dataset_api, "get_dataset", lambda db, dataset_id: foreign
        )
        response = authenticated_client.get(
            f"/datasets/{test_dataset.id}/metadata"
        )
        assert (
            response.status_code == HTTP_403_FORBIDDEN
        ), "A foreign dataset's metadata must not be readable"
        assert "someone-else" not in response.text
