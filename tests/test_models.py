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

from io import BytesIO
import pytest

from fastapi.testclient import TestClient

from starlette.status import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_204_NO_CONTENT,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
)

from agri_gaia_backend import schemas
from agri_gaia_backend.db import model_api

from sqlalchemy.orm import Session
from types import SimpleNamespace

from minio import Minio


def get_model_bucket_prefix(model: schemas.Model) -> str:
    return f"models/{model.id}/"


class TestCreateModel:
    def test_create_model(
        self, testclient: TestClient, minio_client: Minio, db: Session
    ):
        data = {
            "name": "Test-Model",
            "description": "This is a test model with a single file.",
            "labels": ["http://aims.fao.org/aos/agrovoc/c_13551"],
            "format": "pytorch",
            "includes_annotation_file": False,
        }
        files = {
            "modelfile": (
                "testfile.txt",
                BytesIO("This is a test file.".encode("utf-8")),
            ),
        }

        response = testclient.post("/models", data=data, files=files)
        print(response.text)
        assert response.status_code == HTTP_201_CREATED, "Error creating model"

        assert response.headers["Content-Type"] == "application/json"

        response_model = schemas.Model(**response.json())
        model_bucketname = response_model.bucket_name

        testfile_objectname = f"models/{response_model.id}/testfile.txt"

        assert (
            model_api.get_model(db, response_model.id).name == response_model.name
        ), "Dataset name does not equal test dataset name"

        assert minio_client.stat_object(
            bucket_name=model_bucketname, object_name=testfile_objectname
        ).size == len(
            "This is a test file.".encode("utf-8")
        ), "File size in minio differs from size of given file"
        assert response_model.file_size == len(
            "This is a test file.".encode("utf-8")
        ), "Total file size of model differs from total file size of given files"

    def test_create_empty_model_not_allowed(self, testclient: TestClient, db: Session):
        data = {
            "name": "Empty-Test-Model",
            "description": "This is a test model with no files.",
            "labels": ["Potato"],
            "format": "pytorch",
            "includes_annotation_file": False,
        }
        num_models_before = len(model_api.get_models(db))
        response = testclient.post("/models", data=data)

        print(response.text)
        assert not response.is_success, "Creating an empty model shouldn't be allowed"
        assert len(model_api.get_models(db)) == num_models_before, "Model was created"

    def test_create_model_with_wrong_format(self, testclient: TestClient, db: Session):
        data = {
            "name": "Test-Model",
            "description": "This is a test model with a single file.",
            "labels": ["Test"],
            "format": "wrong_one_xd",
            "includes_annotation_file": False,
        }
        files = {
            "modelfile": (
                "test_model.txt",
                BytesIO("This is a test file.".encode("utf-8")),
            ),
        }
        num_models_before = len(model_api.get_models(db))
        response = testclient.post("/models", data=data, files=files)

        print(response.text)
        assert (
            not response.is_success
        ), "Creating a model with the wrong Format should not be possible!"
        assert len(model_api.get_models(db)) == num_models_before, "Model was created"


class TestDownloadModel:
    def test_download_model(self, testclient: TestClient, test_model: schemas.Model):
        response = testclient.get(f"/models/{test_model.id}/download")

        assert response.status_code == HTTP_200_OK, "Error downloading model"

        assert response.text == "This is a test file."

    def test_download_model_incorrect_id(
        self, testclient: TestClient, test_model: schemas.Model
    ):
        response = testclient.get(f"/models/-10/download")

        assert response.status_code == HTTP_404_NOT_FOUND, "Error downloading model"


class TestUpdateModel:
    def test_update_model(
        self, testclient: TestClient, db: Session, test_model: schemas.Model
    ):

        data = {"name": "Updated TestModel"}
        response = testclient.patch(f"/models/{test_model.id}", json=data)

        assert (
            model_api.get_model(db, test_model.id).name == "Updated TestModel"
        ), "Dataset name does not equal new test dataset name"

        assert response.status_code == HTTP_204_NO_CONTENT, "Error updating model"

    def test_update_model_incorrect_id(
        self, testclient: TestClient, db: Session, test_model: schemas.Model
    ):
        data = {"name": "Updated TestData"}
        response = testclient.patch(f"/models/-10", json=data)
        assert response.status_code == HTTP_404_NOT_FOUND, "Error updating dataset"


class TestGetModel:
    def test_get_all_models(self, testclient: TestClient, test_model: schemas.Model):
        response = testclient.get("/models")
        models = response.json()
        assert response.status_code == HTTP_200_OK, "Error getting models"
        filtered_models = [d for d in models if d["id"] == test_model.id]
        assert (
            len(filtered_models) == 1
        ), "Testmodel is not in the returned models (or too many with the same id)"
        returned_model = schemas.Model(**filtered_models[0])

        assert returned_model == test_model, "Returned model is not testmodel"

    def test_get_single_model(self, testclient: TestClient, test_model: schemas.Model):
        response = testclient.get(f"/models/{test_model.id}")
        model = schemas.Model(**response.json())
        assert response.status_code == HTTP_200_OK, "Error getting mnodel"
        assert model.name == test_model.name, "Model is not the test model"

    def test_get_model_incorrect_id(self, testclient: TestClient, db: Session):
        response = testclient.get(f"/models/-10")
        assert (
            response.status_code == HTTP_404_NOT_FOUND
        ), "Model shouldn't have been returned"


class TestDatasetDelete:
    @pytest.mark.filterwarnings("ignore:CleanupError")
    def test_delete_model(
        self,
        testclient: TestClient,
        test_model: schemas.Model,
        db: Session,
        minio_client: Minio,
    ):
        assert (
            model_api.get_model(db, test_model.id) is not None
        ), "Precondition for delete model test failed, model is not in DB"

        model_prefix = get_model_bucket_prefix(test_model)
        minio_objects = list(
            minio_client.list_objects(
                test_model.bucket_name, model_prefix, recursive=True
            )
        )
        assert (
            len(minio_objects) > 0
        ), "Precondition for delete model test failed, model files are not in MinIO"
        # TODO test if metadata is in fuseki

        response = testclient.delete(f"/models/{test_model.id}")
        assert response.status_code == HTTP_204_NO_CONTENT, "Error deleting model"

        minio_objects = list(
            minio_client.list_objects(
                test_model.bucket_name, model_prefix, recursive=True
            )
        )
        assert len(minio_objects) == 0, "Files in MinIO were not deleted"

    @pytest.mark.filterwarnings("ignore:CleanupError")
    def test_delete_model_incorrect_id(
        self,
        testclient: TestClient,
        test_model: schemas.Model,
        db: Session,
        minio_client: Minio,
    ):
        assert (
            model_api.get_model(db, test_model.id) is not None
        ), "Precondition for delete model test failed, model is not in DB"

        model_prefix = get_model_bucket_prefix(test_model)
        minio_objects = list(
            minio_client.list_objects(
                test_model.bucket_name, model_prefix, recursive=True
            )
        )
        assert (
            len(minio_objects) > 0
        ), "Precondition for delete model test failed, model files are not in MinIO"
        # TODO test if metadata is in fuseki

        response = testclient.delete(f"/models/-10")
        assert response.status_code == HTTP_404_NOT_FOUND, "Model was deleted"

        minio_objects = list(
            minio_client.list_objects(
                test_model.bucket_name, model_prefix, recursive=True
            )
        )
        assert len(minio_objects) == 1, "Files in MinIO shouln't have been not deleted"
        # TODO check tt metadata in fuseki is deleted


class TestModelMetadata:
    """#146: the fixture verification reads the description from the triple store."""

    def test_metadata_of_own_model(
        self, testclient: TestClient, test_model: schemas.Model
    ):
        response = testclient.get(f"/models/{test_model.id}/metadata")
        assert response.status_code == HTTP_200_OK, "Error getting model metadata"
        body = response.json()
        assert (
            body["description"] == "This is a test model with a single file."
        ), "Description differs from the one the model was created with"
        assert body["metadata"] == {}, "Models carry no uploader metadata"

    def test_metadata_of_missing_model(self, testclient: TestClient):
        response = testclient.get("/models/-10/metadata")
        assert (
            response.status_code == HTTP_404_NOT_FOUND
        ), "Metadata of a missing model shouldn't have been returned"

    def test_metadata_of_foreign_model(
        self,
        testclient: TestClient,
        test_model: schemas.Model,
        monkeypatch,
    ):
        foreign = SimpleNamespace(
            id=test_model.id, owner="someone-else", metadata_uri=test_model.metadata_uri
        )
        monkeypatch.setattr(model_api, "get_model", lambda db, model_id: foreign)
        response = testclient.get(f"/models/{test_model.id}/metadata")
        assert (
            response.status_code == HTTP_403_FORBIDDEN
        ), "A foreign model's metadata must not be readable"
        assert "someone-else" not in response.text
