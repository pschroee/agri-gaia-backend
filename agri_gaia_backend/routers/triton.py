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

from enum import Enum
import json
import logging
from random import randbytes
import time
import traceback
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List, Optional, Union, Tuple

from agri_gaia_backend.db import model_api as sql_api
from agri_gaia_backend.db import dataset_api as dataset_sql_api
from agri_gaia_backend.db import tasks_api
from agri_gaia_backend.db.database import SessionLocal
from agri_gaia_backend.routers.common import (
    TaskCreator,
    check_exists,
    get_db,
    get_task_creator,
)
from agri_gaia_backend.schemas.keycloak_user import KeycloakUser
from agri_gaia_backend.services import minio_api
from agri_gaia_backend.services.edc.connector import (
    create_catalog_entry_model,
    delete_catalog_entry_model,
)
from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.param_functions import File
from sqlalchemy.orm import Session

import tritonclient.http as httpclient
from tritonclient.grpc import model_config_pb2
from tritonclient.utils import InferenceServerException, triton_to_np_dtype
import tritonclient.grpc.model_config_pb2 as mc
from google.protobuf import json_format

logger = logging.getLogger("api-logger")

ROOT_PATH = "/triton"

router = APIRouter(prefix=ROOT_PATH)


class Datatypes(Enum):
    float16 = "TYPE_FP16"
    float32 = "TYPE_FP32"
    float64 = "TYPE_FP64"
    int8 = "TYPE_INT8"
    int16 = "TYPE_INT16"
    int32 = "TYPE_INT32"
    int64 = "TYPE_INT64"
    uint8 = "TYPE_UINT8"
    uint16 = "TYPE_UINT16"
    uint32 = "TYPE_UINT32"
    uint64 = "TYPE_UINT64"
    bool = "TYPE_BOOL"
    string = "TYPE_STRING"


@router.post("")
def get_tritonInfo(
    request: Request,
    models: List[int],
    datasets: List[int],
    url: Optional[str] = "triton:8000",
    task_creator: TaskCreator = Depends(get_task_creator),
) -> None:
    user: KeycloakUser = request.user

    def _run_inference(
        on_error,
        on_progress_change,
        user: KeycloakUser,
        models,
        datasets,
        url,
    ) -> dict:
        try:
            # The task runs in another thread than the request. It therefore opens
            # its own session: the request session is closed as soon as the
            # response is sent.
            with SessionLocal() as task_db:
                for model_id in models:
                    token = user.minio_token
                    model = check_exists(sql_api.get_model(task_db, model_id))
                    model_name = str(model_id)

                    # initialize Triton connection
                    # Triton Command:
                    # docker run --gpus=1 --rm -p6000:8000 -p6001:8001 -p6002:8002 -v/:/models nvcr.io/nvidia/tritonserver:23.09-py3 tritonserver --model-repository=s3://<platform_path>:9000/triton --model-control-mode=poll
                    try:
                        triton_client = httpclient.InferenceServerClient(
                            url=url, verbose=True
                        )
                    except Exception as e:
                        raise RuntimeError("Context creation failed: " + str(e))

                    # download the model file
                    # get all filenames in the model directory
                    bucket_name = model.bucket_name
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

                    # get the actual model file and upload it to the triton bucket
                    _upload_model_files_to_triton(
                        bucket_name, token, model_objects, model_filepath, model, model_name
                    )

                    try:
                        retries = 0
                        while not triton_client.is_model_ready(model_name=model_name):
                            time.sleep(0.5)
                            retries += 1
                            if retries > 300:
                                raise RuntimeError(
                                    "Triton failed to load the model in time."
                                )

                        model_metadata = triton_client.get_model_metadata(
                            model_name=model_name
                        )
                    except InferenceServerException as e:
                        raise RuntimeError("Failed to retrieve the metadata: " + str(e))

                    try:
                        model_config = triton_client.get_model_config(model_name=model_name)
                    except InferenceServerException as e:
                        raise RuntimeError("Failed to retrieve the config: " + str(e))

                    model_metadata, model_config = convert_http_metadata_config(
                        model_metadata, model_config
                    )

                    # retrieve necessary values for preprocessing from config and metadata
                    (
                        max_batch_size,
                        input_name,
                        output_name,
                        c,
                        h,
                        w,
                        format,
                        dtype,
                    ) = parse_model(model_metadata, model_config, model)

                    # download the dataset files
                    for dataset_id in datasets:
                        image_data, filenames = _load_image_data(
                            token, dtype, c, h, w, task_db, dataset_id, format
                        )

                        # Send requests of FLAGS.batch_size images. If the number of
                        # images isn't an exact multiple of FLAGS.batch_size then just
                        # start over with the first images until the batch is filled.
                        # currently not supporting streaming
                        responses = []
                        filenames.sort()

                        supports_batching = model_config.max_batch_size > 0
                        batch_size = (
                            1
                            if model_config.max_batch_size == 0
                            else model_config.max_batch_size
                        )

                        # Holds the handles to the ongoing HTTP async requests.
                        async_requests = _infer_requests_async(
                            triton_client,
                            model_config,
                            requestGenerator,
                            image_data,
                            input_name,
                            output_name,
                            dtype,
                            filenames,
                            model,
                            batch_size,
                            supports_batching,
                            model_name,
                        )

                        # Collect results from the ongoing async requests
                        # for HTTP Async requests.
                        for async_request in async_requests:
                            responses.append(async_request.get_result())

                        bucket = user.minio_bucket_name

                        _collect_results_from_async_requests(
                            responses,
                            on_progress_change,
                            output_name,
                            batch_size,
                            supports_batching,
                            task_db,
                            bucket,
                            token,
                            model_id,
                            dataset_id,
                        )
        except Exception as e:
            logger.exception("Inference run failed")
            on_error(task_error_message(e))

    _, task_location_url, _ = task_creator.create_background_task(
        func=_run_inference,
        task_title=f"Inference for Model(s) {models} and Dataset(s) {datasets}.",
        user=user,
        models=models,
        datasets=datasets,
        url=url,
    )

    headers = {"Location": task_location_url}
    return Response(status_code=status.HTTP_202_ACCEPTED, headers=headers)


def _upload_model_files_to_triton(
    bucket_name, token, model_objects, model_filepath, model, model_name
):
    for item in model_objects:
        # currently only handling single files
        if not item.is_dir:
            model_filepath == model_objects[0].object_name
            file_bytes = minio_api.get_object(
                bucket_name, object_name=model_filepath, token=token
            ).read()

            # only four filetypes available in triton
            file_ending = model_filepath.split(".")[-1]
            if file_ending in ["plan", "onnx", "pt", "graphdef"]:
                # pt and graphdef need a special model config
                if file_ending in ["pt", "graphdef"]:
                    # check if all needed attributes for config exist
                    if not all(
                        getattr(model, attr) is not None
                        for attr in [
                            "input_datatype",
                            "input_shape",
                            "input_semantics",
                            "input_name",
                            "output_name",
                            "output_datatype",
                            "output_shape",
                        ]
                    ):
                        raise RuntimeError(
                            "For Models of the format .pt or .graphdef the input tensor info and output tensor info must be defined"
                        )

                    backend = (
                        "pytorch" if file_ending == "pt" else "tensorflow_graphdef"
                    )

                    # create the actual config as protobuf
                    config = {
                        "platform": backend,
                        "max_batch_size": 0,
                        "input": [
                            {
                                "name": model.input_name,
                                "data_type": Datatypes[
                                    model.input_datatype.value
                                ].value,
                                "dims": model.input_shape,
                                "format": "FORMAT_" + model.input_semantics.value,
                            }
                        ],
                        "output": [
                            {
                                "name": model.output_name,
                                "data_type": Datatypes[
                                    model.output_datatype.value
                                ].value,
                                "dims": model.output_shape,
                            }
                        ],
                    }

                    cf = model_config_pb2.ModelConfig()
                    cf = json_format.ParseDict(config, cf)
                    logger.info(cf)

                    # upload the config to the correct endpoint
                    minio_api.upload_data(
                        bucket="triton",
                        prefix=model_name,
                        token=token,
                        objectname="config.pbtxt",
                        data=bytes(cf),
                    )

                    logger.info("build model config")

                # upload the actual model file
                minio_api.upload_data(
                    bucket="triton",
                    prefix=f"{model_name}/1",
                    token=token,
                    objectname=f"model.{file_ending}",
                    data=file_bytes,
                )
            else:
                raise RuntimeError(
                    "File ending "
                    + file_ending
                    + " is not supported by the triton backend"
                )


def _infer_requests_async(
    triton_client,
    model_config,
    requestGenerator,
    image_data,
    input_name,
    output_name,
    dtype,
    filenames,
    model,
    batch_size,
    supports_batching,
    model_name,
):
    async_requests = []
    sent_count = 0
    image_idx = 0
    last_request = False

    while not last_request:
        input_filenames = []
        repeated_image_data = []

        for idx in range(batch_size):
            input_filenames.append(filenames[image_idx])
            repeated_image_data.append(image_data[image_idx])
            image_idx = (image_idx + 1) % len(image_data)
            if image_idx == 0:
                last_request = True

        if supports_batching:
            batched_image_data = np.stack(repeated_image_data, axis=0)
        else:
            batched_image_data = repeated_image_data[0]

        # Send request
        try:
            for inputs, outputs in requestGenerator(
                batched_image_data, input_name, output_name, dtype
            ):
                sent_count += 1
                async_requests.append(
                    triton_client.async_infer(
                        model_name,
                        inputs,
                        request_id=str(sent_count),
                        outputs=outputs,
                    )
                )
        except InferenceServerException as e:
            print("inference failed: " + str(e))
            return None

    return async_requests


def _load_image_data(token, dtype, c, h, w, db, dataset_id, format):
    image_data = []
    dataset_files = {}
    filenames = []

    dataset = check_exists(dataset_sql_api.get_dataset(db, dataset_id))
    bucket_name = dataset.bucket_name
    dataset_prefix = f"datasets/{dataset.id}"

    for item in minio_api.get_all_objects(
        bucket_name, prefix=dataset_prefix, token=token
    ):
        if item.is_dir is False and "annotations" not in item.object_name:
            dataset_files[item.object_name] = minio_api.get_object(
                bucket=bucket_name,
                token=token,
                object_name=item.object_name,
            ).read()

            filenames.append(item.object_name)
            img = Image.open(BytesIO(dataset_files[item.object_name]))
            image_data.append(preprocess(img, format, dtype, c, h, w))

    return image_data, filenames


def _collect_results_from_async_requests(
    responses,
    on_progress_change,
    output_name,
    batch_size,
    supports_batching,
    db,
    bucket,
    token,
    model_id,
    dataset_id,
):
    results = {}

    for response in responses:
        this_id = response.get_response()["id"]
        print("Request {}, batch size {}".format(this_id, batch_size))
        results[this_id] = postprocess(
            response, output_name, batch_size, supports_batching
        )
        on_progress_change(int(this_id) / len(results))

    logger.info("PASS")
    logger.info(results)

    tasks = tasks_api.get_tasks(db)
    tasks.sort(key=lambda x: x.id)
    logger.info(tasks)

    minio_api.upload_data(
        bucket,
        prefix="inference/" + str(tasks[-1].id),
        token=token,
        data=json.dumps(results).encode("utf-8"),
        objectname="Model" + str(model_id) + "_Dataset" + str(dataset_id) + ".json",
    )


def preprocess(img, format, dtype, c, h, w):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    # np.set_printoptions(threshold='nan')

    if c == 1:
        sample_img = img.convert("L")
    else:
        sample_img = img.convert("RGB")

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    scaled = typed

    # Swap to CHW if necessary
    if format == "NCHW":
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    # workaround for batchsize 0
    if len(ordered) == 3:
        ordered = np.expand_dims(ordered, axis=0)

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def convert_http_metadata_config(_metadata, _config):
    # NOTE: attrdict broken in python 3.10 and not maintained.
    # https://github.com/wallento/wavedrompy/issues/32#issuecomment-1306701776
    try:
        from attrdict import AttrDict
    except ImportError:
        # Monkey patch collections
        import collections
        import collections.abc

        for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
        from attrdict import AttrDict

    return AttrDict(_metadata), AttrDict(_config)


def parse_model(model_metadata, model_config, model):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception(
            "expecting 1 output, got {}".format(len(model_metadata.outputs))
        )

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)
            )
        )

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception(
            "expecting output datatype to be FP32, model '"
            + model_metadata.name
            + "' output type is "
            + output_metadata.datatype
        )

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = model_config.max_batch_size > 0
    non_one_cnt = 0
    for dim in output_metadata.shape:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    # currently handling the additional entry in shape metadata hardcoded

    input_batch_dim = len(model.input_semantics.value) > 3
    expected_input_dims = 3 + (1 if input_batch_dim else 0)

    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".format(
                expected_input_dims, model_metadata.name, len(input_metadata.shape)
            )
        )

    if type(model.input_semantics.value) == str:
        input_config.format = model.input_semantics.value

    # currently hardcoded comparisons
    if (input_config.format != "NCHW") and (input_config.format != "NHWC"):
        raise Exception(
            "unexpected input format "
            + mc.ModelInput.Format.Name(input_config.format)
            + ", expecting "
            + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW)
            + " or "
            + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC)
        )

    if input_config.format == "NHWC":
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (
        model_config.max_batch_size,
        input_metadata.name,
        output_metadata.name,
        c,
        h,
        w,
        input_config.format,
        input_metadata.datatype,
    )


def postprocess(results, output_name, batch_size, supports_batching):
    """
    Post-process results to show classifications.
    """

    output_array = results.as_numpy(output_name)[0].tolist()

    return output_array


def requestGenerator(batched_image_data, input_name, output_name, dtype):
    client = httpclient

    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = [client.InferRequestedOutput(output_name)]

    yield inputs, outputs


def _validate_parameters(bucket, token):
    try:
        minio_api.valid_params(bucket, token)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
