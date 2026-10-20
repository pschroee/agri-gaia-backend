#!/usr/bin/env bash

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

if [[ -f .env ]] && [[ -s .env ]]; then
  source .env
  echo "Sourced additional environment from .env."
  cat .env
fi

if [[ -z "${DATASET_ID}" ]]; then
  echo "Error: DATASET_ID not in environment." && exit 1;
fi

/bin/bash /mount-s3.sh -t "${OID_ACCESS_TOKEN}" -s "${S3_HOST}" -b "${S3_BUCKET_NAME}" -m '/minio'

tree "/minio/datasets/${DATASET_ID}"

if command -v nvidia-smi &> /dev/null && nvidia-smi -L > /dev/null 2>&1; then
  nvidia-smi -L
  # Sample GPU usage every 5 s; the backend serves the file through the metrics route.
  mkdir -p /train
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -l 5 >> /train/gpu.csv 2>/dev/null &
fi

DATETIME_FORMAT="%Y-%m-%dT%H:%M:%S"

if [[ -f export_config.json ]] && [[ "$(jq 'length == 0' export_config.json)" == "false" ]]; then
  echo "ONNX model export will be attemped using the following configuration":
  jq '.' < export_config.json
fi

pwd \
  && tree . \
  && echo "Setup started: $(date +"${DATETIME_FORMAT}")" \
  && time python setup.py \
  && echo "Setup ended: $(date +"${DATETIME_FORMAT}")" \
  && jq '.' < train_config.json \
  && echo "Training started: $(date +"${DATETIME_FORMAT}")" \
  && eval "time python train.py $(python config_to_args.py)" \
  && echo "Training ended: $(date +"${DATETIME_FORMAT}")"