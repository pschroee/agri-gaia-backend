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

FROM nvcr.io/nvidia/pytorch:26.03-py3

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Berlin

LABEL agrigaia.metrics="jsonl/1"

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

COPY mount-s3.sh /mount-s3.sh
RUN chmod +x /mount-s3.sh

RUN apt update \
  && apt install -y --no-install-recommends \
  kmod fuse s3fs jq libxml2-utils curl wget tree \
  python3 python-is-python3 python3-pip
  
RUN sed -i '/user_allow_other/s/^#//g' /etc/fuse.conf

RUN pip install onnx==1.21.0 tf2onnx==1.17.0 onnxmltools==1.16.0 onnxconverter-common==1.16.0

WORKDIR /workspace

RUN rm -rf ./*
RUN mkdir lightning_logs

COPY config_to_args.py config.py dataset.py cvat_dataset.py export.py ./

COPY train.py args.py util.py setup.py efficientnet.py efficientnet_dataset.py \
  efficientnet_datamodule.py metrics_callback.py requirements.txt train_config.json presets.json export_config.json ./

# Wheel index auto-selected from host GPUs by the backend (build arg).
# Default cu126 covers CC 7.0-9.0 (incl. Volta V100) for manual builds / detection failure.
ARG TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126
RUN pip3 install --index-url ${TORCH_CUDA_INDEX} torch~=2.11.0 torchvision~=0.26.0
RUN pip3 install -r requirements.txt

COPY .env ./

ENTRYPOINT ["/bin/bash", "/docker-entrypoint.sh"]
CMD ["$@"]
