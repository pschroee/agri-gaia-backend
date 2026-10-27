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

FROM nvcr.io/nvidia/pytorch:26.03-py3

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Berlin

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

# Only requirements.txt before the heavy pip layers: code and the per-build
# configuration follow the last RUN so that they never invalidate those layers.
COPY requirements.txt ./

# Wheel index auto-selected from host GPUs by the backend (build arg).
# Default cu126 covers CC 7.0-9.0 (incl. Volta V100) for manual builds / detection failure.
ARG TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126
RUN pip3 install --index-url ${TORCH_CUDA_INDEX} torch~=2.11.0 torchvision~=0.26.0
RUN pip3 install -r requirements.txt

COPY config_to_args.py config.py dataset.py cvat_dataset.py export.py train.py args.py \
  util.py setup.py maskrcnn.py maskrcnn_datamodule.py maskrcnn_dataset.py ./

COPY train_config.json presets.json export_config.json ./

COPY .env ./

ENTRYPOINT ["/bin/bash", "/docker-entrypoint.sh"]
CMD ["$@"]
