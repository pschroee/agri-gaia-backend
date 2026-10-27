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

FROM ultralytics/ultralytics:8.4.46-runner

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

# Only requirements.txt before the heavy pip layers: code and the per-build
# configuration follow the last RUN so that they never invalidate those layers.
COPY requirements.txt ./

# Re-point the base image's torch at the host-matched CUDA wheel, keeping the
# exact version Ultralytics ships. Index is auto-selected from host GPUs by the
# backend (build arg); default cu126 covers CC 7.0-9.0 (incl. Volta V100).
#
# IMPORTANT: request the +cuXXX *local* version label. The base image ships
# torch==2.11.0+cu128; a bare `pip install torch==2.11.0` is a PEP 440 no-op
# (the installed 2.11.0+cu128 already satisfies ==2.11.0), so the cu126 wheel
# would never replace it and Volta (sm_70) stays unsupported. Deriving the tag
# (e.g. cu126) from the index URL and pinning torch==<ver>+cu126 forces pip to
# actually swap in the Volta-capable build.
ARG TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126
RUN CU="$(basename "${TORCH_CUDA_INDEX}")" && \
    TORCH_VER="$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')" && \
    TV_VER="$(python3 -c 'import torchvision; print(torchvision.__version__.split("+")[0])')" && \
    pip3 install --index-url "${TORCH_CUDA_INDEX}" \
      "torch==${TORCH_VER}+${CU}" "torchvision==${TV_VER}+${CU}"
RUN pip3 install -r requirements.txt

COPY config_to_args.py config.py dataset.py cvat_dataset.py setup.py util.py train.py \
  export.py ./

COPY train_config.json custom.json presets.json export_config.json ./

COPY .env ./

ENTRYPOINT ["/bin/bash", "/docker-entrypoint.sh"]
CMD ["$@"]
