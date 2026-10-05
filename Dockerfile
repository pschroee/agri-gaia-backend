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

FROM alpine AS backend_binaries

ARG DOCKER_VERSION
ARG NUCLIO_VERSION

RUN wget https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz && tar xzf docker-${DOCKER_VERSION}.tgz

RUN wget https://github.com/nuclio/nuclio/releases/download/${NUCLIO_VERSION}/nuctl-${NUCLIO_VERSION}-linux-amd64
RUN chmod +x nuctl-${NUCLIO_VERSION}-linux-amd64 && mv nuctl-${NUCLIO_VERSION}-linux-amd64 nuctl

FROM alpine AS docker_config_stage

RUN apk update && apk add jq moreutils

ARG NVIDIA_NGC_API_KEY

WORKDIR /root
COPY ./config/docker-client/config.json ./config/docker-client/add-nvidia-key.sh ./
RUN ./add-nvidia-key.sh config.json "$NVIDIA_NGC_API_KEY"

FROM python:3.12-bookworm

WORKDIR /code
RUN apt update && apt install -y dnsutils

COPY --from=backend_binaries docker/docker /usr/local/bin
COPY --from=docker/buildx-bin:latest /buildx /usr/local/lib/docker/cli-plugins/docker-buildx

COPY --from=backend_binaries nuctl /usr/local/bin

COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir --upgrade --timeout 120 --retries 5 -r /code/requirements.txt -c /code/constraints.txt

COPY . .
ARG KEYCLOAK_REALM_NAME
RUN sed -i "s/test-realm/${KEYCLOAK_REALM_NAME}/g" agri_gaia_backend/services/portainer/portainer_api.py

RUN mkdir -p /root/.docker
COPY --from=docker_config_stage /root/config.json /root/.docker/config.json

CMD ["python3", "-m", "debugpy", "--listen", "0.0.0.0:5678", "-m", "uvicorn", "agri_gaia_backend.main:app", "--reload", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["uvicorn", "agri_gaia_backend.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
