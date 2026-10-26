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

from typing import Optional, List
import datetime
from pydantic import BaseModel

from agri_gaia_backend.schemas.container_deployment import ContainerDeployment


class EdgeDeviceBase(BaseModel):
    name: str


class EdgeDeviceCreate(EdgeDeviceBase):
    tags: List[str]


class EdgeDevice(EdgeDeviceBase):
    id: int
    arch: Optional[str]
    portainer_id: Optional[int]
    os: Optional[str]
    cpu_count: Optional[int]
    arch: Optional[str]
    memory: Optional[int]
    tags: Optional[List[str]]
    last_heartbeat: Optional[datetime.datetime]
    registered: bool
    edge_key: str
    container_deployments: List[ContainerDeployment]

    class Config:
        orm_mode = True


class EdgeContainer(BaseModel):
    name: str
    state: str
    host_ports: List[int] = []
