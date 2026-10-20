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

import enum

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    Numeric,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, column_property
from sqlalchemy.dialects import postgresql

from geoalchemy2 import Geometry
from geoalchemy2.functions import ST_AsGeoJSON, ST_Transform

Base = declarative_base()


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    owner = Column(String)
    name = Column(String)
    public = Column(Boolean)
    last_modified = Column(DateTime)
    annotation_task_id = Column(Integer, nullable=True)
    annotation_labels = Column(postgresql.ARRAY(String), nullable=True)
    filecount = Column(Integer)
    total_filesize = Column(BigInteger)
    metadata_uri = Column(String, nullable=True)
    bucket_name = Column(String)
    minio_location = Column(String)

    # temporary just a String. ALembic seems to have issues handling some Postgres Enums from time to time
    # https://github.com/sqlalchemy/alembic/issues/278
    dataset_type = Column(String, nullable=False)


class Service(Base):
    __tablename__ = "services"

    id = Column(Integer, primary_key=True, index=True)
    owner = Column(String)
    name = Column(String)
    public = Column(Boolean)
    last_modified = Column(DateTime)
    bucket_name = Column(String)
    metadata_uri = Column(String, nullable=True)


class Connector(Base):
    __tablename__ = "connectors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(String)
    data_url = Column(String)
    ids_url = Column(String)
    minio_url = Column(String)
    api_key = Column(String)


class ModelFormat(enum.Enum):
    onnx = "onnx"
    pytorch = "pytorch"
    tensorflow = "tensorflow"
    tensorrt = "tensorrt"


class InputTensorShapeSemantics(enum.Enum):
    HWC = "HWC"
    NHWC = "NHWC"
    CHW = "CHW"
    NCHW = "NCHW"


class TensorDatatype(enum.Enum):
    """
    Tensor datatypes inspired by supported triton datatypes: https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#datatypes
    looked up version: https://github.com/triton-inference-server/server/blob/81fd197588025604eb863635c0238adf051babbb/docs/model_configuration.md
    """

    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    bool = "bool"
    string = "string"


# AI Model
class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    owner = Column(String)
    name = Column(String)
    public = Column(Boolean)
    last_modified = Column(DateTime)
    file_size = Column(BigInteger, nullable=True)
    bucket_name = Column(String)
    file_name = Column(String, nullable=True)
    metadata_uri = Column(String, nullable=True)
    deployments = relationship("ModelDeployment", back_populates="model")

    format = Column(Enum(ModelFormat), nullable=True)
    input_name = Column(String, nullable=True)
    input_datatype = Column(Enum(TensorDatatype), nullable=True)
    input_shape = Column(ARRAY(Integer), nullable=True)
    input_semantics = Column(Enum(InputTensorShapeSemantics), nullable=True)
    output_name = Column(String, nullable=True)
    output_datatype = Column(Enum(TensorDatatype), nullable=True)
    output_shape = Column(ARRAY(Integer), nullable=True)
    output_labels = Column(ARRAY(String), nullable=True)


class DeploymentType(enum.Enum):
    edge = "edge"
    cloud = "cloud"


class ModelDeploymentStatus(enum.Enum):
    created = "created"
    running = "running"
    exited = "exited"
    failed = "failed"


class ContainerDeploymentStatus(enum.Enum):
    created = "created"
    deployed = "deployed"
    undeployed = "undeployed"
    failed = "failed"


class PortBindingProtocol(enum.Enum):
    udp = "udp"
    tcp = "tcp"


class PortBinding(Base):
    __tablename__ = "port_bindings"

    host_port = Column(Integer, primary_key=True)
    container_port = Column(Integer, primary_key=True)
    protocol = Column(Enum(PortBindingProtocol), primary_key=True)
    container_deployment_id = Column(
        Integer, ForeignKey("container_deployments.id"), primary_key=True
    )


class ModelDeployment(Base):
    __tablename__ = "model_deployments"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(Enum(DeploymentType))
    model_id = Column(Integer, ForeignKey("models.id"))
    model = relationship("Model", back_populates="deployments")
    edge_device_id = Column(Integer, ForeignKey("edge_devices.id"), nullable=True)
    edge_device = relationship("EdgeDevice", back_populates="model_deployments")
    creation_date = Column(DateTime)
    status = Column(Enum(ModelDeploymentStatus))

    __table_args__ = (
        UniqueConstraint("model_id", "edge_device_id", name="_model_edge_device_uc"),
    )


class ContainerDeployment(Base):
    __tablename__ = "container_deployments"

    id = Column(Integer, primary_key=True, index=True)
    container_image_id = Column(Integer, ForeignKey("container_images.id"))
    container_image = relationship("ContainerImage", back_populates="deployments")
    edge_device_id = Column(Integer, ForeignKey("edge_devices.id"), nullable=True)
    edge_device = relationship("EdgeDevice", back_populates="container_deployments")
    creation_date = Column(DateTime)
    name = Column(String)
    status = Column(Enum(ContainerDeploymentStatus))
    port_bindings = relationship("PortBinding", cascade="delete")
    docker_container_id = Column(String, nullable=True)


class EdgeDevice(Base):
    __tablename__ = "edge_devices"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    portainer_id = Column(Integer, nullable=True)
    os = Column(String, nullable=True)
    cpu_count = Column(Integer, nullable=True)
    arch = Column(String, nullable=True)
    memory = Column(Integer, nullable=True)
    last_heartbeat = Column(DateTime, nullable=True)
    edge_key = Column(String)
    registered = Column(Boolean, default=False)
    container_deployments = relationship(
        "ContainerDeployment", back_populates="edge_device", cascade="delete"
    )
    model_deployments = relationship(
        "ModelDeployment", back_populates="edge_device", cascade="delete"
    )


class ContainerImage(Base):
    __tablename__ = "container_images"

    id = Column(Integer, primary_key=True, index=True)
    owner = Column(String)
    repository = Column(String)
    tag = Column(String)
    platform = Column(String)
    exposed_ports = Column(ARRAY(String))
    last_modified = Column(DateTime)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=True)
    metadata_uri = Column(String, nullable=True)
    deployments = relationship("ContainerDeployment", back_populates="container_image")
    compressed_image_size = Column(BigInteger, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "repository", "tag", "platform", name="_repository_tag_platform_uc"
        ),
    )


class InferenceContainerTemplate(Base):
    __tablename__ = "inference_container_templates"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    git_url = Column(String, nullable=True)
    git_ref = Column(String, nullable=True)
    dirname = Column(String, nullable=False)


class TrainContainer(Base):
    __tablename__ = "train_containers"

    id = Column(Integer, primary_key=True, index=True)
    container_id = Column(String, nullable=True)
    image_id = Column(String)
    owner = Column(String)
    repository = Column(String)
    tag = Column(String)
    last_modified = Column(DateTime)

    provider = Column(String)
    category = Column(String)
    architecture = Column(String)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    dataset = relationship("Dataset")
    status = Column(String, nullable=True)
    model_filepath = Column(String)
    score_name = Column(String)
    score_regexp = Column(String)
    score = Column(Float, nullable=True)


class TaskStatus(enum.Enum):
    created = "created"
    inprogress = "inprogress"
    completed = "completed"
    failed = "failed"


class Task(Base):
    """
    Class for long-running tasks in the backend
    """

    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)
    creation_date = Column(DateTime)
    initiator = Column(String)
    status = Column(Enum(TaskStatus))
    completion_percentage = Column(Float)
    message = Column(String, nullable=True)


class Application(Base):
    """
    An Application is a combination of multiple Docker Images building a Docker Compose Stack
    """

    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    yaml = Column(String)
    portainer_edge_group_ids = Column(ARRAY(Integer))
    portainer_edge_stack_id = Column(Integer)
    last_modified = Column(DateTime)


# This is actually an NDS/Schlaege_2022 (nds_2022) fieldborder.
# For other types of Fieldborders use polymorphism with base class Fieldborder and NdsFieldborder, ... children.
# See: https://docs.sqlalchemy.org/en/14/orm/inheritance.html#concrete-table-inheritance
class Fieldborder(Base):
    __tablename__ = "fieldborders"
    __table_args__ = {"info": {"skip_autogenerate": True}}

    gid = Column(Integer, primary_key=True)
    schlagnr = Column(Numeric(19, 0))
    flik = Column(String)
    geom = Column(Geometry("POLYGON"))
    nrle = Column(Numeric(19, 0))
    flek = Column(String)
    akt_fl = Column(Numeric(13, 4))
    antjahr = Column(Numeric(10, 0))
    kc_festg = Column(String)
    tsbez = Column(String)
    typ_le = Column(String)

    polygon = column_property(ST_AsGeoJSON(ST_Transform(geom, 4326)))
