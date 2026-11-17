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

# load environment variables
import logging
from agri_gaia_backend import db
from agri_gaia_backend.db import tasks_api
from agri_gaia_backend.util.common import get_stacktrace
from agri_gaia_backend.util.env import bool_from_env
from agri_gaia_backend.services.portainer.portainer_api import portainer
from agri_gaia_backend.services.docker import image_builder
from agri_gaia_backend.routers.exception_handlers import (
    _missing_input_data_exception_handler,
)
from agri_gaia_backend.routers import (
    applications,
    container_images,
    datasets,
    cvat,
    inference_container_templates,
    models,
    users,
    backend_services,
    container_deployments,
    model_deployments,
    edge_devices,
    edge_groups,
    agrovoc,
    geonames,
    train,
    tasks,
    tags,
    open_data,
    urls,
    integrated_services,
    licenses,
    network,
    triton,
)
from agri_gaia_backend.util.auth.bearer_token_auth_backend import BearerTokenAuthBackend
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi import Response, status
from starlette.authentication import UnauthenticatedUser
from starlette.middleware.authentication import AuthenticationMiddleware
import re
import os
from .util.log_config import log_config
from logging.config import dictConfig
from dotenv import load_dotenv


load_dotenv()

# setup logging

dictConfig(log_config)


# from starlette.middleware.sessions import SessionMiddleware


logger = logging.getLogger("api-logger")


#### Portainer ####
portainer.setup()


#### SQLAlchemy ####
# Run DB Migrations with Alembic
should_migrate = bool_from_env("MIGRATE_DB")
if should_migrate:
    command = "alembic upgrade head"
    logger.info(f"Executing '{command}' to migrate db if necessary")
    os.system(command)
else:
    logger.info("Creating db tables from sqlalchemy definitions. Not using alembic")
    db.models.Base.metadata.create_all(bind=db.database.engine)
####################

debug = bool_from_env("DEBUG")
app = FastAPI(debug=debug)


@app.on_event("startup")
def fail_interrupted_tasks_on_startup():
    """Tasks only exist as threads of this process: after a restart, every
    unfinished task is orphaned and would stay `inprogress` forever (#259)."""
    tasks_api.fail_interrupted_tasks_on_startup(db.database.SessionLocal)


#### MIDDLEWARE ####

# Middlewares have to be added in reveser order (from inside out),
# because they are wrapped around the previous one.
# CORS has to be the outer most all the time, so that all requests get the needed CORS-Headers

# Current config:
# CORS -> Expose Headers -> Logs -> Prometheus -> Auth -> AuthCheck -> Auth -> Prometheus -> Logs -> Expose Headers -> CORS


# add a custom middleware that checks if the user created in AuthenticationMiddleware
# is actually authenticated or not.
re_whitelist = [
    "\/edge-devices\/\d+\/config",
    "\/edge-devices\/\d+\/register",
    "\/metrics",
]
if debug:
    debug_routes = ["\/docs", "\/openapi.json"]
    re_whitelist.extend(debug_routes)

re_combined = "(" + ")|(".join(re_whitelist) + ")"


@app.middleware("http")
async def check_auth_middleware(request: Request, call_next):
    if type(request.user) is UnauthenticatedUser:
        if not re.match(re_combined, request.url.path):
            return Response(status_code=status.HTTP_401_UNAUTHORIZED)

    response = await call_next(request)
    return response


# Create a SimpleUser if the auth token is correct, or an UnauthenticatedUser else.
app.add_middleware(AuthenticationMiddleware, backend=BearerTokenAuthBackend())
# app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET_KEY"))


# Prometheus Monitoring
Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def log_errors_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error("Caught Exception. Stacktrace:\n" + get_stacktrace(exc))
        return Response("Internal Server Error", status_code=500)


@app.middleware("http")
async def add_exposed_headers_middleware(request: Request, call_next):
    response = await call_next(request)

    aceKey = "Access-Control-Expose-Headers"
    aceHeaders = "Location"
    if aceKey in response.headers and response.headers[aceKey]:
        aceHeaders += f", {response.headers[aceKey]}"

    response.headers[aceKey] = aceHeaders
    return response


# manage CORS Requests from the Frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=f"https://.*\.{os.environ.get('PROJECT_BASE_URL')}",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#### ROUTERS ####

app.include_router(users.router)
app.include_router(applications.router)
app.include_router(datasets.router)
app.include_router(cvat.router)
app.include_router(models.router)
app.include_router(backend_services.router)
app.include_router(edge_devices.router)
app.include_router(edge_groups.router)
app.include_router(model_deployments.router)
app.include_router(container_images.router)
app.include_router(container_deployments.router)
app.include_router(inference_container_templates.router)
app.include_router(agrovoc.router)
app.include_router(geonames.router)
app.include_router(train.router)
app.include_router(tasks.router)
app.include_router(tags.router)
app.include_router(open_data.router)
app.include_router(urls.router)
app.include_router(integrated_services.router)
app.include_router(licenses.router)
app.include_router(network.router)
app.include_router(triton.router)

app.exception_handler(image_builder.MissingInputDataException)(
    _missing_input_data_exception_handler
)
