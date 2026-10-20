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

import inspect
import io
import mimetypes
from zipfile import ZipFile
from typing import Callable, List, Tuple, TypeVar, Dict
from concurrent.futures import ThreadPoolExecutor

from fastapi import Request, Response, HTTPException
from sqlalchemy.orm import Session
from agri_gaia_backend.db.database import SessionLocal
from agri_gaia_backend.db import tasks_api
from agri_gaia_backend.db.models import Task, TaskStatus
from agri_gaia_backend.schemas.keycloak_user import KeycloakUser
from agri_gaia_backend.util import env
from agri_gaia_backend.routers.paths import TASKS_ROOT_PATH

import logging

logger = logging.getLogger("api-logger")


# FastAPI Dependency
def get_db() -> SessionLocal:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class TaskCreator:
    """
    TaskCreator class. This class is responsible for creating tasks and running them in a background thread.

    In this implementation a ThreadPoolExecutor is used instead of FastAPIs BackgroundTasks.
    The reason for this is that we use the BaseHTTPMiddleware class of Starlette which is incompatible
    with the BackgroundTasks in FastAPI (which needs a rework anyways as far as I understood).
    See:
    - https://github.com/encode/starlette/blob/4e3a8c570ffc801d0dcffe3a0a69bb57dac0d1eb/docs/middleware.md#limitations
    - https://github.com/encode/starlette/issues/919
    - https://github.com/encode/starlette/issues/1438
    - https://github.com/encode/starlette/issues/1678
    """

    executor: ThreadPoolExecutor = ThreadPoolExecutor()

    def __init__(self, initiator: str) -> None:
        self.initiator = initiator

    @staticmethod
    def _get_task_location_url(task_id: int) -> str:
        return f"https://api.{env.PROJECT_BASE_URL}{TASKS_ROOT_PATH}/{task_id}"

    def create_background_task(
        self, func: Callable, task_title: str, *args, **kwargs
    ) -> Tuple[Task, str]:
        """
        Runs the given callable in a background thread and creates a Task object
        which is stored in the database. The Tasks status is updated when the
        Task is completed.

        Args:
            func (Callable): The callable to be executed in the background thread.
                            In addition to the given positional and keyword arguments the
                            function gets two callback functions:
                            on_progress_change (float -> None): which can be called to update the tasks progress.
                                                                The progress should be between 0 and 1
                            on_error (str -> None): which should be called if an error occurs.
                                                    The given message will be displayed in the task and
                                                    the task will be marked as failed.
                            on_message (str -> None): optional, passed only if func declares it. Sets the
                                                    task message, kept when the task completes.
            args: positional arguments given to func
            kwargs: keyword arguments given to func

        Returns:
            Tuple[Task, str]: The created Task and the task objects location as url string.
        """

        def task_func():
            def task_progress_change_handler(completion_percentage: float) -> None:
                if completion_percentage <= 1.0 and completion_percentage > 0.0:
                    task.completion_percentage = completion_percentage
                    tasks_api.update_task(db, task)
                else:
                    logger.warn("Invalid value range of completion percentage")

            def on_error(message: str) -> None:
                nonlocal error_message
                error_message = message
                nonlocal task_execution_failed
                task_execution_failed = True

            def on_message(message: str) -> None:
                task.message = message
                tasks_api.update_task(db, task)

            error_message = "Task failed. See backend logs for details."
            task_execution_failed = False
            try:
                task.status = TaskStatus.inprogress
                tasks_api.update_task(db, task)
                extra = (
                    {"on_message": on_message}
                    if "on_message" in inspect.signature(func).parameters
                    else {}
                )
                try:
                    func(
                        *args,
                        on_progress_change=task_progress_change_handler,
                        on_error=on_error,
                        **extra,
                        **kwargs,
                    )
                except NotImplementedError:
                    task_execution_failed = True
                    error_message = (
                        "This task failed because this feature is not yet implemented."
                    )
                except Exception as e:
                    logger.exception(e)
                    task_execution_failed = True
                    raise e
            finally:
                if task_execution_failed:
                    task.status = TaskStatus.failed
                    task.message = error_message
                else:
                    task.status = TaskStatus.completed
                    task.completion_percentage = 1.0
                tasks_api.update_task(db, task)
                db.close()

        try:
            db: Session = SessionLocal()
            task = tasks_api.create_task(db, initiator=self.initiator, title=task_title)
            future = TaskCreator.executor.submit(task_func)
        except Exception as e:
            db.close()
            raise e

        return task, self._get_task_location_url(task.id), future


def get_task_creator(request: Request) -> TaskCreator:
    user: KeycloakUser = request.user
    initiator = user.username

    return TaskCreator(initiator)


def create_zip_file_response(files: Dict[str, bytes], filename: str) -> Response:
    """
    Creates an HTTP Resposnse with a ZIP arhcive containing all the files, that are passed to the function.

    Args:
        files: The files which should be included in tzhe resulting zip archive.
        filename: the filename of the resulting archive.

    Returns:
        The zip archive as a Response object
    """
    s = create_zip_file(files)

    response = create_single_file_response(
        s.getvalue(), filename=filename, content_type="application/x-zip-compressed"
    )
    return response


def extract_zip(input_zip):
    """
    Extracts all entries from a zip file into a dictionary object.

    Args:
        input_zip: The archive which should be unzipped

    Returns:
        A dictionary containing everything extracted from the zip archive.
    """
    input_zip = ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}


def create_zip_file(files: Dict[str, bytes]) -> io.BytesIO:
    """
    Creates an in memory ZIP archive containing all the files, that are passed to the funktion

    Args:
        files: The files which should be included in tzhe resulting zip archive.

    Returns:
        The in memory archive.
    """
    s = io.BytesIO()
    zf = ZipFile(s, "w")

    for key in files:
        zf.writestr(key, files[key])

    zf.close()
    return s


def create_single_file_response(
    file: bytes, filename: str, content_type=None
) -> Response:
    """
    Creates a Response object containing a single file.

    Args:
        file: The file, whicch should be included in the response.
        filename: The filename of the included file.
        content_type: the content_type of the file. Will be filled automatically, if no type is given and defaults to "application/octet-stream".

    Returns:
        The created Respose object.
    """
    if content_type is None:
        content_type, _ = mimetypes.guess_type(filename)
        if content_type is None:
            content_type = "application/octet-stream"

    response = Response(
        file,
        headers={
            "Content-Type": content_type,
            "Content-Disposition": f"attachment;filename={filename}",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )

    return response


T = TypeVar("T")


def check_exists(obj: T, detail: str = None) -> T:
    """
    Throws 404 HTTPException when obj is None

    obj         object to be checked if it is None
    detail      detail for the 404 message

    return      object
    """

    if not obj:
        raise HTTPException(status_code=404, detail=detail)
    return obj
