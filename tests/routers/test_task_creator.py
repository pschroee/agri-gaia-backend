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

from agri_gaia_backend.routers.common import TaskCreator
from agri_gaia_backend.db import tasks_api
from agri_gaia_backend.db.models import TaskStatus


def test_create_background_task(request, task_creator: TaskCreator, test_user, db):

    executed = False

    def test_task(on_error, on_progress_change):
        nonlocal executed
        executed = True

    task, _, future = task_creator.create_background_task(test_task, "Task Title")
    while not future.done():
        pass
    task = tasks_api.get_task(db, task.id)
    request.addfinalizer(lambda: tasks_api.delete_task(db, task))

    assert executed, "Task wasn't executed"
    assert task is not None, "Task is not in db"
    assert task.status == TaskStatus.completed, "Task status not completed"
    assert task.initiator == test_user.username, "Task initiator not set"


def test_background_task_fails(request, task_creator: TaskCreator, db):

    executed = False
    error_message = "Task failed"

    def test_task(on_error, on_progress_change):
        on_error(error_message)

    task, _, future = task_creator.create_background_task(test_task, "Task Title")
    while not future.done():
        pass
    task = tasks_api.get_task(db, task.id)
    request.addfinalizer(lambda: tasks_api.delete_task(db, task))

    assert not executed, "Task wasn't executed"
    assert task is not None, "Task is not in db"
    assert task.status == TaskStatus.failed, "Task status wrong"
    assert task.message == error_message, "Error message not set"


def test_background_task_fails_exception(request, task_creator: TaskCreator, db):

    executed = False

    def test_task(on_error, on_progress_change):
        raise Exception()

    task, _, future = task_creator.create_background_task(test_task, "Task Title")
    while not future.done():
        pass
    task = tasks_api.get_task(db, task.id)
    request.addfinalizer(lambda: tasks_api.delete_task(db, task))

    assert not executed, "Task wasn't executed"
    assert task is not None, "Task is not in db"
    assert task.status == TaskStatus.failed, "Task status wrong"


def test_background_task_message(request, task_creator: TaskCreator, db):
    def test_task(on_error, on_progress_change, on_message):
        on_message("Train Container #7 created.")

    task, _, future = task_creator.create_background_task(test_task, "Task Title")
    while not future.done():
        pass
    task = tasks_api.get_task(db, task.id)
    request.addfinalizer(lambda: tasks_api.delete_task(db, task))

    assert task.status == TaskStatus.completed, "Task status not completed"
    assert task.message == "Train Container #7 created.", "Message not kept"


def test_background_task_without_message_param(
    request, task_creator: TaskCreator, db
):
    def test_task(on_error, on_progress_change):
        pass

    task, _, future = task_creator.create_background_task(test_task, "Task Title")
    while not future.done():
        pass
    task = tasks_api.get_task(db, task.id)
    request.addfinalizer(lambda: tasks_api.delete_task(db, task))

    assert task.status == TaskStatus.completed, "Task status not completed"


def test_train_container_created_message():
    from agri_gaia_backend.routers.train import train_container_created_message

    assert train_container_created_message(63) == "Train Container #63 created."
