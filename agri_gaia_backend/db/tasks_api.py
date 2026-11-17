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

import datetime
import logging
import os
from typing import Callable, List, Optional
from sqlalchemy.orm import Session

from agri_gaia_backend.db.models import Task, TaskStatus

# from agri_gaia_backend.schemas import task as schemas

logger = logging.getLogger("api-logger")

INTERRUPTED_MESSAGE = "Interrupted by a backend restart."


def get_task(db: Session, task_id: int) -> Task:
    return db.query(Task).filter(Task.id == task_id).first()


def get_tasks(
    db: Session, skip: int = 0, limit: int = 100, ids: Optional[List[int]] = None
) -> List[Task]:
    q = db.query(Task)
    if ids:
        q = q.filter(Task.id.in_(ids))
    return q.offset(skip).limit(limit).all()


def get_tasks_by_initiator(
    db: Session, initiator: str, skip: int = 0, limit: int = 100
) -> List[Task]:
    return (
        db.query(Task)
        .filter(Task.initiator == initiator)
        .order_by(Task.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def create_task(db: Session, initiator: str, title: Optional[str] = None) -> Task:
    db_task = Task(
        initiator=initiator,
        title=title,
        creation_date=datetime.datetime.now(),
        status=TaskStatus.created,
        completion_percentage=0.0,
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task


def update_task(db: Session, task: Task) -> Task:
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def delete_task(db: Session, task: Task) -> bool:
    db.delete(task)
    db.commit()
    return True


def fail_interrupted_tasks(
    db: Session, message: str = INTERRUPTED_MESSAGE
) -> List[int]:
    """Mark every task that is still `created` or `inprogress` as `failed`.

    Tasks run as threads inside the backend process. When the process starts,
    no such thread exists any more, so every unfinished row is orphaned: its
    `finally` block never ran. Returns the ids of the tasks that were changed.
    """
    tasks = (
        db.query(Task)
        .filter(Task.status.in_([TaskStatus.created, TaskStatus.inprogress]))
        .all()
    )
    for task in tasks:
        task.status = TaskStatus.failed
        task.message = message
    if tasks:
        db.commit()
    return [task.id for task in tasks]


def fail_interrupted_tasks_on_startup(session_factory: Callable[[], Session]) -> None:
    """Startup hook: clean up orphaned tasks, never fail the startup.

    The backend runs as a single Uvicorn process. With several workers, a
    starting worker could not tell its own tasks from those of a running
    sibling, so nothing is done in that case.
    """
    try:
        workers = int(os.environ.get("WEB_CONCURRENCY") or 1)
    except ValueError:
        workers = 1
    if workers > 1:
        logger.warning(
            "Interrupted task cleanup skipped: more than one worker (%d).", workers
        )
        return
    db = None
    try:
        db = session_factory()
        ids = fail_interrupted_tasks(db)
        if ids:
            logger.warning(
                "Marked %d interrupted task(s) as failed: %s", len(ids), ids
            )
    except Exception:
        logger.exception("Could not mark interrupted tasks as failed")
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                logger.exception("Could not close the session")
