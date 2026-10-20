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
from typing import List, Optional
from sqlalchemy.orm import Session

from agri_gaia_backend.db.models import Task, TaskStatus

# from agri_gaia_backend.schemas import task as schemas


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
