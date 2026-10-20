#!/usr/bin/env python

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

# -*- coding: utf-8 -*-

import datetime
from typing import Dict, List, Optional

from agri_gaia_backend.db.models import TrainContainer, Dataset
from sqlalchemy.orm import Session


def create_train_container(
    db: Session,
    owner: str,
    image_id: str,
    repository: str,
    tag: str,
    last_modified: datetime.datetime,
    provider: str,
    category: str,
    architecture: str,
    dataset_id: int,
    dataset: Dataset,
    model_filepath: str,
    score_regexp: str,
    score_name: str,
    score: Optional[float] = None,
    config_override: Optional[Dict] = None,
) -> TrainContainer:
    db_train_container = TrainContainer(
        owner=owner,
        image_id=image_id,
        repository=repository,
        tag=tag,
        last_modified=last_modified,
        provider=provider,
        category=category,
        architecture=architecture,
        dataset_id=dataset_id,
        dataset=dataset,
        model_filepath=model_filepath,
        score_regexp=score_regexp,
        score_name=score_name,
        score=score,
        config_override=config_override,
    )
    db.add(db_train_container)
    db.commit()
    db.refresh(db_train_container)
    return db_train_container


def update_train_container(db: Session, train_container: TrainContainer):
    db.add(train_container)
    db.commit()
    db.refresh(train_container)
    return train_container


def get_train_containers(
    db: Session, skip: int = 0, limit: int = 100
) -> List[TrainContainer]:
    return (
        db.query(TrainContainer)
        .order_by(TrainContainer.id)
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_train_container(db: Session, train_container_id: int) -> TrainContainer:
    return (
        db.query(TrainContainer).filter(TrainContainer.id == train_container_id).first()
    )


def delete_train_container(db: Session, train_container: TrainContainer) -> bool:
    db.delete(train_container)
    db.commit()
    return True


def count_train_containers_with_image(
    db: Session, image_id: str, exclude_id: Optional[int] = None
) -> int:
    query = db.query(TrainContainer).filter(TrainContainer.image_id == image_id)
    if exclude_id is not None:
        query = query.filter(TrainContainer.id != exclude_id)
    return query.count()
