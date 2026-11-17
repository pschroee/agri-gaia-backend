# SPDX-FileCopyrightText: 2026 Philipp Schröer
#
# SPDX-License-Identifier: MIT

"""Tests for #259: tasks only exist as threads of the backend process, so after
a restart every task still `created` or `inprogress` is orphaned and is marked
`failed` by a startup hook.

No infrastructure is needed: the session is a fake. Run with `--noconftest`.
"""

import logging
from types import SimpleNamespace

from agri_gaia_backend.db import tasks_api
from agri_gaia_backend.db.models import TaskStatus

MESSAGE = "Interrupted by a backend restart."


class _Query:
    def __init__(self, tasks):
        self._tasks = tasks

    def filter(self, _criterion):
        # The fake plays the database: it applies the filter the function is
        # expected to ask for (status IN created, inprogress).
        open_states = {TaskStatus.created, TaskStatus.inprogress}
        return _Query([t for t in self._tasks if t.status in open_states])

    def all(self):
        return list(self._tasks)


class _Db:
    def __init__(self, tasks):
        self.tasks = tasks
        self.commits = 0
        self.closed = False

    def query(self, _model):
        return _Query(self.tasks)

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


def _task(task_id, status, message=None):
    return SimpleNamespace(id=task_id, status=status, message=message)


def test_fail_interrupted_marks_created_and_inprogress():
    tasks = [
        _task(1, TaskStatus.created),
        _task(2, TaskStatus.inprogress, "building"),
    ]
    db = _Db(tasks)

    ids = tasks_api.fail_interrupted_tasks(db)

    assert ids == [1, 2]
    assert all(t.status == TaskStatus.failed for t in tasks)
    assert all(t.message == MESSAGE for t in tasks)
    assert db.commits == 1


def test_fail_interrupted_keeps_finished():
    done = _task(3, TaskStatus.completed, "done")
    failed = _task(4, TaskStatus.failed, "no space left")
    running = _task(5, TaskStatus.inprogress)
    db = _Db([done, failed, running])

    ids = tasks_api.fail_interrupted_tasks(db)

    assert ids == [5]
    assert done.status == TaskStatus.completed and done.message == "done"
    assert failed.status == TaskStatus.failed and failed.message == "no space left"
    assert running.message == MESSAGE


def test_fail_interrupted_without_orphans_commits_nothing():
    db = _Db([_task(6, TaskStatus.completed)])
    assert tasks_api.fail_interrupted_tasks(db) == []
    assert db.commits == 0


def test_startup_hook_marks_and_closes_session(monkeypatch, caplog):
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    task = _task(7, TaskStatus.inprogress)
    db = _Db([task])

    with caplog.at_level(logging.WARNING, logger="api-logger"):
        tasks_api.fail_interrupted_tasks_on_startup(lambda: db)

    assert task.status == TaskStatus.failed
    assert db.closed
    assert "Marked 1 interrupted task(s) as failed" in caplog.text


def test_startup_hook_swallows_db_errors(monkeypatch, caplog):
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)

    class _BrokenDb(_Db):
        def query(self, _model):
            raise RuntimeError("database not reachable")

    db = _BrokenDb([])

    with caplog.at_level(logging.ERROR, logger="api-logger"):
        tasks_api.fail_interrupted_tasks_on_startup(lambda: db)  # must not raise

    assert db.closed
    assert "Could not mark interrupted tasks as failed" in caplog.text

    def _no_session():
        raise RuntimeError("no connection")

    tasks_api.fail_interrupted_tasks_on_startup(_no_session)  # must not raise


def test_startup_hook_skips_with_several_workers(monkeypatch, caplog):
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    called = []

    with caplog.at_level(logging.WARNING, logger="api-logger"):
        tasks_api.fail_interrupted_tasks_on_startup(lambda: called.append(1))

    assert called == []
    assert "more than one worker" in caplog.text
