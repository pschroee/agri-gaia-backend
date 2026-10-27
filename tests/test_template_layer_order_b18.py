# SPDX-FileCopyrightText: 2026 Philipp Schröer
#
# SPDX-License-Identifier: MIT

"""B18/T1: layer order of the training template Dockerfiles.

Before the last RUN instruction (the heavy pip layers) only the entrypoint
scripts and requirements.txt may be copied. Code and the per-build
configuration written by build_train_image follow it, so a changed
train_config.json never invalidates the torch layer.

Pure file checks, no infrastructure. Run with `--noconftest`.
"""

import glob
import os
import re

import pytest

TEMPLATES = os.path.join(os.path.dirname(__file__), "..", "model-training", "templates")
DOCKERFILES = sorted(
    glob.glob(os.path.join(TEMPLATES, "*", "*", "*", "docker", "Dockerfile"))
)
CONFIG_FILES = {"train_config.json", "export_config.json", "presets.json", "custom.json"}
ALLOWED_BEFORE_LAST_RUN = {"docker-entrypoint.sh", "mount-s3.sh", "requirements.txt"}


def _instructions(path):
    """Returns [(keyword, argument string)] with line continuations joined."""
    with open(path) as fh:
        text = fh.read()
    text = re.sub(r"\\\n", " ", text)
    result = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, rest = line.partition(" ")
        result.append((keyword.upper(), rest.strip()))
    return result


def _copy_sources(argument):
    tokens = [t for t in argument.split() if not t.startswith("--")]
    return tokens[:-1]


def _name(path):
    return os.path.relpath(path, TEMPLATES)


def _last_run(instructions):
    return max(i for i, (k, _) in enumerate(instructions) if k == "RUN")


def test_four_templates_found():
    assert len(DOCKERFILES) == 4


@pytest.mark.parametrize("path", DOCKERFILES, ids=_name)
def test_only_requirements_copied_before_last_run(path):
    instructions = _instructions(path)
    for keyword, argument in instructions[: _last_run(instructions)]:
        if keyword == "COPY":
            extra = set(_copy_sources(argument)) - ALLOWED_BEFORE_LAST_RUN
            assert not extra, f"{_name(path)}: COPY before the last RUN: {sorted(extra)}"


@pytest.mark.parametrize("path", DOCKERFILES, ids=_name)
def test_requirements_copied_before_pip_install(path):
    instructions = _instructions(path)
    copy_index = next(
        i
        for i, (k, a) in enumerate(instructions)
        if k == "COPY" and "requirements.txt" in _copy_sources(a)
    )
    pip_index = next(
        i for i, (k, a) in enumerate(instructions) if k == "RUN" and "-r requirements.txt" in a
    )
    assert copy_index < pip_index


@pytest.mark.parametrize("path", DOCKERFILES, ids=_name)
def test_config_files_after_last_run_and_last_before_env(path):
    instructions = _instructions(path)
    config_indexes = [
        i
        for i, (k, a) in enumerate(instructions)
        if k == "COPY" and CONFIG_FILES & set(_copy_sources(a))
    ]
    assert config_indexes, f"{_name(path)}: no config COPY found"
    assert min(config_indexes) > _last_run(instructions)
    # The configuration is what changes per build: only COPY .env, ENTRYPOINT
    # and CMD may follow it.
    tail = instructions[max(config_indexes) + 1 :]
    assert all(k in ("COPY", "ENTRYPOINT", "CMD") for k, _ in tail)
    assert [a for k, a in tail if k == "COPY"] == [".env ./"]


@pytest.mark.parametrize("path", DOCKERFILES, ids=_name)
def test_config_files_are_a_separate_copy_from_code(path):
    """Code changes must not rebuild the configuration layer."""
    for keyword, argument in _instructions(path):
        if keyword == "COPY":
            sources = set(_copy_sources(argument))
            if sources & CONFIG_FILES:
                assert sources <= CONFIG_FILES, f"{_name(path)}: config mixed with code"


@pytest.mark.parametrize("path", DOCKERFILES, ids=_name)
def test_workdir_cleanup_stays_before_pip(path):
    instructions = _instructions(path)
    pip_index = next(
        i for i, (k, a) in enumerate(instructions) if k == "RUN" and "pip3 install" in a
    )
    for i, (k, a) in enumerate(instructions):
        if k == "RUN" and (a.startswith("rm -rf") or a.startswith("mkdir")):
            assert i < pip_index
