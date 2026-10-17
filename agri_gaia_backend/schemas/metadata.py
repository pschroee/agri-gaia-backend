# SPDX-FileCopyrightText: 2026 Philipp Schröer
#
# SPDX-License-Identifier: MIT

from typing import Any, Dict

from pydantic import BaseModel, Field


class MetadataDocument(BaseModel):
    """Description and metadata of an asset, read from the triple store (X4, #146).

    The fixture verification of the experiment runner needs both to compare what was
    stored with what the manifest asked for. Only the owner may read them.
    """

    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
