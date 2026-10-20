# SPDX-FileCopyrightText: 2026 Philipp Schröer
#
# SPDX-License-Identifier: MIT

"""Parsing of the figures a training container writes (metrics.jsonl and gpu.csv).

Pure functions without Docker access, so the train router and the tests share them.
"""

import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

METRICS_LABEL = "agrigaia.metrics"
METRICS_LABEL_VALUE = "jsonl/1"
MAX_LINES = 2000
# The training images set TZ=Europe/Berlin, and nvidia-smi prints local time.
CONTAINER_TIMEZONE = ZoneInfo("Europe/Berlin")


def metrics_supported(labels: Optional[Dict[str, str]]) -> bool:
    return (labels or {}).get(METRICS_LABEL) == METRICS_LABEL_VALUE


def parse_metrics_jsonl(data: Optional[bytes], limit: int = MAX_LINES) -> List[Dict]:
    """The last `limit` lines that are JSON objects; broken lines are skipped."""
    if not data:
        return []
    rows = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows[-limit:]


def _number(text: str) -> Optional[float]:
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_gpu_csv(data: Optional[bytes], limit: int = MAX_LINES) -> List[Dict]:
    """Lines `timestamp, util %, mem used MiB, mem total MiB` (nvidia-smi csv,noheader,nounits)."""
    if not data:
        return []
    rows = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        fields = line.split(",")
        if len(fields) != 4:
            continue
        util, used, total = (_number(f) for f in fields[1:])
        at = _timestamp(fields[0])
        if at is None or util is None or used is None or total is None:
            continue
        rows.append(
            {"at": at, "util": util, "mem_used_mb": used, "mem_total_mb": total}
        )
    return rows[-limit:]


def _timestamp(text: str) -> Optional[str]:
    text = text.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            local = datetime.strptime(text, fmt).replace(tzinfo=CONTAINER_TIMEZONE)
        except ValueError:
            continue
        return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return None
