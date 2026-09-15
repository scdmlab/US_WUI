#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
"""Shared path configuration for the repository scripts.

Copy ``config/paths.example.json`` to ``config/paths.json`` and edit only that
file for a new computer.  The configuration file may also be selected with
the ``WUI_CONFIG_FILE`` environment variable.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = REPOSITORY_ROOT / "config" / "paths.json"
SELECTED_CONFIG_FILE = Path(os.environ.get("WUI_CONFIG_FILE", DEFAULT_CONFIG_FILE))


def _read_config() -> dict[str, Any]:
    if not SELECTED_CONFIG_FILE.exists():
        return {}
    with SELECTED_CONFIG_FILE.open("r", encoding="utf-8") as stream:
        content = json.load(stream)
    if not isinstance(content, dict):
        raise ValueError(
            f"Configuration root must be an object: {SELECTED_CONFIG_FILE}"
        )
    return content


CONFIG = _read_config()


def _configured_root(scope: str) -> Path:
    section = CONFIG.get(scope, {})
    configured = section.get("root") if isinstance(section, dict) else None
    if configured:
        value = Path(os.path.expandvars(os.path.expanduser(str(configured))))
        if not value.is_absolute():
            value = REPOSITORY_ROOT / value
        return value

    defaults = {
        "project": REPOSITORY_ROOT,
        "data": REPOSITORY_ROOT / "data",
        "legacy": REPOSITORY_ROOT / "legacy_data",
        "attachments": REPOSITORY_ROOT / "attachments",
        "researchdrive": REPOSITORY_ROOT / "external_data",
        "software": Path(os.environ.get("CONDA_PREFIX", sys.prefix)),
        "temp": Path(tempfile.gettempdir()),
    }
    try:
        return defaults[scope]
    except KeyError as exc:
        raise KeyError(f"Unknown path scope: {scope}") from exc


def portable_path(scope: str, relative: str = "") -> str:
    """Return a configured path as a string for old and new APIs."""

    base = _configured_root(scope)
    return str(base / relative) if relative else str(base)


PROJECT_ROOT = Path(portable_path("project"))
DATA_ROOT = Path(portable_path("data"))
