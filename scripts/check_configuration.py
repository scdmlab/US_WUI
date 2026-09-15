#!/usr/bin/env python3
# Copyright (C) 2026 WUI Mapping Project Authors
# SPDX-License-Identifier: GPL-3.0-only
# Repository version: filesystem paths are loaded through repo_config.py.
# Copy config/paths.example.json to config/paths.json before a new run.
# Raw input data and large intermediate files are stored outside GitHub.

"""Display the configured roots and report whether they currently exist."""

from __future__ import annotations

import argparse
from pathlib import Path

from repo_config import SELECTED_CONFIG_FILE, portable_path


SCOPES = (
    "project",
    "data",
    "legacy",
    "attachments",
    "researchdrive",
    "software",
    "temp",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-existing",
        action="store_true",
        help="Return a nonzero status when any configured root is missing.",
    )
    args = parser.parse_args()

    print(f"Configuration file: {SELECTED_CONFIG_FILE}")
    print(f"Configuration exists: {SELECTED_CONFIG_FILE.exists()}")

    missing = []
    for scope in SCOPES:
        path = Path(portable_path(scope))
        exists = path.exists()
        print(f"{scope:14s} {path}  exists={exists}")
        if not exists:
            missing.append(scope)

    if args.require_existing and missing:
        print("Missing configured roots: " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
