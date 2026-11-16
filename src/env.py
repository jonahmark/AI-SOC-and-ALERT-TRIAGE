"""Project-root `.env` defaults, shared by every settings module.

Real environment variables always win: a KEY=VALUE line only fills a name
that is not already set. Values are never logged or echoed.
"""
from __future__ import annotations

import os

import config


def load_dotenv_defaults() -> None:
    path = config.PROJECT_ROOT / ".env"
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value