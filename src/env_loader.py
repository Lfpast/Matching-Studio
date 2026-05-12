from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / "config" / ".env"


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_project_env(env_path: Path | None = None) -> None:
    target_path = env_path or DEFAULT_ENV_PATH
    if not target_path.exists() or not target_path.is_file():
        return

    for raw_line in target_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        env_key = key.strip()
        if not env_key:
            continue

        env_value = _strip_wrapping_quotes(value.strip())
        os.environ.setdefault(env_key, env_value)