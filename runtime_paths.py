"""Runtime path helpers for source and PyInstaller builds."""

from __future__ import annotations

import sys
import shutil
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_base_dir() -> Path:
    """Directory for writable runtime files and bundled sidecar executables."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return SOURCE_DIR


def resource_base_dir() -> Path:
    """Directory containing bundled read-only resources."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", app_base_dir())).resolve()
    return SOURCE_DIR


def resource_path(name: str) -> Path:
    return resource_base_dir() / name


def writable_path(name: str) -> Path:
    return app_base_dir() / name


def editable_resource_path(name: str) -> Path:
    target = writable_path(name)
    if target.exists():
        return target

    source = resource_path(name)
    if is_frozen() and source.exists():
        try:
            shutil.copy2(source, target)
            return target
        except OSError:
            return source

    return source


def bundled_executable_path(stem: str) -> Path | None:
    if not is_frozen():
        return None

    executable_name = f"{stem}.exe" if sys.platform.startswith("win") else stem
    candidate = app_base_dir() / executable_name
    return candidate if candidate.exists() else None
