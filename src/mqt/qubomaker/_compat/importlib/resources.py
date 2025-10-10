from __future__ import annotations

from importlib.resources import as_file, files

__all__ = ["as_file", "files"]


def __dir__() -> list[str]:
    return __all__
