"""Expose the repository's ``src`` tree as the ``patchalign3d`` package."""

from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
__path__ = [str(_SRC_DIR)]

