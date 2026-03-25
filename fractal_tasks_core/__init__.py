# Copyright 2022-2026 (C) BioVisionCenter, University of Zurich
"""Reference task package for the Fractal analytics platform."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fractal-tasks-core")
except PackageNotFoundError:
    __version__ = "uninstalled"
