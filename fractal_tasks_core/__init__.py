"""Reference task package for the Fractal analytics platform."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fractal-tasks-core")
except PackageNotFoundError:
    __version__ = "uninstalled"
