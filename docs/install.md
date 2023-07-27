# How to install

The `fractal_tasks_core` Python package is hosted on PyPI ([https://pypi.org/project/fractal-tasks-core](https://pypi.org/project/fractal-tasks-core)), and can be installed via `pip`.
It includes three (sub)packages:

1. The main `fractal_tasks_core` package: a set of helper functions to be used in the Fractal tasks (and possibly in other independent packages).
2. The `fractal_tasks_core.tasks` subpackage: a set of standard Fractal tasks.
3. The `fractal_tasks_core.dev` subpackage: a set of developement tools (mostly related to creation of JSON Schemas for task arguments).


## Minimal installation

The minimal installation command is
```console
pip install fractal-tasks-core
```
which only installs the dependencies necessary for the main package and for the `dev` subpackage.

## Full installation

In order to also use the `tasks` subpackage, the additional extra `fractal-tasks` must be included, as in
```console
pip install fractal-tasks-core[fractal-tasks]
```
**Warning**: This command installs heavier dependencies (e.g. `torch`).
