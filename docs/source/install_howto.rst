How to install
==============

The ``fractal_tasks_core`` package is hosted on PyPI (https://pypi.org/project/fractal-tasks-core) and it includes:

* The main ``fractal_tasks_core`` package, which includes several helper functions to be used in the Fractal tasks (and possibly in other independent packages). This can be installed as::

    pip install fractal-tasks-core

* The ``fractal_tasks_core.tasks`` subpackage, which inlcudes some standard Fractal tasks. This subpackage requires additional dependencies, which are installed as::

    pip install fractal-tasks-core[fractal-tasks]

* The ``fractal_tasks_core.dev`` subpackage, which includes some developement tools, mostly related to creation of JSON Schemas for task arguments.


**Note**: this package is tested with python version 3.9 and 3.10.
