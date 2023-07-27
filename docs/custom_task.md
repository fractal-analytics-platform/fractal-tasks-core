# How to write a Fractal-compatible custom task

The `fractal-tasks-core` repository constitutes the reference for Fractal tasks
and for packages of Fractal tasks, but the Fractal platform can also be used to
execute custom (Fractal-compatible) tasks. This page lists the
Fractal-compatibility requirements for a single custom task or for a task
package.

> Note that these specifications evolve frequently, see e.g. discussion at
> https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/151.


A Fractal task is mainly formed by two compontens:

1. An executable command, which can take some specific command-line arguments
   (see [below](#command-line-interface)). The standard example is a Python
   script, that uses a specific set of arguments.
2. A set of metadata, which are stored in the `task` table of the database of a
   `fractal-server` instance (see relevant definition
   [here](https://fractal-analytics-platform.github.io/fractal-server/reference/fractal_server/app/models/task/#fractal_server.app.models.task.Task)).
   There are multiple ways to get the appropriate metadata into the database,
   including a POST request to the `fractal-server` API or a the automated
   addition of a whole set of tasks through specific API endpoints (see
   [below](#custom-task-package)).

In the following we explain what are the Fractal-compatibility requirements for
a single task, and then for a task package.

## Single custom task

----

# FIXME CONTINUE FROM HERE

----

### Command-line interface

A task is assoicThe executable command `command`  should
```
        f"{wftask.task.command} -j {task_files.args} "
        f"--metadata-out {task_files.metadiff}"
```

Use `run_fractal_task` helper function

The `run_fractal_task` function is available as part of [`fractal_tasks_core.tasks._utils`](https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/main/fractal_tasks_core/tasks/_utils.py).

Fully-custom executable



### Task input parameters

`input_paths, output_paths, and metadata, plus the optional component`
output dict[str,Any]

### Task meta-parameters

In the simplest example, a task will run

parallel or not

### Advanced features

The description of some more advanced features is not yet available in this
page.

1. There exist other attributes that can be included in the task metadata in
   the database, and will be recognized by other Fractal components (e.g.
   `fractal-server` or `fractal-web`). These include JSON Schemas for input
   parameters and additional documentation-related attributes.
2. In `fractal-tasks-core`, we use [`pydantic
   v1`](https://docs.pydantic.dev/1.10) to fully coerce and validate the input
   parameters into a set of given types.

Moreover


### Full example

Here is a somewhat artificial example of a Python task (based on those in the
[`fractal-demos`
repository](https://github.com/fractal-analytics-platform/fractal-demos/tree/main/examples/99_stress_test/fractal-tasks-stresstest/fractal_tasks_stresstest)):

```python

import logging
from typing import Any
from typing import dict
from typing import Optional
from typing import Sequence


logger = logging.getLogger(__name__)


def prepare_metadata(
    *,
    input_paths: Sequence[str],
    output_path: str,
    metadata: Dict[str, Any],
    # Task-specific arguments
    num_components: int = 10,
) -> Dict[str, Any]:
    """
    Prepare metadata

    :param num_components: Number of components
    """

    list_components = [f"{ind:04d}" for ind in range(num_components)]
    metadata_update = {"component": list_components}
    logger.info(f"This is a log from prepare_metadata, with {num_components=}")
    return metadata_update


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=prepare_metadata)
```


## Custom task package

Given a set of Python scripts corresponding to Fractal tasks, it is often a good practice to combine them into a single Python package, using the [standard tools](https://packaging.python.org/en/latest/tutorials/packaging-projects) or other options (e.g. for `fractal-tasks-core` we use [poetry](https://python-poetry.org/)).

Some reasons are unrelated to Fractal:
* Avoid duplication:
    * The different scripts may depend on a shared set of external packages, which only need to be defined once when packaging.
    * The difference scripts may use a shared set of helper functions, which can be included in the package.
* It is simple to assign a global version to the package;

(possibly using some common external dependencies, and possibly sharing some common helper functions), it is often good practice to transform them into a built package.using

Building a package for a set of task Python

After creating a bunch of Fractal-compatible tasks, it's useful to bundle them
up in a single package (possibly hosted on a public index like PyPI). If this is done for a package `MyTasks`, then the Fractal platform (i.e. the server and the command-line/web clients) offers a feature that automatically:
* Downloads the wheel file of packge `MyTasks` (if it's on a public index, rather than a local file);
* Creates a Python virtual environment (venv) which is specific for a given version of the `MyTasks` package, and installs the `MyTasks` package;
* Populates all the corresponding rows in the `task` database table with the appropriate metadata, which are extracted from the package manifest.

This feature is currently exposed in the `/api/v1/task/collect/pip/` endpoint of `fractal-server` (see [API documentation](https://fractal-analytics-platform.github.io/fractal-server/openapi)).

### How

- The package is built as a a wheel file, and can be installed via `pip`.
- The `__FRACTAL_MANIFEST__.json` file is bundled in the package, at its root. If you are using `poetry`, no special operation is needed. If you are using a `setup.cfg` file, see [here](https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/151#issuecomment-1524929477).
- Include JSON Schemas. The tools in `fractal_tasks_core.dev` are used to generate JSON Schema's for the input parameters of each task in `fractal-tasks-core`. They are meant to be flexible and re-usable to perform the same operation on an independent package, but they are not thoroughly documented/tested for more general use; feel free to open an issue if something is not clear.
