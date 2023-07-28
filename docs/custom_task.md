# How to write a Fractal-compatible custom task

The `fractal-tasks-core` repository is the reference implementation for Fractal
tasks and for Fractal task packages, but the Fractal platform can also be
used to execute custom tasks. This page lists the Fractal-compatibility
requirements, for a [single custom task](#single-custom-task) or for a [task
package](#task-package).

> Note: these specifications evolve frequently, see e.g. discussion at
> https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/151.

A Fractal task is mainly formed by two components:

1. A set of metadata, which are stored in the `task` table of the database of a
   `fractal-server` instance, see [Task metadata](#task-metadata).
2. An executable command, which can take some specific command-line arguments
   (see [Command-line interface](#command-line-interface)); the standard
   example is a Python script.


In the following we explain what are the Fractal-compatibility requirements for
a single task, and then for a task package.

## Single custom task

We describe how to define the multiple aspects of a task, and provide a [Full task example](#full-task-example).

### Task metadata

Each task must be associated to some metadata, so that it can be used in
Fractal. The full specification is
[here](https://fractal-analytics-platform.github.io/fractal-server/reference/fractal_server/app/models/task/#fractal_server.app.models.task.Task),
and the required attributes are:

* `name`: the task name, e.g. `"Create OME-Zarr structure"`;
* `command`: a command that can be executed from the command line;
* `input_type`: this can be any string (typical examples: `"image"` or `"zarr"`);
  the special value `"Any"` means that Fractal won't perform any check of the
  `input_type` when applying the task to a dataset.
* `output_type`: same logic as `input_type`.
* `source`: this is meant to be as close as possible to unique task identifier;
  for custom tasks, it can be anything (e.g. `"my_task"`), but for task that
  are collected automatically from a package (see [Task package](#task-package) this
   attribute will have a very specific form (e.g.
   `"pip_remote:fractal_tasks_core:0.10.0:fractal-tasks::convert_yokogawa_to_ome-zarr"`).
* `meta`: a JSON object (similar to a Python dictionary) with some additional
  information, see [Task meta-parameters](#task-meta-parameters).

There are multiple ways to get the appropriate metadata into the database,
including a POST request to the `fractal-server` API (see `Tasks` section in
the [`fractal-server` API
documentation](https://fractal-analytics-platform.github.io/fractal-server/openapi))
or the automated addition of a whole set of tasks through specific API
endpoints (see [Task package](#task-package)).


### Command-line interface

Some examples of task commands may look like

* `python3 /some/path/my_task.py`,
* `/some/absolute/path/python3.10 /some/other/absolute/path/my_task.py`,
* `/some/path/my_executable_task.py`,
* any other executable command (not necessarily based on Python).

Given a task command, Fractal will add two additional command-line arguments to it:

* `-j /some/path/input-arguments.json`
* `--metadata-out /some/path/output-metadata-update.json`

Therefore the task command must accept these additional command-line arguments.
If the task is a Python script, this can be achieved easily by using the
`run_fractal_task` function - which is available as part of
[`fractal_tasks_core.tasks._utils`](https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/main/fractal_tasks_core/tasks/_utils.py).

### Task input parameters

**WIP**

`input_paths, output_path, and metadata, plus the optional component`
output dict[str,Any]

### Task meta-parameters

**WIP**

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

### Full task example

Here we describe a simplified example of a Fractal-compatible Python task (for
more realistic examples see the `fractal-task-core` [tasks
folder](https://github.com/fractal-analytics-platform/fractal-tasks-core/tree/main/fractal_tasks_core/tasks)).

The script `/some/path/my_task.py` may look like
```python
# Import a helper function from fractal_tasks_core
from fractal_tasks_core.tasks._utils import run_fractal_task

def my_task_function(
    # Reserved Fractal arguments
    input_paths,
    output_path,
    metadata,
    # Task-specific arguments
    argument_A,
    argument_B = "default_B_value",
):
    # Do something, based on the task parameters
    print("Here we go, we are in `my_task_function`")
    with open(f"{output_path}/output.txt", "w") as f:
        f.write(f"argument_A={argument_A}\n")
        f.write(f"argument_B={argument_B}\n")
    # Compile the output metadata update and return
    output_metadata_update = {"nothing": "to add"}
    return output_metadata_update

# Thi block is executed when running the Python script directly
if __name__ == "__main__":
    run_fractal_task(task_function=my_task_function)
```
where we use `run_fractal_task` so that we don't have to take care of the [command-line arguments](#command-line-interface).

Some valid [metadata attributes](#task-metadata) for this task would be:
```python
name="My Task"
command="python3 /some/path/my_task.py"
input_type="Any"
output_type="Any"
source="my_custom_task"
meta={}
```

> Note that this was an example of a non-parallel tasks; to have a parallel
> one, we would also need to:
>
> 1. Set `meta={"parallelization_level": "something"}`;
> 2. Include `component` in the input arguments of `my_task_function`.

## Task package

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
* Downloads the wheel file of package `MyTasks` (if it's on a public index, rather than a local file);
* Creates a Python virtual environment (venv) which is specific for a given version of the `MyTasks` package, and installs the `MyTasks` package;
* Populates all the corresponding rows in the `task` database table with the appropriate metadata, which are extracted from the package manifest.

This feature is currently exposed in the `/api/v1/task/collect/pip/` endpoint of `fractal-server` (see [API documentation](https://fractal-analytics-platform.github.io/fractal-server/openapi)).

### How

- The package is built as a a wheel file, and can be installed via `pip`.
- The `__FRACTAL_MANIFEST__.json` file is bundled in the package, at its root. If you are using `poetry`, no special operation is needed. If you are using a `setup.cfg` file, see [here](https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/151#issuecomment-1524929477).
- Include JSON Schemas. The tools in `fractal_tasks_core.dev` are used to generate JSON Schema's for the input parameters of each task in `fractal-tasks-core`. They are meant to be flexible and re-usable to perform the same operation on an independent package, but they are not thoroughly documented/tested for more general use; feel free to open an issue if something is not clear.
