# Development

## Setting up environment

We use [poetry](https://python-poetry.org/docs) to manage both development environments and package building. A simple way to install it is `pipx install poetry==1.8.2`, or you can look at the installation section [here](https://python-poetry.org/docs#installation).

From the repository root folder, running any of
```bash
# Install the core library only
poetry install

# Install the core library and the tasks
poetry install -E fractal-tasks

# Install the core library and the development/documentation dependencies
poetry install --with dev --with docs
```
will take care of installing all the dependencies in a separate environment (handled by `poetry` itself), optionally installing also the dependencies for developement and to build the documentation.

## Testing

We use [pytest](https://docs.pytest.org) for unit and integration testing of Fractal. If you installed the development dependencies, you may run the test suite by invoking commands like:
```bash
# Run all tests
poetry run pytest

# Run all tests with a verbose mode, and stop at the first failure
poetry run pytest -x -v

# Run all tests and also print their output
poetry run pytest -s

# Ignore some tests folders
poetry run pytest --ignore tests/tasks
```

The tests files are in the `tests` folder of the repository. Its structure reflects the `fractal_tasks_core` structure, with tests for the core library in the main folder and tests for `tasks` and `dev` subpckages in their own subfolders.

Tests are also run through GitHub Actions, with Python 3.10, 3.11 and 3.12. Note that within GitHub actions we run tests for both the `poetry`-installed and `pip`-installed versions of the code, which may e.g. have different versions of some dependencies (since `pip install` does not rely on the `poetry.lock` lockfile).

## Documentation

The documentations is built with mkdocs.
To build the documentation locally, setup a development python environment (e.g. with `poetry install --with docs`) and then run one of these commands:
```
poetry run mkdocs serve --config-file mkdocs.yml  # serves the docs at http://127.0.0.1:8000
poetry run mkdocs build --config-file mkdocs.yml  # creates a build in the `site` folder
```

A [dedicated GitHub action](https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/main/.github/workflows/documentation.yaml) takes care of building the documentation and pushing it to https://fractal-analytics-platform.github.io/fractal-tasks-core, when commits are pushed to the `main` branch.


## Release to PyPI

### Preliminary check-list

1. The `main` branch is checked out.
2. All tests are passing, for the `main` branch.
3. `CHANGELOG.md` is up to date.
4. If appropriate (e.g. if you added some new task arguments, or if you modified some of their descriptions), update the JSON Schemas in the manifest via:
```bash
poetry run python fractal_tasks_core/dev/create_manifest.py
```
(note that the CI will fail if you forgot to update the manifest,, but it is good to be aware of it)

### Actual release

1. From within the `main` branch, use a command like:
```bash
# Automatic bump of release number
poetry run bumpver update --[tag-num|patch|minor] --dry

# Set a specific version
poetry run bumpver update --set-version 1.2.3 --dry
```
to test updating the version bump
2. If the previous step looks good, remove the `--dry` and re-run the same command. This will commit both the edited files and the new tag, and push.
3. Approve the new version deployment at [Publish package to PyPI](https://github.com/fractal-analytics-platform/fractal-tasks-core/actions/workflows/publish_pypi.yml) (or have it approved); the corresponding GitHub action will take care of running `poetry build` and `poetry publish` with the appropriate credentials.


## Static type checker

We do not enforce strict `mypy` compliance, but we do run it as part of [a specific GitHub Action](https://github.com/fractal-analytics-platform/fractal-tasks-core/actions/workflows/package.yml).
You can run `mypy` locally for instance as:
```console
poetry run mypy --package fractal_tasks_core --ignore-missing-imports --warn-redundant-casts --warn-unused-ignores --warn-unreachable --pretty
```
