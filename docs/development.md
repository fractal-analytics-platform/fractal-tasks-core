# Development

## Setting up environment

We use [poetry](https://python-poetry.org/docs) v1.5 to manage the development environment and the dependencies. A simple way to install it is `pipx install poetry==1.5`, or you can look at the installation section [here](https://python-poetry.org/docs#installation).

Running:

```console
poetry install [--with dev] [--with docs]
```

will take care of installing all the dependencies in a separate environment, optionally installing also the dependencies for developement and to build the documentation.

## Testing

We use [pytest](https://docs.pytest.org) for unit and integration testing of Fractal. If you installed the development dependencies, you may run the test suite by invoking:
```console
poetry run pytest
```

The tests files are in the `tests` folder of the repository, and they are also
run through GitHub Actions; both the main _fractal_tasks_core_ tests (in
`tests/`) and the _fractal_tasks_core.tasks_ tests (in `tests/tasks/`) are run
with Python 3.9, 3.10 and 3.11.

## Documentation

The documentations is built with mkdocs.
To build the documentation locally, setup a development python environment (e.g. with `poetry install --with docs`) and then run one of these commands:
```
poetry run mkdocs serve --config-file mkdocs.yml  # serves the docs at http://127.0.0.1:8000
poetry run mkdocs build --config-file mkdocs.yml  # creates a build in the `site` folder
```

## Mypy

We do not enforce strict `mypy` compliance, but we do run it as part of [a specific GitHub Action](https://github.com/fractal-analytics-platform/fractal-tasks-core/actions/workflows/package.yml).
You can run `mypy` locally for instance as:
```console
poetry run mypy --package fractal_tasks_core --ignore-missing-imports --warn-redundant-casts --warn-unused-ignores --warn-unreachable --pretty
```

## How to release

### Preliminary checklist

1. The main branch is checked out.
2. You reviewed dependencies and dev dependencies and the lock file is up to date with `pyproject.toml` (it is useful to have a look at the output of `deptry . -v`, where `deptry` is already installed as part of the dev dependencies - NOTE: `deptry` should be installed independently, e.g. via `pipx install deptry`).
3. The current HEAD of the main branch passes all the tests (note: make sure that you are using the poetry-installed local package).
4. Update changelog. First look at the list of commits since the last tag, via:
```console
git log --pretty="[%cs] %h - %s" `git tag --sort version:refname | tail -n 1`..HEAD
```
then add the upcoming release to `docs/source/changelog.rst` with the main information about it, using standard categories like "New features", "Fixes" and "Other changes", and including PR numbers when relevant. Commit `docs/source/changelog.rst` and push.
5. If appropriate (e.g. if you added some new task arguments, or if you modified some of their descriptions), update the JSON Schemas in the manifest via:
```console
poetry run python fractal_tasks_core/dev/create_args_schemas.py
```

### Actual release

6. Use:
```console
poetry run bumpver update --[tag-num|patch|minor] --tag-commit --commit --dry
```
to test updating the version bump.
7. If the previous step looks good, use:
```console
poetry run bumpver update --[tag-num|patch|minor] --tag-commit --commit
```
to actually bump the version and commit the changes locally.
8. Test the build with:
```console
poetry build
```
9. Finally, publish the updated package to PyPI with::
```console
poetry publish --dry-run
```
replacing `--dry-run` with `--username YOUR_USERNAME --password YOUR_PASSWORD` when you made sure that everything looks good.
