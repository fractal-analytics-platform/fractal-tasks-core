# Development

## Setting up environment

We use [pixi](https://pixi.sh) to manage development environments. See the [pixi installation guide](https://pixi.sh/latest/#installation) for setup instructions.

From the repository root folder, run:
```bash
# Install the default environment
pixi install

# Install the development environment (includes test dependencies)
pixi install -e dev

# Install the documentation environment
pixi install -e docs
```

## Testing

We use [pytest](https://docs.pytest.org) for unit and integration testing. If you installed the development environment, you may run the test suite by invoking commands like:
```bash
# Run all tests
pixi run -e dev pytest

# Run all tests with a verbose mode, and stop at the first failure
pixi run -e dev pytest -x -v

# Run all tests and also print their output
pixi run -e dev pytest -s
```

The test files are in the `tests` folder of the repository.

Tests are also run through GitHub Actions, with Python 3.11 and 3.12. Within GitHub Actions we run tests for both the `pixi`-installed and `pip`-installed versions of the code, which may have different versions of some dependencies.

## Documentation

The documentation is built with mkdocs.
To build the documentation locally, setup the docs environment and then run one of these commands:
```bash
pixi run -e docs mkdocs serve --config-file mkdocs.yml  # serves the docs at http://127.0.0.1:8000
pixi run -e docs mkdocs build --config-file mkdocs.yml  # creates a build in the `site` folder
```

A [dedicated GitHub action](https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/main/.github/workflows/documentation.yaml) takes care of building the documentation and pushing it to https://fractal-analytics-platform.github.io/fractal-tasks-core, when commits are pushed to the `main` branch.


## Release to PyPI

### Preliminary check-list

1. The `main` branch is checked out.
2. All tests are passing, for the `main` branch.
3. `CHANGELOG.md` is up to date.
4. If appropriate (e.g. if you added some new task arguments, or if you modified some of their descriptions), update the manifest via:
```bash
pixi run -e dev fractal-manifest create --package fractal_tasks_core
```
(note that the CI will fail if you forgot to update the manifest, but it is good to be aware of it)

### Actual release

1. From within the `main` branch, use a command like:
```bash
# Automatic bump of release number
pixi run -e dev bumpver update --[tag-num|patch|minor] --dry

# Set a specific version
pixi run -e dev bumpver update --set-version 1.2.3 --dry
```
to test updating the version bump
2. If the previous step looks good, remove the `--dry` and re-run the same command. This will commit both the edited files and the new tag, and push.
3. Approve the new version deployment at [Publish package to PyPI](https://github.com/fractal-analytics-platform/fractal-tasks-core/actions/workflows/publish_pypi.yml) (or have it approved); the corresponding GitHub action will take care of building and publishing the package with the appropriate credentials.


## Static type checker

We do not enforce strict `mypy` compliance, but we do run it as part of [a specific GitHub Action](https://github.com/fractal-analytics-platform/fractal-tasks-core/actions/workflows/package.yml).
You can run `mypy` locally for instance as:
```console
pixi run -e dev mypy --package fractal_tasks_core --ignore-missing-imports --warn-redundant-casts --warn-unused-ignores --warn-unreachable --pretty
```
