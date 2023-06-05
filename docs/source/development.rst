Development
===========


Setting up environment
~~~~~~~~~~~~~~~~~~~~~~

We use `poetry <https://python-poetry.org/docs>`_ v1.5 to manage the development environment and the dependencies. A simple way to install it is ``pipx install poetry==1.5``, or you can look at the installation section `here <https://python-poetry.org/docs#installation>`_.
Running::

    poetry install [--with dev] [--with docs]

will take care of installing all the dependencies in a separate environment, optionally installing also the dependencies for developement and to build the documentation.

Testing
~~~~~~~

We use `pytest <https://docs.pytest.org>`_ for unit and integration testing of Fractal. If you installed the development dependencies, you may run the test suite by invoking::

    poetry run pytest

The tests files are in the ``tests`` folder of the repository, and they are also run on GitHub (with both python 3.8 and 3.9 versions).

How to release
~~~~~~~~~~~~~~

Preliminary checklist
^^^^^^^^^^^^^^^^^^^^^

1. The main branch is checked out.
2. You reviewed dependencies and dev dependencies and the lock file is up to date with ``pyproject.toml`` (it is useful to have a look at the output of ``deptry . -v``, where ``deptry`` is already installed as part of the dev dependencies).
3. The current HEAD of the main branch passes all the tests (note: make sure that you are using the poetry-installed local package).
4. Update changelog. First look at the list of commits since the last tag, via::

    git log --pretty="[%cs] %h - %s" `git tag --sort version:refname | tail -n 1`..HEAD

  then add the upcoming release to ``docs/source/changelog.rst`` with the main information about it, using standard categories like "New features", "Fixes" and "Other changes", and including PR numbers when relevant. Commit ``docs/source/changelog.rst`` and push.

Actual release
^^^^^^^^^^^^^^

5. Use::

    poetry run bumpver update --[tag-num|patch|minor] --tag-commit --commit --dry

  to test updating the version bump.

6. If the previous step looks good, use::

    poetry run bumpver update --[tag-num|patch|minor] --tag-commit --commit

  to actually bump the version and commit the changes locally.

7. Test the build with::

    poetry build

8. If the previous step was successful, push the version bump and tags::

    git push && git push --tags

9. Finally, publish the updated package to PyPI with::

    poetry publish --dry-run

  replacing ``--dry-run`` with ``--username YOUR_USERNAME --password YOUR_PASSWORD`` when you made sure that everything looks good.
