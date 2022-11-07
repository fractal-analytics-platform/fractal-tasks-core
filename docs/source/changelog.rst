Changelog
=========

0.3.0
-----

New features
~~~~~~~~~~~~
* Conform to Fractal v1, through new task manifest (#162) and standard input/output interface (#155, #157).
* Add several type hints (#148) and validate them in the standard task interface (#175).
* Update ``napari_worfklow_wrapper``: pyramid level for labeling worfklows (#148), label-only inputs (#163, #171), relabeling (#167), 2D/3D handling (#166).

Other changes
~~~~~~~~~~~~~
* Deprecate `dummy` and `dummy_fail` tasks.

0.2.6
-----

New features
~~~~~~~~~~~~
* Setup sphinx docs, to be built and hosted on https://fractal-tasks-core.readthedocs.io; include some preliminary updates of docstrings (#143).

Other changes
~~~~~~~~~~~~~
* Dependency cleanup via deptry (#144).

0.2.5
-----

New features
~~~~~~~~~~~~
* Add ``napari_workflows_wrapper`` task (#141).
* Add ``lib_upscale_array.py`` module (#141).

0.2.4
-----

New features
~~~~~~~~~~~~
* Major updates to ``metadata_parsing.py`` (#136).
