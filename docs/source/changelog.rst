Changelog
=========

.. note::

   Numbers like (#123) point to `closed Pull Requests on the fractal-tasks-core repository <https://github.com/fractal-analytics-platform/fractal-tasks-core/pulls?q=is%3Apr+is%3Aclosed>`_.


0.6.4
-----

Other changes
~~~~~~~~~~~~~
* Created ``tools`` submodule and installation extra (#262).


0.6.3
-----

Other changes
~~~~~~~~~~~~~
* Added napari dependency, pinned to 0.4.16 version.
* Fixed type-hinting bug in task to create multiplexing OME-Zarr structure (#258).


0.6.2
-----

New features
~~~~~~~~~~~~

* Support passing a pre-made metadata table to tasks creating the OME-Zarr structure (#252).


0.6.1
-----

New features
~~~~~~~~~~~~
* Add option for padding an array with zeros in ``upscale_array`` (#251).


Other changes
~~~~~~~~~~~~~
* Simplified ``imagecodecs`` and ``PyQt5`` dependencies (#248).


0.6.0
-----

New features
~~~~~~~~~~~~
* **(major)** Refactor of how to address channels (#239).

Other changes
~~~~~~~~~~~~~
* Fix bug in well ROI table (#245).

0.5.1
-----

New features
~~~~~~~~~~~~
* Fix sorting of image files when number of Z planes passes 100 (#237).


0.5.0
-----

New features
~~~~~~~~~~~~
* **(major)** Deprecate ``measurement`` task (#235).
* **(major)** Use more uniform names for tasks, both in python modules and manifest (#235).

Other changes
~~~~~~~~~~~~~
* Remove deprecated manifest from ``__init__.py`` (#233).

0.4.6
-----

Other changes
~~~~~~~~~~~~~
* Skip image files if filename is not parsable (#219).
* Preserve order of ``input_paths`` for multiplexing subfolders (#222).
* Major refactor of ``replicate_zarr_structure``, also enabling support for zarr files with multiple images (#223).


0.4.5
-----

New features
~~~~~~~~~~~~
* Replace ``Cellpose`` wrapper with ``CellposeModel``, to support ``pretrained_model`` argument (#218).
* Update cellpose version (it was pinned to 2.0, in previous versions) (#218).
* Pin ``torch`` dependency to version 1.12.1, to support CUDA version 10.2 (#218).

0.4.4
-----

Missing due to releasing error.

0.4.3
-----

Other changes
~~~~~~~~~~~~~
* In ``create_zarr_structure_multiplex``, always use/require strings for ``acquisition`` field (#217).


0.4.2
-----

Other changes
~~~~~~~~~~~~~
* Bugfixes


0.4.1
-----

Other changes
~~~~~~~~~~~~~
* Only use strings as keys of ``channel_parameters`` (in ``create_zarr_structure_multiplex``).


0.4.0
-----

New features
~~~~~~~~~~~~
* **(major)** Rename ``well`` to ``image`` (both in metadata list and in manifest) and add an actual ``well`` field (#210).
* Add ``create_ome_zarr_multiplexing``, and adapt ``yokogawa_to_zarr`` (#210).

Other changes
~~~~~~~~~~~~~
* Relax constraint about outputs in ``napari_worfklows_wrapper`` (#209).


0.3.4
-----

Other changes
~~~~~~~~~~~~~
* Always log START/END times for each task (#204).
* Add ``label_name`` argument to ``cellpose_segmentation`` (#207).
* Add ``pretrained_model`` argument to ``cellpose_segmentation`` (#207).

0.3.3
-----

New features
~~~~~~~~~~~~
* Added ``napari_worfklows_wrapper`` to manifest.


0.3.2
-----

New features
~~~~~~~~~~~~
* Compute bounding boxes of labels, in ``cellpose_segmentation`` (#192).

Other changes
~~~~~~~~~~~~~
* Parse image filenames in a more robust way (#191).
* Update manifest, moving ``parallelization_level`` and ``executor`` to ``meta`` attribute.

0.3.1
-----

Other changes
~~~~~~~~~~~~~
* Fix ``executable`` fields in manifest.
* Remove ``graphviz`` dependency.


0.3.0
-----

New features
~~~~~~~~~~~~
* Conform to Fractal v1, through new task manifest (#162) and standard input/output interface (#155, #157).
* Add several type hints (#148) and validate them in the standard task interface (#175).
* Update ``napari_worfklows_wrapper``: pyramid level for labeling worfklows (#148), label-only inputs (#163, #171), relabeling (#167), 2D/3D handling (#166).

Other changes
~~~~~~~~~~~~~
* Deprecate ``dummy`` and ``dummy_fail`` tasks.

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
