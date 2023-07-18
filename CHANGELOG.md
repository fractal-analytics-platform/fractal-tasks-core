
* Update dependencies (\#455):
    * Relax `dask` required version to `>=2023.1.0`;
    * Relax `zarr` required version to `>=2.13.6,<3`;
    * Relax `pandas` required version to `>=1.2.0,<2`;
    * Relax `Pillow` required version to `>=9.1.1,<10.0.0`;
* Update default values for tolerance (`tol`) in `lib_ROI_overlaps.py` functions (\#466).
* Internal changes (\#455):
    * Improve coverage of core library (\#459, \#467).
    * Update Zenodo datasets used in tests (\#454).
    * Update the `poetry.lock` version of several dependencies (`cellpose`, `dask`, `napari-skimage-regionprops`, `Pillow`, `scikit-image`, `zarr`, `lxml`, `pydantic`).
    * Include `requests` and `wget` in the `dev` poetry dependency group.
    * Run tests both for the poetry- and pip-installed package.
    * Update generic type hints (\#462).

# 0.10.0

* Restructure the package and repository:
    * Move tasks to `tasks` subpackage (\#390)
    * Create new `dev` subpackage (\#384).
    * Make tasks-related dependencies optional, and installable via `fractal-tasks` extra (\#390).
    * Remove `tools` package extra (\#384), and split the subpackage content into `lib_ROI_overlaps` and `examples` (\#390).
* **(major)** Modify task arguments
    * Add Pydantic model `lib_channels.OmeroChannel` (\#410, \#422);
    * Add Pydantic model `tasks._input_models.Channel` (\#422);
    * Add Pydantic model `tasks._input_models.NapariWorkflowsInput` (\#422);
    * Add Pydantic model `tasks._input_models.NapariWorkflowsOutput` (\#422);
    * Move all Pydantic models to main package (\#438).
    * Modify arguments of `illumination_correction` task (\#431);
    * Modify arguments of `create_ome_zarr` and `create_ome_zarr_multiplex` (\#433).
    * Modify argument default for `ROI_table_names`, in `copy_ome_zarr` (\#449).
    * Remove the delete option from yokogawa to ome zarr (\#443).
    * Reorder task inputs (\#451).
* JSON Schemas for task arguments:
    * Add JSON Schemas for task arguments in the package manifest (\#369, \#384).
    * Add JSON Schemas for attributes of custom task-argument Pydantic models (\#436).
    * Make schema-generation tools more general, when handling custom Pydantic models (\#445).
    * Include titles for custom-model-typed arguments and argument attributes (\#447).
    * Remove `TaskArguments` models and switch to Pydantic V1 `validate_arguments` (\#369).
    * Make coercing&validating task arguments required, rather than optional (\#408).
    * Remove `default_args` from manifest (\#379, \#393).
* Other:
    * Make pydantic dependency required for running tasks, and pin it to V1 (\#408).
    * Remove legacy executor definitions from manifest (\#361).
    * Add GitHub action for testing `pip install` with/without `fractal-tasks` extra (\#390).
    * Remove `sqlmodel` from dev dependencies (\#374).
    * Relax constraint on torch version, from `==1.12.1` to `<=2.0.0` (\#406).
    * Review task docstrings and improve documentation (\#413, \#416).
    * Update `anndata` dependency requirements (from `^0.8.0` to `>=0.8.0,<=0.9.1`), and replace `anndata.experimental.write_elem` with `anndata._io.specs.write_elem` (\#428).

# 0.9.4

* Relax constraint on `scikit-image` version, by only requiring a version `>=0.19` (\#367).

# 0.9.3

* For labeling tasks (`cellpose_segmentation` or `napari_worfklows_wrapper`), allow empty ROI tables as input or output (\#365).
* Relax constraint related to the presence of channels in `create_zarr_structure_multiplex` task (\#365).

# 0.9.2

* Increase memory requirements for some tasks in manifest (\#363).

# 0.9.1

* Add `use_gpu` argument for `cellpose_segmentation` task (\#350).
* Add dummy return object to napari-workflows task (\#359).
* Include memory/cpu/gpu requirements in manifest, in view of new fractal-server SLURM backend (\#360).

# 0.9.0

* Introduce a module for masked loading of ROIs, and update the `cellpose_segmentation` task accordingly (\#306).
* Rename task arguments: `ROI_table_name->input_ROI_table` and `bounding_box_ROI_table_name->output_ROI_table` (\#306).
* Implement part of the [proposed table support in OME-NGFF specs](https://github.com/ome/ngff/pull/64), both for the `tables` zarr group and then for each table subgroup (\#306).
* Rename module: `lib_remove_FOV_overlaps.py->lib_ROI_overlaps.py` (\#306).
* Add new functions to existing modules: `lib_regions_of_interest.convert_region_to_low_res`, `lib_ROI_overlaps.find_overlaps_in_ROI_indices` (\#306).

# 0.8.1

* Disable bugged validation of `model_type` argument in `cellpose_segmentation` (\#344).
* Raise an error if the user provides an unexpected argument to a task (\#337); this applies to the case of running a task as a script, with a pydantic model for task-argument validation.

# 0.8.0

* **(major)** Update task interface: remove filename extension from `input_paths` and `output_path` for all tasks, and add new arguments `(image_extension,image_glob_pattern)` to `create_ome_zarr` task (\#323).
* Implement logic for handling `image_glob_patterns` argument, both when globbing images and in Yokogawa metadata parsing (\#326).
* Fix minor bugs in task arguments (\#329).

# 0.7.5

- Update `cellpose_segmentation` defaults and parse additional parameters (\#316).
- Add dual-channel input for `cellpose_segmentation` task (\#315).

# 0.7.4

- Add tests for python 3.10 (\#309).
- Drop support for python 3.8 (\#319).
- Update task interface: use string arguments instead of `pathlib.Path`, and only set defaults in function call signatures (\#303).

# 0.7.3

- Add `reset_origin` argument to `convert_ROI_table_to_indices` (\#305).
- Do not overwrite existing labels in `cellpose_segmentation` task (\#308).

# 0.7.2

- Remove pyqt5-related dependencies (\#288).

# 0.7.1

Missing

# 0.7.0

- Replace `dask.array.core.get_mapper()` with `zarr.storage.FSStore()` (\#282).
- Pin dask version to >=2023.1.0, <2023.2.
- Pin zarr version to >=2.13.6, <2.14.
- Pin numpy version to >=1.23.5,<1.24.
- Pin cellpose version to >=2.2,<2.3.

# 0.6.5

- Remove FOV overlaps with more flexibility (\#265).

# 0.6.4

- Created `tools` submodule and installation extra (\#262).

# 0.6.3

- Added napari dependency, pinned to 0.4.16 version.
- Fixed type-hinting bug in task to create multiplexing OME-Zarr structure (\#258).

# 0.6.2

- Support passing a pre-made metadata table to tasks creating the OME-Zarr structure (\#252).

# 0.6.1

- Add option for padding an array with zeros in `upscale_array` (\#251).
- Simplified `imagecodecs` and `PyQt5` dependencies (\#248).

# 0.6.0

- **(major)** Refactor of how to address channels (\#239).
- Fix bug in well ROI table (\#245).

# 0.5.1

- Fix sorting of image files when number of Z planes passes 100 (\#237).

# 0.5.0

- **(major)** Deprecate `measurement` task (\#235).
- **(major)** Use more uniform names for tasks, both in python modules and manifest (\#235).
- Remove deprecated manifest from `__init__.py` (\#233).

# 0.4.6

- Skip image files if filename is not parsable (\#219).
- Preserve order of `input_paths` for multiplexing subfolders (\#222).
- Major refactor of `replicate_zarr_structure`, also enabling support for zarr files with multiple images (\#223).

# 0.4.5

- Replace `Cellpose` wrapper with `CellposeModel`, to support `pretrained_model` argument (\#218).
- Update cellpose version (it was pinned to 2.0, in previous versions) (\#218).
- Pin `torch` dependency to version 1.12.1, to support CUDA version 10.2 (\#218).

# 0.4.4

Missing due to releasing error.

# 0.4.3

- In `create_zarr_structure_multiplex`, always use/require strings for `acquisition` field (\#217).

# 0.4.2

- Bugfixes

# 0.4.1

- Only use strings as keys of `channel_parameters` (in `create_zarr_structure_multiplex`).

# 0.4.0

- **(major)** Rename `well` to `image` (both in metadata list and in manifest) and add an actual `well` field (\#210).
- Add `create_ome_zarr_multiplexing`, and adapt `yokogawa_to_zarr` (\#210).
- Relax constraint about outputs in `napari_worfklows_wrapper` (\#209).

# 0.3.4

- Always log START/END times for each task (\#204).
- Add `label_name` argument to `cellpose_segmentation` (\#207).
- Add `pretrained_model` argument to `cellpose_segmentation` (\#207).

# 0.3.3

- Added `napari_worfklows_wrapper` to manifest.

# 0.3.2

- Compute bounding boxes of labels, in `cellpose_segmentation` (\#192).
- Parse image filenames in a more robust way (\#191).
- Update manifest, moving `parallelization_level` and `executor` to `meta` attribute.

# 0.3.1

- Fix `executable` fields in manifest.
- Remove `graphviz` dependency.

# 0.3.0

- Conform to Fractal v1, through new task manifest (\#162) and standard input/output interface (\#155, \#157).
- Add several type hints (\#148) and validate them in the standard task interface (\#175).
- Update `napari_worfklows_wrapper`: pyramid level for labeling worfklows (\#148), label-only inputs (\#163, \#171), relabeling (\#167), 2D/3D handling (\#166).
- Deprecate `dummy` and `dummy_fail` tasks.

# 0.2.6

- Setup sphinx docs, to be built and hosted on <https://fractal-tasks-core.readthedocs.io>; include some preliminary updates of docstrings (\#143).
- Dependency cleanup via deptry (\#144).

# 0.2.5

- Add `napari_workflows_wrapper` task (\#141).
- Add `lib_upscale_array.py` module (\#141).

# 0.2.4

- Major updates to `metadata_parsing.py` (\#136).
