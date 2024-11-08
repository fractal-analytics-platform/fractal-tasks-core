**Note**: Numbers like (\#123) point to closed Pull Requests on the fractal-tasks-core repository.

* Tasks:
  * Refactor projection task to use ngio
* Dependencies:
  * Add ngio
* CI:
    * Remove Python 3.9 from the CI matrix

# 1.3.2
* Tasks:
    * Add percentile-based rescaling to calculate registration task to make it more robust (\#848)
* Dependencies:
  * Relax pandas constraint to `<2`.
  * Relax torch constraint to `<=3.0.0`.
  * Relax numpy constraint to `<2.1.0`.
  * Add python 3.12 to the CI matrix (\#770).
* Chores:
  * fix future warning when using Series.__getitem__ with positional arguments.

# 1.3.1

* Testing
    * Use latest version of Zenodo tiny-ome-zarr dataset (\#817).
    * Relax pip-version constraint in external-packages-manifest tests (\#825).
    * Run pip-based CI also regularly and on-demand (\#829).
    * Update GitHub actions for upload/download/coverage (\#832).
* Dependencies:
    * Require `pydantic<=2.8.2` (\#836).
    * Require `stackview<=0.9.0` (\#829).
* Documentation:
    * Bump `mkdocstrings` and `mkdocstrings-python` to support `griffe` v1 (\#818).

# 1.3.0

* Tasks:
    * `image_glob_patterns` are renamed to `include_glob_patterns` in Convert Cellvoyager to OME-Zarr (regular & multiplexing) (\#812).
    * Convert Cellvoyager to OME-Zarr (regular & multiplexing) gain exclusion patterns to exclude specific patterns of images from being processed (\#812).
    * Fix issue with arbitrary acquisition names in Convert Cellvoyager Multiplexing to OME-Zarr (\#812).
    * In Convert Cellvoyager to OME-Zarr (regular & multiplexing), handle channels in the mrf metadata file that aren't present in the mlf metadata better (\#812).
    * In Convert Cellvoyager to OME-Zarr, improve plate metadata for image list when multiple plates with the same plate name are processed (\#812).
    * Catch errors for missing mlf & mrf files better in Convert Cellvoyager to OME-Zarr (regular & multiplexing) (\#812).
    * Drop defusexml dependency for cellvoyager metadata conversion (\#812).
    * Rename `Maximum Intensity Projection HCS Plate` task to `Project Image (HCS Plate)` (\#814).
    * Expose selection of projection modes in `Project Image (HCS Plate)`: MIP, MINIP, MEANIP & SUMIP (\#814).
    * Rename task function from `maximum_intensity_projection` to `projection` and modified parameters in `fractal_tasks_core.tasks.io_models.InitArgsMIP` (\#814).

# 1.2.1
* Core-library
    * Add `create_roi_table_from_df_list` library function in `fractal_tasks_core.v1.roi`: It combines a list of ROI table dataframes into an AnnData ROI table and handles repeating labels (\#811).
    * Move `create_well_acquisition_dict` & `_split_well_path_image_path` from `fractal_tasks_core.tasks._registration_utils.py` & `fractal_tasks_core.tasks._zarr_utils` into `fractal_tasks_core.utils` (\#811).
* Tasks:
    * Fixes issue 810 for Cellpose task: Avoids creating duplicated entries in ROI tables when masking ROI table input was used (\#811).

# 1.2.0

* Core-library and tasks:
    * Switch all core models to Pydantic V2 (\#793).
* JSON Schema generation tools:
    * Move JSON-Schema tools to Pydantic V2 (\#793).
* Testing:
    * Remove dependency on `pytest-pretty` (\#793).
    * Update `manifest_external_packages.yml` GitHub Action so that it installs the current `fractal-tasks-core` (\#793).

# 1.1.1

* Tasks:
    * Fix issue with masked ROI & relabeling in Cellpose task (\#786).
    * Fix issue with masking ROI label types in `masked_loading_wrapper` for Cellpose task (\#786).
    * Enable workaround to support yx images in Cellpose task (\#789).
    * Fix error handling in `calculate_registration_image_based` (\#799).
    * Fix minor issues with call-signature and type hints in `calculate_registration_image_based` (\#799).

# 1.1.0

> NOTE: Starting from this release, `fractal-tasks-core` can coexist
> with Pydantic V2 but it still uses pydantic
> v1 under the hood for the time being. When working with Pydantic V1, the
> required version is `>=1.10.16`.

* Tasks:
    * Refactor Cellpose Task inputs: Combine Channel inputs & channel normalization parameters (\#738).
    * Refactor Cellpose Task inputs: Group advanced Cellpose parameters into the `CellposeModelParams` model that's provided via `advanced_cellpose_model_params` (\#738).
    * Refactor Cellpose Task inputs: Support independent normalization of 2 input channels in the Cellpose task (\#738).
    * Rename `task.cellpose_transforms` into `tasks.cellpose_utils` (\#738).
    * Fix wrong repeated overlap checks for bounding-boxes in Cellpose task (\#778).
    * Fix minor MIP issues related to plate metadata and expecting acquisition metadata in all NGFF plates (\#781).
    * Add `chi2_shift` option to Calculate Registration (image-based) task (\#741).
* Development:
    * Switch to transitional pydantic.v1 imports, changes pydantic requirement to `==1.10.16` or `>=2.6.3` (\#760).
    * Support JSON-Schema generation for `Enum` task arguments (\#749).
    * Make JSON-Schema generation tools more flexible, to simplify testing (\#749).
* Documentation:
    * Update documentation (\#751).
    * Improve/extend page showing tasks from other packages (\#759, \#777).
* JSON Schema generation:
    * Test manifest creation for three other tasks packages (\#763).
* NGFF subpackage
    * Fix Plate model to correspond better to 0.4.0 NGFF spec: Now makes acquisition metadata optional (\#781).
* Dependencies:
    * Add `image_registration` within `fractal-tasks` extra (\#741).

# 1.0.2

* Fix bug in plate metadata in MIP task (in the `copy_ome_zarr_hcs_plate` init function) (\#736).

# 1.0.1

* Add support for converting 1536 well plates in cellvoyager converters (\#715).
* Testing:
    * Make validation of NGFF Zarr attributes more strict, in tests (\#732).
* Development:
    * Update poetry to 1.8.2 (\#734).

# 1.0.0

* Update all tasks to use the new Fractal API from Fractal server 2.0 (\#671)
* Provide new dev tooling to create Fractal manifest for new task API (\#671)
* Add Pydantic models for OME-NGFF HCS Plate validation (\#671)
* Breaking changes in core library:
    * In `get_acquisition_paths` helper function of `NgffWellMeta`:
        The dictionary now contains a list of paths as values, not single paths.
        The NotImplementedError for multiple images with the same acquisition was removed.
    * The `utils.get_table_path_dict` helper function was made private & changed its input parameters:
        It's now `_get_table_path_dict(zarr_url: str)`
* Breaking changes in task sub-package:
    * Rename tasks for increase clarity (\#671 & \#706).
    * Changed registration tasks flow: Now 2 Compound tasks, 1 Parallel task (\#671).
    * Changed parameter names in registration tasks: acquisition instead of cycle (\#719).
    * Changed parameter names & defaults in illumination correction task: (\#671).
        * Now uses `illumination_profiles` instead of `dict_corr`.
        * Changes default of background subtraction from 110 to 0.
* Dependencies:
    * Add `filelock` (\#718).

# 0.14.3

* Make Cellpose task work for non HCS plate OME-Zarr images (\#659)
* Add option to Illumination Correction to specify the ROI table name (\#665)

# 0.14.2

* Add custom normalization options to the Cellpose task (\#650)
* Add more options to the Cellpose task to control model behavior (\#650)
* For Cellpose task, switch to using Enums for `model_type` (see issue \#401)

# 0.14.1

* Fix bug in `cellpose_segmentation` upon using masked loading and setting `channel2` (\#639). Thanks [@FranziskaMoos-FMI](https://github.com/FranziskaMoos-FMI) and [@enricotagliavini](https://github.com/enricotagliavini).
* Improve handling of potential race condition in "Apply Registration to image" task (\#638).

# 0.14.0

* Breaking changes in tasks:
    * Make `NapariWorkflowsOutput.label_name` attribute required, and use it to fill the `region["path"]` table attribute (\#613).
* Breaking changes in core library:
    * ⚠️ Refactor the whole package structure, leading to breaking changes for most imports (\#613); more details at [this page](https://fractal-analytics-platform.github.io/fractal-tasks-core/version_updates/v0_14_0/).
    * In `prepare_label_group` helper function:
        * Make `label_attrs` function argument required (\#613).
        * Validate `label_attrs` with `NgffImageMeta` model (\#613).
        * Override multiscale name in `label_attrs` with `label_name` (\#613).
    * In `write_table` helper function:
        * Drop `logger` function argument (\#613).
        * Add `table_name` function argument, taking priority over `table_attrs` (\#613).
        * Raise an error if no table type is not provided (\#613).
        * Raise an error if table attributes do not comply with table specs (\#613).
* Other internal changes:
    * Comply with table specs V1, by writing all required Zarr attributes (\#613).
    * Remove `has_args_schema` obsolete property from manifest (\#603).
    * Handle `GroupNotFoundError` in `load_NgffImageMeta` and `load_NgffWellMeta` (\#622).
* Bug fixes:
    * Fix table selection in calculate registration image-based (\#615).
* Documentation
    * Clarify table specs V1 (\#613).
* Testing:
    * Use more recent Zenodo datasets, created with `fractal-tasks-core>=0.12` (\#623).
    * Use poetry 1.7.1 in GitHub actions (\#620).
    * Align with new Zenodo API (\#601).
    * Update `test_valid_manifest` (\#606).
    * Use [pooch](https://www.fatiando.org/pooch) to download test files (\#610).
* Documentation:
    * Add list of tasks (\#625).
* Dependencies:
    * Remove Pillow `<10.1.0` constraint (\#626).

# 0.13.1

* Always use `write_table` in tasks, rather than AnnData `write_elem` (\#581).
* Remove assumptions on ROI-table columns from `get_ROI_table_with_translation` helper function of `calculate_registration_image_based` task (\#591).
* Testing:
    * Cache Zenodo data, within GitHub actions (\#585).
* Documentation:
    * Define V1 of table specs (\#582).
    * Add mathjax support (\#582).
    * Add cross-reference inventories to external APIs (\#582).

# 0.13.0

* Tasks:
    * New task and helper functions:
        * Introduce `import_ome_zarr` task (\#557, \#579).
        * Introduce `get_single_image_ROI` and `get_image_grid_ROIs` (\#557).
        * Introduce `detect_ome_ngff_type` (\#557).
        * Introduce `update_omero_channels` (\#579).
    * Make `maximum_intensity_projection` independent from ROI tables (\#557).
    * Make Cellpose task work when `input_ROI_table` is empty (\#566).
    * Fix bug of missing attributes in ROI-table Zarr group (\#573).
* Dependencies:
    * Restrict `Pillow` version to `<10.1` (\#571).
    * Support AnnData `0.10` (\#574).
* Testing:
    * Align with new Zenodo API (\#568).
    * Use ubuntu-22 for GitHub CI (\#576).

# 0.12.2

* Relax `check_valid_ROI_indices` to support search-first scenario (\#555).
* Do not install `docs` dependencies in GitHub CI (\#551).

# 0.12.1

* Make `Channel.window` attribute optional in `lib_ngff.py` (\#548).
* Automate procedure for publishing package to PyPI (\#545).

# 0.12.0

This release includes work on Pydantic models for NGFF specs and on ROI tables.

* NGFF Pydantic models:
    * Introduce Pydantic models for NGFF metadata in `lib_ngff.py` (\#528).
    * Extract `num_levels` and `coarsening_xy` parameters from NGFF objects, rather than from `metadata` task input (\#528).
    * Transform several `lib_zattrs_utils.py` functions (`get_axes_names`, `extract_zyx_pixel_sizes` and `get_acquisition_paths`) into `lib_ngff.py` methods (\#528).
    * Load Zarr attributes from groups, rather than from `.zattrs` files (\#528).
* Regions of interest:
    * Set `FOV_ROI_table` and `well_ROI_table` ZYX origin to zero (\#524).
    * Remove heuristics to determine whether to reset origin, in `cellpose_segmentation` task (\#524).
    * Remove obsolete `reset_origin` argument from `convert_ROI_table_to_indices` function (\#524).
    * Remove redundant `reset_origin` call from `apply_registration_to_ROI_tables` task (\#524).
    * Add check on non-negative ROI indices (\#534).
    * Add check on ROI indices not starting at `(0,0,0)`, to highlight v0.12/v0.11 incompatibility (\#534).
    * Fix bug in creation of bounding-box ROIs when `cellpose_segmentation` loops of FOVs (\#524).
    * Update type of `metadata` parameter of `prepare_FOV_ROI_table` and `prepare_well_ROI_table` functions (\#524).
    * Fix `reset_origin` so that it returns an updated copy of its input (\#524).
* Dependencies:
    * Relax `fsspec<=2023.6` constraint into `fsspec!=2023.9.0` (\#539).

# 0.11.0

* Tasks:
    * **(major)** Introduce new tasks for registration of multiplexing cycles: `calculate_registration_image_based`, `apply_registration_to_ROI_tables`, `apply_registration_to_image` (\#487).
    * **(major)** Introduce new `overwrite` argument for tasks `create_ome_zarr`, `create_ome_zarr_multiplex`, `yokogawa_to_ome_zarr`, `copy_ome_zarr`, `maximum_intensity_projection`, `cellpose_segmentation`, `napari_workflows_wrapper` (\#499).
    * **(major)** Rename `illumination_correction` parameter from `overwrite` to `overwrite_input` (\#499).
    * Fix plate-selection bug in `copy_ome_zarr` task (\#513).
    * Fix bug in definition of `metadata["plate"]` in `create_ome_zarr_multiplex` task (\#513).
    * Introduce new helper functions `write_table`, `prepare_label_group` and `open_zarr_group_with_overwrite` (\#499).
    * Introduce new helper functions `are_ROI_table_columns_valid`, `convert_indices_to_regions`, `reset_origin`, `is_standard_roi_table`, `get_acquisition_paths`, `get_table_path_dict`, `get_axes_names`, `add_zero_translation_columns`, `calculate_min_max_across_dfs`, `apply_registration_to_single_ROI_table`, `write_registered_zarr`, `calculate_physical_shifts`, `get_ROI_table_with_translation` (\#487).
* Testing:
    * Add tests for `overwrite`-related task behaviors (\#499).
    * Introduce mock-up of `napari_skimage_regionprops` package, for testing of
      `napari_workflows_wrapper` task (\#499).
* Dependencies:
    * Require `fsspec` version to be `<=2023.6` (\#509).

# 0.10.1

* Tasks:
    * Improve validation for `OmeroChannel.color` field (\#488).
    * Include `image-label/source/image` OME-NGFF attribute when creating labels (\#478).
    * Update default values for tolerance (`tol`) in `lib_ROI_overlaps.py` functions (\#466).
* Development tools:
    * Include `docs_info` and `docs_link` attributes in manifest tasks (\#486).
    * Rename and revamp scripts to update/check the manifest (\#486).
    * Improve logging and error-handling in tools for args-schema creation (\#469).
* Documentation:
    * Convert docstrings to Google style (\#473, \#479).
    * Switch from sphinx to mkdocs for documentation (\#479).
    * Update generic type hints (\#462, \#479).
    * Align examples to recent package version, and mention them in the documentation (\#470).
* Testing:
    * Improve coverage of core library (\#459, \#467, \#468).
    * Update Zenodo datasets used in tests (\#454).
    * Run tests both for the poetry-installed and pip-installed package (\#455).
* Dependencies:
    * Relax `numpy` required version to `<2` (\#477).
    * Relax `dask` required version to `>=2023.1.0` (\#455).
    * Relax `zarr` required version to `>=2.13.6,<3` (\#455).
    * Relax `pandas` required version to `>=1.2.0,<2` (\#455).
    * Relax `Pillow` required version to `>=9.1.1,<10.0.0` (\#455).
    * Full update of `poetry.lock` file (mutiple PRs, e.g. \#472).
    * Include `requests` and `wget` in the `dev` poetry dependency group (\#455).

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
