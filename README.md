# Fractal Core Tasks

[![PyPI version](https://img.shields.io/pypi/v/fractal-tasks-core?color=gree)](https://pypi.org/project/fractal-tasks-core/)
[![CI Status](https://github.com/fractal-analytics-platform/fractal-tasks-core/actions/workflows/ci_pip.yml/badge.svg)](https://github.com/fractal-analytics-platform/fractal-tasks-core/actions/workflows/ci_pip.yml)
[![Coverage](https://raw.githubusercontent.com/fractal-analytics-platform/fractal-tasks-core/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)
[![Documentation Status](https://github.com/fractal-analytics-platform/fractal-tasks-core/actions/workflows/documentation.yaml/badge.svg)](https://fractal-analytics-platform.github.io/fractal-tasks-core)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Fractal is a framework to process bioimaging data at scale in the OME-Zarr format and prepare the images for interactive visualization.

![Fractal_Overview](https://fractal-analytics-platform.github.io/assets/fractal_overview.jpg)

Fractal provides distributed workflows that convert TBs of image data into OME-Zarr files. The platform then processes the 3D image data by applying tasks like illumination correction, maximum intensity projection, 3D segmentation using [cellpose](https://cellpose.readthedocs.io/en/latest/) and measurements using [napari workflows](https://github.com/haesleinhuepf/napari-workflows). The resulting pyramidal OME-Zarr files enable interactive visualization in different modern viewers like MoBIE and napari.

This is the **core-tasks repository**, containing the python tasks that converts Cellvoyager CV7000 & CV8000 images into OME-Zarr and process OME-Zarr files. Find more information about Fractal in general and the other repositories at the [Fractal home page](https://fractal-analytics-platform.github.io).

Besides tasks, this repository contains library functions for processing OME-Zarr images. The core library parts are:
- an NGFF sub-package to validate OME-Zarr metadata and provide convenience functions to read & write it
- a tables sub-package to handle reading and writing of AnnData tables for ROI tables & feature tables (see [table specification](https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/))
- a roi sub-package to handle ROI table related processing
- a dev subpackage that handles Fractal manifest creation & other validation of tasks for Fractal server


## Documentation

See https://fractal-analytics-platform.github.io/fractal-tasks-core


## Available tasks

Currently, the following tasks are available:
- Convert Cellvoyager to OME-Zarr: Task to convert Cellvoyager CV7000 & CV8000 images into OME-Zarr.
- Convert Cellvoyager Multiplexing to OME-Zarr: Task to convert multiplexed images from Cellvoyager CV7000 & CV8000 images into OME-Zarr.
- Maximum Intensity Projection HCS Plate: This task creates a new OME-Zarr HCS plate and puts in maximum intensity projections along the Z axis for all images.
- Illumination Correction: Task to apply flatfield correction & background subtraction based on pre-calculated illumination profiles.
- Cellpose Segmentation: This task performs image segmentation using custom or pre-trained Cellpose models and exposes many model options to be tuned by the user.
- Napari Workflows Wrapper: Task to run existing napari workflows through Fractal to process images and labels and to generate new labels or measurement tables. Takes an arbitrary napari workflow yaml file to run.
- Calculate Registration (image-based): Registration task for multiplexed image analysis. This task calculated the translation needed to align images between multiple acquisitions.
- Find Registration Consensus: Registration task for multiplexed image analysis. This task calculates the consensus transformation across all acquisitions to get aligned images. It creates new ROI tables for them.
- Apply Registration to Image: Registration task for multiplexed image analysis. This task applies registration based on the ROI tables generated in `Find Registration Consensus` to existing images or it creates new, registered images.
- Import OME-Zarr: Helper task to validate existing OME-Zarr files, handle the addition of ROI tables to existing Zarrs & to create the necessary metadata for Fractal server about an existing OME-Zarr image to allow further processing.


## Installation
See [details on installation in the documentation](https://fractal-analytics-platform.github.io/fractal-tasks-core/install/). This package can be installed in 2 main ways:

To use the library, just install the core package:

```
pip install fractal-tasks-core
```

This installs the core library and allows you to use all the sub-packages for ngff metadata, tables, rois etc.

If you want to run the actual Fractal tasks, you need to install the task extra:
```
pip install "fractal-tasks-core[fractal-tasks]"
```

This installs the heavier dependencies like torch for Cellpose & napari for napari workflows.

If you collect this package on Fractal server to run the task, make sure to add the fractal-tasks extra in the corresponding field for extras.

# Contributors and license

Unless otherwise stated in each individual module, all Fractal components are released according to a BSD 3-Clause License, and Copyright is with the BioVisionCenter at the University of Zurich.

Fractal was conceived in the Liberali Lab at the Friedrich Miescher Institute for Biomedical Research and in the Pelkmans Lab at the University of Zurich by [@jluethi](https://github.com/jluethi) and [@gusqgm](https://github.com/gusqgm). The Fractal project is now developed at the BioVisionCenter at the University of Zurich and the project lead is with [@jluethi](https://github.com/jluethi). The core development is done under contract by [eXact lab S.r.l.](https://www.exact-lab.it).
