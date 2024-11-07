# Fractal Tasks Core

<p align="center">
  <img src="https://raw.githubusercontent.com/fractal-analytics-platform/fractal-logos/refs/heads/main/projects/fractal_tasks_core.png" alt="Fractal tasks core logo" width="400">
</p>

[![PyPI version](https://img.shields.io/pypi/v/fractal-tasks-core?color=gree)](https://pypi.org/project/fractal-tasks-core/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI Status](https://github.com/fractal-analytics-platform/fractal-tasks-core/actions/workflows/ci_pip.yml/badge.svg)](https://github.com/fractal-analytics-platform/fractal-tasks-core/actions/workflows/ci_pip.yml)
[![Coverage](https://raw.githubusercontent.com/fractal-analytics-platform/fractal-tasks-core/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)
[![Documentation Status](https://github.com/fractal-analytics-platform/fractal-tasks-core/actions/workflows/documentation.yaml/badge.svg)](https://fractal-analytics-platform.github.io/fractal-tasks-core)

Fractal tasks core is the official task package for the [Fractal](https://fractal-analytics-platform.github.io/) framework. It provides essential tools for building Fractal tasks, helpful utility functions, and a collection of ready-to-use tasks for processing bioimaging data. These tasks include OME-Zarr conversion, image registration, segmentation, and measurements.

![Fractal_overview_small](https://github.com/user-attachments/assets/666c8797-2594-4b8e-b1d2-b43fca66d1df)

[Fractal](https://fractal-analytics-platform.github.io/) is a framework developed at the [BioVisionCenter](https://www.biovisioncenter.uzh.ch/en.html) to process bioimaging data at scale in the OME-Zarr format and prepare the images for interactive visualization.

## Core Library Components
This repository includes several core sub-packages:
- **NGFF Sub-package**: Validates OME-Zarr metadata and provides utilities for reading and writing it.
- **Tables Sub-package**: Handles AnnData tables for ROIs and features, including reading and writing operations. (See the [Fractal table specification](https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/)).
- **ROI Sub-package**: Manages ROI-related table processing.
- **Dev Sub-package**: Handles Fractal manifest creation and task validation for the Fractal server.

The NGFF, Tables & ROI Sub-package functionality will get replaced by functionality in [ngio](https://github.com/fractal-analytics-platform/ngio) over the coming months.


## Documentation

See https://fractal-analytics-platform.github.io/fractal-tasks-core

## Available Tasks

This package includes the following tasks:

- **Image Conversion**:
  - *Convert Cellvoyager to OME-Zarr*: Converts CV7000/CV8000 images to OME-Zarr format.
  - *Convert Cellvoyager Multiplexing to OME-Zarr*: Converts multiplexed images from CV7000/CV8000 to OME-Zarr.

- **Image Processing**:
  - *Project Image (HCS Plate)*: Generates intensity projections (e.g., maximum intensity projection) for images in an HCS plate.
  - *Illumination Correction*: Applies flatfield correction and background subtraction using pre-calculated illumination profiles.

- **Segmentation**:
  - *Cellpose Segmentation*: Segments images using custom or pre-trained Cellpose models, with user-tunable options.

- **Registration**:
  - *Calculate Registration*: Computes translations for aligning images in multiplexed image analysis.
  - *Find Registration Consensus*: Generates consensus transformations for aligning multiple acquisitions, updating ROI tables as necessary.
  - *Apply Registration to Image*: Applies registration to images based on existing or newly created ROI tables.

- **Measurements**:
  - *Napari Workflows Wrapper*: Task to run existing napari workflows through Fractal to process images and labels and to generate new labels or measurement tables. Takes an arbitrary napari workflow yaml file to run.

- **Other Utilities**:
  - *Import OME-Zarr*: Validates and processes existing OME-Zarr files, adding ROI tables and metadata for further processing in Fractal.


## Installation

To install and use the library components, run:

```
pip install fractal-tasks-core
```

This will install the core library, including all sub-packages for working with NGFF metadata, tables, ROIs, and more.

If you intend to run Fractal tasks (such as segmentation or registration), install with the additional task dependencies:
```
pip install "fractal-tasks-core[fractal-tasks]"
```

This includes larger dependencies such as Torch (for Cellpose) and Napari (for Napari workflows).

If you collect this package on Fractal server to run the task, make sure to add the fractal-tasks extra in the corresponding field for extras.

## Contributors and license

Fractal was conceived in the Liberali Lab at the Friedrich Miescher Institute for Biomedical Research and in the Pelkmans Lab at the University of Zurich by [@jluethi](https://github.com/jluethi) and [@gusqgm](https://github.com/gusqgm). The Fractal project is now developed at the [BioVisionCenter](https://www.biovisioncenter.uzh.ch/en.html) at the University of Zurich and the project lead is with [@jluethi](https://github.com/jluethi). The core development is done under contract by [eXact lab S.r.l.](https://www.exact-lab.it).

Unless otherwise specified, Fractal components are released under the BSD 3-Clause License, and copyright is with the BioVisionCenter at the University of Zurich.
