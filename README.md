# Fractal Core Tasks

[![PyPI version](https://img.shields.io/pypi/v/fractal-tasks-core?color=gree)](https://pypi.org/project/fractal-tasks-core/)
[![CI Status](https://github.com/fractal-analytics-platform/fractal-tasks-core/actions/workflows/ci_pip.yml/badge.svg)](https://github.com/fractal-analytics-platform/fractal-tasks-core/actions/workflows/ci_pip.yml)
[![Coverage](https://raw.githubusercontent.com/fractal-analytics-platform/fractal-tasks-core/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)
[![Documentation Status](https://github.com/fractal-analytics-platform/fractal-tasks-core/actions/workflows/documentation.yaml/badge.svg)](https://fractal-analytics-platform.github.io/fractal-tasks-core)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Fractal is a framework to process high-content imaging data at scale and prepare it for interactive visualization.

![Fractal_Overview](https://fractal-analytics-platform.github.io/assets/fractal_overview.jpg)

Fractal provides distributed workflows that convert TBs of image data into OME-Zarr files. The platform then processes the 3D image data by applying tasks like illumination correction, maximum intensity projection, 3D segmentation using [cellpose](https://cellpose.readthedocs.io/en/latest/) and measurements using [napari workflows](https://github.com/haesleinhuepf/napari-workflows). The pyramidal OME-Zarr files enable interactive visualization in the napari viewer.

This is the **core-tasks repository**, containing the python tasks that parse Yokogawa CV7000 images into OME-Zarr and process OME-Zarr files. Find more information about Fractal in general and the other repositories at the [Fractal home page](https://fractal-analytics-platform.github.io).

## Documentation

See https://fractal-analytics-platform.github.io/fractal-tasks-core


## Available tasks

Currently, the following tasks are available:
- Create Zarr Structure: Task to generate the zarr structure based on Yokogawa metadata files
- Yokogawa to Zarr: Parses the Yokogawa CV7000 image data and saves it to the Zarr file
- Illumination Correction: Applies an illumination correction based on a flatfield image & subtracts a background from the image.
- Image Labeling (& Image Labeling Whole Well): Applies a cellpose network to the image of a single ROI or the whole well. cellpose parameters can be tuned for optimal performance.
- Maximum Intensity Projection: Creates a maximum intensity projection of the whole plate.
- Measurement: Make some standard measurements (intensity & morphology) using napari workflows, saving results to AnnData tables.

Some additional tasks are currently being worked on and some older tasks are still present in the fractal_tasks_core folder.

# Contributors and license

Unless otherwise stated in each individual module, all Fractal components are released according to a BSD 3-Clause License, and Copyright is with Friedrich Miescher Institute for Biomedical Research and University of Zurich.

Fractal was conceived in the Liberali Lab at the Friedrich Miescher Institute for Biomedical Research and in the Pelkmans Lab at the University of Zurich by [@jluethi](https://github.com/jluethi) and [@gusqgm](https://github.com/gusqgm). The Fractal project is now developed at the [BioVisionCenter](https://www.biovisioncenter.uzh.ch/en.html) at the University of Zurich and the project lead is with [@jluethi](https://github.com/jluethi). The core development is done under contract by [eXact lab S.r.l.](https://www.exact-lab.it/).
