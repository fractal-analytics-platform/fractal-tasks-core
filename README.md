# Fractal Core Tasks

[![PyPI version](https://img.shields.io/pypi/v/fractal-tasks-core?color=gree)](https://pypi.org/project/fractal-tasks-core/)
[![Documentation Status](https://readthedocs.org/projects/fractal-tasks-core/badge/?version=latest)](https://fractal-tasks-core.readthedocs.io/en/latest)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Fractal is a framework to process high content imaging data at scale and prepare it for interactive visualization.

Fractal provides distributed workflows that convert TBs of image data into OME-Zarr files. The platform then processes the 3D image data by applying tasks like illumination correction, maximum intensity projection, 3D segmentation using [cellpose](https://cellpose.readthedocs.io/en/latest/) and measurements using [napari workflows](https://github.com/haesleinhuepf/napari-workflows). The pyramidal OME-Zarr files enable interactive visualization in the napari viewer.

![Fractal_Overview](https://user-images.githubusercontent.com/18033446/190978261-2e7b57e9-72c7-443e-9202-15d233f8416d.jpg)

This is the tasks repository, containing the python tasks that parse Yokogawa CV7000 images into OME-Zarr and process OME-Zarr files. Find more information about Fractal in general and the other repositories at the [main repository here](https://github.com/fractal-analytics-platform/fractal).

All tasks are written as python functions and are optimized for usage in Fractal workflows. But they can also be used as standalone functions to parse data or process OME-Zarr files. We heavily use regions of interest (ROIs) in our OME-Zarr files to store the positions of field of views. ROIs are saved as AnnData tables following [this spec proposal](https://github.com/ome/ngff/pull/64). We save wells as large Zarr arrays instead of a collection of arrays for each field of view ([see details here](https://github.com/ome/ngff/pull/137)).

Here is an example of the interactive visualization in napari using the newly-proposed async loading in [NAP4](https://github.com/napari/napari/pull/4905) and the [napari-ome-zarr plugin](https://github.com/ome/napari-ome-zarr):

![napari_plate_overview](https://user-images.githubusercontent.com/18033446/190983839-afb9743f-530c-4b00-bde7-23ad62404ee8.gif)


## Installation instructions

TBD

## Available tasks

Currently, the following tasks are available:
- Create Zarr Structure: Task to generate the zarr structure based on Yokogawa metadata files
- Yokogawa to Zarr: Parses the Yokogawa CV7000 image data and saves it to the Zarr file
- Illumination Correction: Applies an illumination correction based on a flatfield image & subtracts a background from the image.
- Image Labeling (& Image Labeling Whole Well): Applies a cellpose network to the image of a single ROI or the whole well. cellpose parameters can be tuned for optimal performance.
- Maximum Intensity Projection: Creates a maximum intensity projection of the whole plate.
- Measurement: Make some standard measurements (intensity & morphology) using napari workflows, saving results to AnnData tables.

Some additional tasks are currently being worked on and some older tasks are still present in the fractal_tasks_core folder.

## Contributors

Fractal was conceived in the Liberali Lab at the Friedrich Miescher Institute
for Biomedical Research and in the Pelkmans Lab at the University of Zurich
(both in Switzerland). The project lead is with
[@gusqgm](https://github.com/gusqgm) & [@jluethi](https://github.com/jluethi).
The core development is done under contract by
[@mfranzon](https://github.com/mfranzon), [@tcompa](https://github.com/tcompa)
& [@jacopo-exact](https://github.com/jacopo-exact) from [eXact lab
S.r.l.](https://exact-lab.it).

## License

Fractal is released according to a BSD 3-Clause License. See `LICENSE`.
