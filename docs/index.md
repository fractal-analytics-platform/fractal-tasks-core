---
hide:
  - toc
---

# Welcome to Fractal Tasks Core's documentation!

Fractal is a framework to process high content imaging data at scale and prepare it for interactive visualization.

> This project is under active development 🔨. If you need help or found a bug, please _open an issue [here](https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/new)_.

Fractal provides distributed workflows that convert TBs of image data into [OME-Zar](https://ngff.openmicroscopy.org) files.
The platform then processes the 3D image data by applying tasks like illumination correction, maximum intensity projection, 3D segmentation using [cellpose](https://cellpose.readthedocs.io) and measurements using [napari workflows](https://github.com/haesleinhuepf/napari-workflows).
The pyramidal OME-Zarr files enable interactive visualization in the napari viewer.

![Fractal overview](https://user-images.githubusercontent.com/18033446/190978261-2e7b57e9-72c7-443e-9202-15d233f8416d.jpg)


The **fractal-tasks-core** package contains the python tasks that parse Yokogawa CV7000 images into OME-Zarr and process OME-Zarr files. Find more information about Fractal in general and the other repositories at [this link](https://fractal-analytics-platform.github.io/). All tasks are written as Python functions and are optimized for usage in Fractal workflows, but they can also be used as standalone functions to parse data or process OME-Zarr files. We heavily use regions of interest (ROIs) in our OME-Zarr files to store the positions of field of views. ROIs are saved as AnnData tables following [this spec proposal](https://github.com/ome/ngff/pull/64). We save wells as large Zarr arrays instead of a collection of arrays for each field of view (see details [here](https://github.com/ome/ngff/pull/137)).

Here is an example of the interactive visualization in napari using the newly-proposed async loading in [NAP4](https://github.com/napari/napari/pull/4905) and the [napari-ome-zarr plugin](https://github.com/ome/napari-ome-zarr):

![Napari plate overview](https://user-images.githubusercontent.com/18033446/190983839-afb9743f-530c-4b00-bde7-23ad62404ee8.gif)


## Available tasks

Currently, the following tasks are available:

* <u>Create Zarr Structure</u>: Task to generate the zarr structure based on Yokogawa metadata files
* <u>Yokogawa to Zarr</u>: Parses the Yokogawa CV7000 image data and saves it to the Zarr file
* <u>Illumination Correction</u>: Applies an illumination correction based on a flatfield image & subtracts a background from the image.
* <u>Image Labeling</u> (& <u>Image Labeling Whole Well</u>): Applies a cellpose network to the image of a single ROI or the whole well. cellpose parameters can be tuned for optimal performance.
* <u>Maximum Intensity Projection</u>: Creates a maximum intensity projection of the whole plate.
* <u>Measurement</u>: Make some standard measurements (intensity & morphology) using napari workflows, saving results to AnnData tables.

Some additional tasks are currently being worked on and some older tasks are still present in the `fractal_tasks_core` folder. See [the package page](https://fractal-analytics-platform.github.io/fractal-tasks-core/api_files/fractal_tasks_core.html) for the detailed description of all tasks.

## Contributors

Fractal was conceived in the Liberali Lab at the Friedrich Miescher Institute for Biomedical Research and in the Pelkmans Lab at the University of Zurich (both in Switzerland). The project lead is with [@gusqgm](https://github.com/gusqgm) & [@jluethi](https://github.com/jluethi). The core development is done under contract by [eXact lab S.r.l.](https://exact-lab.it).

## License

Fractal is released according to a BSD 3-Clause License, see [LICENSE](https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/
main/LICENSE).
