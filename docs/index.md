---
hide:
  - toc
---

# Welcome to Fractal Tasks Core's documentation!

<p align="center">
  <img src="https://raw.githubusercontent.com/fractal-analytics-platform/fractal-logos/refs/heads/main/projects/fractal_tasks_core.png" alt="Fractal tasks core logo" width="400">
</p>

Fractal tasks core is the reference task package for the [Fractal](https://fractal-analytics-platform.github.io/) framework. It provides a collection of ready-to-use tasks for processing bioimaging data, including OME-Zarr conversion, image registration, and illumination correction.

> This project is under active development 🔨. If you need help or found a bug, **open an issue [here](https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/new)**.

![Fractal overview](https://github.com/user-attachments/assets/666c8797-2594-4b8e-b1d2-b43fca66d1df)
Fractal is a framework developed at the [BioVisionCenter](https://www.biovisioncenter.uzh.ch/en.html) to process bioimaging data at scale in the OME-Zarr format and prepare the images for interactive visualization.

## Available Tasks

For a complete list of all available Fractal tasks (including tasks from other packages), visit the [Fractal task list](https://fractal-analytics-platform.github.io/fractal_tasks/).

This package includes the following tasks:

**Image Conversion**:

  - *Convert Cellvoyager to OME-Zarr*: Converts CV7000/CV8000 images to OME-Zarr format.
  - *Convert Cellvoyager Multiplexing to OME-Zarr*: Converts multiplexed images from CV7000/CV8000 to OME-Zarr.

**Image Processing**:

  - *Project Image (HCS Plate)*: Generates intensity projections (e.g., maximum intensity projection) for images in an HCS plate.
  - *Illumination Correction*: Applies flatfield correction and background subtraction using pre-calculated illumination profiles.

**Registration**:

  - *Calculate Registration (image-based)*: Computes translations for aligning images in multiplexed image analysis.
  - *Find Registration Consensus*: Generates consensus transformations for aligning multiple acquisitions, updating ROI tables as necessary.
  - *Apply Registration to Image*: Applies registration to images based on existing or newly created ROI tables.

**Other Utilities**:

  - *Import OME-Zarr*: Validates and processes existing OME-Zarr files, adding ROI tables and metadata for further processing in Fractal.

## Contributors and license

Fractal was conceived in the Liberali Lab at the Friedrich Miescher Institute for Biomedical Research and in the Pelkmans Lab at the University of Zurich by [@jluethi](https://github.com/jluethi) and [@gusqgm](https://github.com/gusqgm). The Fractal project is now developed at the [BioVisionCenter](https://www.biovisioncenter.uzh.ch/en.html) at the University of Zurich and the project lead is with [@jluethi](https://github.com/jluethi). The core development is done under contract by [eXact lab S.r.l.](https://www.exact-lab.it).

Unless otherwise specified, Fractal components are released under the BSD 3-Clause License, and copyright is with the BioVisionCenter at the University of Zurich.
