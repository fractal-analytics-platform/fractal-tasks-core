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

**Image Processing**:

  - *Project Image (HCS Plate)*: Generates intensity projections (e.g., maximum intensity projection) for images in an HCS plate.
  - *Project Image*: Generates intensity projections for generic (non-HCS) OME-Zarr images.
  - *Illumination Correction*: Applies flatfield correction and background subtraction using pre-calculated illumination profiles.

**Segmentation**:

  - *Threshold Segmentation*: Segments objects using Otsu or simple threshold methods, with optional ROI masking table creation.

**Measurement**:

  - *Measure Features*: Extracts morphological and intensity features from label images and stores results as Fractal tables.

**Registration**:

  - *Calculate Registration (image-based)*: Computes translations for aligning images in multiplexed image analysis.
  - *Find Registration Consensus*: Generates consensus transformations for aligning multiple acquisitions, updating ROI tables as necessary.
  - *Apply Registration to Image*: Applies registration to images based on existing or newly created ROI tables.

**Other Utilities**:

  - *Import OME-Zarr*: Validates and processes existing OME-Zarr files, adding ROI tables and metadata for further processing in Fractal.

## Contributors and license

The Fractal project is developed by the [BioVisionCenter](https://www.biovisioncenter.uzh.ch/en.html) at the University of Zurich, who contracts [eXact lab s.r.l.](https://www.exact-lab.it/en/) for software engineering and development support.

Unless otherwise specified, Fractal components are released under the BSD 3-Clause License, and copyright is with the BioVisionCenter at the University of Zurich.
