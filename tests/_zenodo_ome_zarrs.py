"""
Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
University of Zurich

Original authors:
Marco Franzon <marco.franzon@exact-lab.it>
Tommaso Comparin <tommaso.comparin@exact-lab.it>

This file is part of Fractal and was originally developed by eXact lab S.r.l.
<exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
Institute for Biomedical Research and Pelkmans Lab from the University of
Zurich.
"""
import json
import shutil
from pathlib import Path
from typing import Any

import dask.array as da
from devtools import debug


def prepare_3D_zarr(
    zarr_path: str,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
    remove_tables: bool = False,
):
    zenodo_zarr_3D, zenodo_zarr_2D = zenodo_zarr[:]
    metadata_3D, metadata_2D = zenodo_zarr_metadata[:]
    shutil.copytree(
        zenodo_zarr_3D, str(Path(zarr_path) / Path(zenodo_zarr_3D).name)
    )
    if remove_tables:
        shutil.rmtree(
            str(Path(zarr_path) / Path(zenodo_zarr_3D).name / "B/03/0/tables")
        )
    metadata = metadata_3D.copy()
    return metadata


def prepare_2D_zarr(
    zarr_path: str,
    zenodo_zarr: list[str],
    zenodo_zarr_metadata: list[dict[str, Any]],
    remove_labels: bool = False,
    make_CYX: bool = False,
):
    zenodo_zarr_3D, zenodo_zarr_2D = zenodo_zarr[:]
    metadata_3D, metadata_2D = zenodo_zarr_metadata[:]
    shutil.copytree(
        zenodo_zarr_2D, str(Path(zarr_path) / Path(zenodo_zarr_2D).name)
    )
    FOV_path = Path(zarr_path) / Path(zenodo_zarr_2D).name / "B/03/0"

    # Preliminary check
    if make_CYX and not remove_labels:
        raise ValueError(f"Cannot set {make_CYX=} and {remove_labels=}.")

    # Remove existing labels
    if remove_labels:
        label_dir = str(FOV_path / "labels")
        debug(label_dir)
        shutil.rmtree(label_dir)

    # Transform zarr array from CZYX to CYX
    if make_CYX:
        # Transform zarr array into CYX - part 1 (metadata)
        with (FOV_path / ".zattrs").open("r") as f:
            zattrs = json.load(f)
        for ind, ds in enumerate(zattrs["multiscales"][0]["datasets"]):
            new_ds = ds.copy()
            transf = new_ds["coordinateTransformations"][0]
            new_transf = transf.copy()
            new_transf["scale"] = [new_transf["scale"][x] for x in [0, 2, 3]]
            new_ds["coordinateTransformations"][0] = new_transf
            zattrs["multiscales"][0]["datasets"][ind] = new_ds
        zattrs["multiscales"][0]["axes"] = [
            ax for ax in zattrs["multiscales"][0]["axes"] if ax["name"] != "z"
        ]
        with (FOV_path / ".zattrs").open("w") as f:
            json.dump(zattrs, f, indent=2)
        # Transform zarr array into CYX - part 2 (zarr arrays)
        for ind, ds in enumerate(zattrs["multiscales"][0]["datasets"]):
            zarr_path = str(FOV_path / ds["path"])
            debug(zarr_path)
            data_czyx = da.from_zarr(zarr_path).compute()
            data_cyx = data_czyx[:, 0, :, :]
            assert data_czyx.shape[1] == 1
            shutil.rmtree(zarr_path)
            da.array(data_cyx).to_zarr(zarr_path, dimension_separator="/")

    metadata = metadata_2D.copy()
    return metadata
