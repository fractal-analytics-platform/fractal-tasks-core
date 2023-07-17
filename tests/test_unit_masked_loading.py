import logging

import numpy as np
from devtools import debug

from ._zenodo_ome_zarrs import prepare_2D_zarr  # noqa
from ._zenodo_ome_zarrs import prepare_3D_zarr  # noqa
from fractal_tasks_core.lib_masked_loading import (
    masked_loading_wrapper,
)  # noqa


def patched_segment_ROI(img):

    logger = logging.getLogger("cellpose_segmentation.py")
    logger.info("[patched_segment_ROI] START")

    # Expects x to always be a 4D image
    assert img.ndim == 4

    # Actual labeling: segment_ROI returns a 3D mask with the same shape as
    # img, except for the first dimension
    mask = np.zeros_like(img[0, :, :, :], dtype=np.uint32)
    nz, ny, nx = mask.shape
    mask[:, 0 : ny // 4, 0 : nx // 4] = 1  # noqa
    mask[:, ny // 4 : ny // 2, 0:nx] = 2  # noqa

    logger.info("[patched_segment_ROI] END")
    return mask


def test_masked_loading_wrapper_use_masks_false():
    use_masks = False
    img_np = np.zeros((1, 1, 16, 16), dtype=np.uint16)
    debug(img_np)
    new_label_img = masked_loading_wrapper(
        image_array=img_np,
        function=patched_segment_ROI,
        use_masks=use_masks,
        kwargs={},
        preprocessing_kwargs={},
    )
    debug(new_label_img)
    # FIXME: what should we assert here?


"""
# FIXME: this is all in progress
def test_masked_loading_wrapper_use_masks_true():
    # Init
    zarr_path = tmp_path / "tmp_out/"
    metadata = prepare_2D_zarr(
      str(zarr_path),
      zenodo_zarr,
      zenodo_zarr_metadata,
      remove_labels=False,
      make_CYX=False,
    )
    debug(zarr_path)
    debug(metadata)
    # Prepare keyword arguments for preprocessing function
    preprocessing_kwargs = {}
    if use_masks:
        preprocessing_kwargs = dict(
            region=region,
            current_label_path=f"{zarrurl}/labels/{output_label_name}/0",
            ROI_table_path=ROI_table_path,
            ROI_positional_index=i_ROI,
        )
"""
