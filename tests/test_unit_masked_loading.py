import logging

import numpy as np
from devtools import debug

from fractal_tasks_core.masked_loading import (
    masked_loading_wrapper,
)  # noqa


def patched_segment_ROI(img):

    logging.info("[patched_segment_ROI] START")

    # Expects x to always be a 4D image
    assert img.ndim == 4

    # Actual labeling: segment_ROI returns a 3D mask with the same shape as
    # img, except for the first dimension
    mask = np.zeros_like(img[0, :, :, :], dtype=np.uint32)
    nz, ny, nx = mask.shape
    mask[:, 0 : ny // 4, 0 : nx // 4] = 1  # noqa
    mask[:, ny // 4 : ny // 2, 0:nx] = 2  # noqa

    logging.info("[patched_segment_ROI] END")
    return mask


def test_masked_loading_wrapper_use_masks_false():
    """
    FIXME
    """
    img_np = np.zeros((1, 1, 16, 16), dtype=np.uint16)
    debug(img_np)
    new_label_img = masked_loading_wrapper(
        image_array=img_np,
        function=patched_segment_ROI,
        use_masks=False,
        kwargs={},
        preprocessing_kwargs={},
    )
    debug(new_label_img)
    # FIXME: what should we assert here?
