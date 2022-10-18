import numpy as np


def upscale_array(reference_array=None, array=None):

    # Find upscale_factor for labels array
    ref_shape = reference_array.shape
    array_shape = array.shape

    upscale_factor_x = ref_shape[-1] // array_shape[-1]
    upscale_factor_y = ref_shape[-2] // array_shape[-2]
    msg = (
        f"Trying to upscale from {array_shape=} to {ref_shape=}, with"
        f"{upscale_factor_x=} and {upscale_factor_y=}."
    )

    if upscale_factor_x != upscale_factor_y:
        error_msg = f"{msg} Expecting upscale_factor_x=upscale_factor_y."
        raise ValueError(error_msg)
    upscale_factor = upscale_factor_x

    if (
        array_shape[-1] * upscale_factor != ref_shape[-1]
        or array_shape[-2] * upscale_factor != ref_shape[-2]
    ):
        raise ValueError(msg)

    # Upscale labels array - see https://stackoverflow.com/a/7525345/19085332
    x_rescaled_array = np.repeat(array, upscale_factor, axis=-1)
    xy_rescaled_array = np.repeat(x_rescaled_array, upscale_factor, axis=-2)
    if not xy_rescaled_array.shape == ref_shape:
        error_msg = f"{msg} Upscaled-array shape: {xy_rescaled_array.shape}."
        raise Exception(error_msg)

    return xy_rescaled_array
