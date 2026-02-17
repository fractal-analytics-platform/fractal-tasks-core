import numpy as np

from fractal_tasks_core.tasks.illumination_correction import correct


def test_correct_constant_background() -> None:
    image = np.array(
        [[[10, 20], [30, 40]], [[100, 200], [300, 400]]],
        dtype=np.uint16,
    )
    flatfield = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    background = 10
    image = image + background

    corrected_image = correct(
        image=image,
        flatfield=flatfield / np.max(flatfield),
        background_constant=background,
    )

    expected_corrected_image = np.array(
        [[[40, 40], [40, 40]], [[400, 400], [400, 400]]], dtype=np.uint16
    )

    np.testing.assert_array_almost_equal(
        corrected_image, expected_corrected_image
    )


def test_correct_without_background() -> None:
    image = np.array(
        [[[10, 20], [30, 40]], [[100, 200], [300, 400]]],
        dtype=np.uint16,
    )
    flatfield = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    corrected_image = correct(
        image=image,
        flatfield=flatfield / np.max(flatfield),
    )

    expected_corrected_image = np.array(
        [[[40, 40], [40, 40]], [[400, 400], [400, 400]]], dtype=np.uint16
    )
    np.testing.assert_array_almost_equal(
        corrected_image, expected_corrected_image
    )


def test_correct_with_background_profiles() -> None:
    image = np.array(
        [[[10, 20], [30, 40]], [[100, 200], [300, 400]]],
        dtype=np.uint16,
    )
    flatfield = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    background_profile = np.array([[5, 20], [15, 15]], dtype=np.uint16)

    corrected_image = correct(
        image=image,
        flatfield=flatfield / np.max(flatfield),
        darkfield=background_profile,
    )

    expected_corrected_image = np.array(
        [[[20, 0], [20, 25]], [[380, 360], [380, 385]]],
        dtype=np.uint16,
    )

    np.testing.assert_array_almost_equal(
        corrected_image, expected_corrected_image
    )
