import glob
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from devtools import debug
from PIL import Image


def _shift_image(
    img_path: str,
    shift_x_pxl: int = 200,
    shift_y_pxl: int = 50,
) -> None:
    """
    Load an image, apply a XY shift, replace the remaining stripes with noise.
    """

    # Open old image as array
    old_img = Image.open(img_path)
    new_array = np.asarray(old_img).copy()

    # Shift image by (shift_y_pxl, shift_x_pxl)
    new_array[:, :-shift_x_pxl] = new_array[:, shift_x_pxl:]
    new_array[:-shift_y_pxl] = new_array[shift_y_pxl:, :]

    # Replace offset stripes with random values
    new_array[:, -shift_x_pxl:] = np.random.randint(
        0, 1000, size=new_array[:, -shift_x_pxl:].shape
    )
    new_array[-shift_y_pxl:, :] = np.random.randint(
        0, 1000, size=new_array[-shift_y_pxl:, :].shape
    )

    # Save new image
    new_img = Image.fromarray(new_array, mode="I")
    new_img.save(img_path, mode="png")


@pytest.fixture(scope="session")
def zenodo_images_multiplex_shifted(
    zenodo_images_multiplex: list[str],
    testdata_path: Path,
) -> list[str]:
    """
    Return a list of strings, like
    ```
    [
        "/some/path/fake_multiplex_shifted/cycle1",
        "/some/path/fake_multiplex_shifted/cycle2"
    ]
    ```
    """
    # Define old and new folders
    old_folder = str(testdata_path / "fake_multiplex")
    new_folder = old_folder.replace("fake_multiplex", "fake_multiplex_shifted")

    # Define output folders (one per multiplexing cycle)
    cycle_folders = [f"{new_folder}/cycle{ind}" for ind in (1, 2)]

    if os.path.isdir(new_folder):
        # If the shifted-images folder already exists, return immediately
        print(f"{new_folder} already exists")
        return cycle_folders
    else:
        # Copy the fake_multiplex folder into a new one
        shutil.copytree(old_folder, new_folder)
        # Loop over images of cycle2 and apply a shift
        for img_path in glob.glob(f"{cycle_folders[1]}/2020*.png"):
            print(f"Now shifting {img_path}")
            _shift_image(str(img_path))
        return cycle_folders


def test_multiplexing_registration(
    zenodo_images_multiplex_shifted: list[str],
):
    debug(zenodo_images_multiplex_shifted)

    # TODO: do something with this input folders
