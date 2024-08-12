from devtools import debug

from fractal_tasks_core.cellvoyager.filenames import (
    glob_with_multiple_patterns,
)


def test_unit_glob_with_multiple_patterns(tmp_path):

    NUM_FILES = 20
    NUM_FOLDERS = 5
    for ind in range(NUM_FILES):
        (tmp_path / f"{ind:03d}.png").touch()
        (tmp_path / f"{ind:03d}.tif").touch()
    for ind in range(NUM_FOLDERS):
        (tmp_path / f"folder_{ind:03d}").mkdir()

    folder = str(tmp_path)
    debug(folder)

    # Look for all files and folders
    patterns = None
    items = glob_with_multiple_patterns(
        folder=folder, include_patterns=patterns
    )
    assert len(items) == 2 * NUM_FILES + NUM_FOLDERS

    # Look for all files and folders (with trailing slash)
    patterns = None
    items = glob_with_multiple_patterns(
        folder=f"{folder}/", include_patterns=patterns
    )
    assert len(items) == 2 * NUM_FILES + NUM_FOLDERS

    # Look for a subset of files with a single pattern
    patterns = ["*.tif"]
    items = glob_with_multiple_patterns(
        folder=folder, include_patterns=patterns
    )
    assert len(items) == NUM_FILES

    # Look for a subset of files with two patterns
    patterns = ["*.tif", "00*"]
    items = glob_with_multiple_patterns(
        folder=folder, include_patterns=patterns
    )
    assert len(items) == 10

    # Look for a subset of files with two patterns and a dummy catch-all one
    patterns = ["*.tif", "00*", "*"]
    items = glob_with_multiple_patterns(
        folder=folder, include_patterns=patterns
    )
    assert len(items) == 10

    # Look for an empty set of files
    patterns = ["*.tif", "00*", "*invalid_pattern*"]
    items = glob_with_multiple_patterns(
        folder=folder, include_patterns=patterns
    )
    assert len(items) == 0

    # Look for two patterns (one with no matches and one with matches)
    patterns = ["*invalid_pattern*", "0*"]
    items = glob_with_multiple_patterns(
        folder=folder, include_patterns=patterns
    )
    assert len(items) == 0


def test_unit_exclude_patterns(tmp_path):

    NUM_FILES = 20
    NUM_FOLDERS = 5
    for ind in range(NUM_FILES):
        (tmp_path / f"{ind:03d}.png").touch()
        (tmp_path / f"{ind:03d}.tif").touch()
    for ind in range(NUM_FOLDERS):
        (tmp_path / f"folder_{ind:03d}").mkdir()

    folder = str(tmp_path)
    debug(folder)

    # Test empty exclude patterns
    include_patterns = ["*.tif"]
    exclude_patterns = []
    items = glob_with_multiple_patterns(
        folder=folder,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    assert len(items) == NUM_FILES

    # Test non-overlapping exclude patterns
    include_patterns = ["*.tif"]
    exclude_patterns = ["*.png"]
    items = glob_with_multiple_patterns(
        folder=folder,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    assert len(items) == NUM_FILES

    # Test single exclude patterns
    include_patterns = ["*.tif"]
    exclude_patterns = ["019.tif"]
    items = glob_with_multiple_patterns(
        folder=folder,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    assert len(items) == NUM_FILES - 1

    # Test overlapping exclude patterns
    include_patterns = ["*.tif"]
    exclude_patterns = ["01*.tif"]
    items = glob_with_multiple_patterns(
        folder=folder,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    assert len(items) == NUM_FILES / 2

    # Test complete exclusion patterns
    include_patterns = ["*.tif"]
    exclude_patterns = ["*.tif"]
    items = glob_with_multiple_patterns(
        folder=folder,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    assert len(items) == 0

    # Test multiple exclude patterns
    include_patterns = ["*.tif"]
    exclude_patterns = ["019.tif", "018.tif", "017.tif", "016.tif"]
    items = glob_with_multiple_patterns(
        folder=folder,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    assert len(items) == NUM_FILES - 4
