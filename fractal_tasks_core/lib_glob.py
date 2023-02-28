"""
Auxiliary functions related to globbing (i.e. listing some files from a
directory)
"""
from glob import glob
from typing import Optional


def glob_with_extension_and_pattern(
    *,
    folder: str,
    extension: str,
    pattern: Optional[str] = None,
):
    """
    TBD
    """

    list_files_extension = glob(f"{folder}/*.{extension}")
    if pattern:
        list_files_pattern = glob(f"{folder}/{pattern}")
        return tuple(
            set(list_files_extension).intersection(list_files_pattern)
        )
    else:
        return list_files_extension
