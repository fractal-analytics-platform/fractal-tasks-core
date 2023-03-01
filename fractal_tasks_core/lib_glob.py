"""
Auxiliary functions related to globbing (i.e. listing some items from a
directory)
"""
from glob import glob
from typing import Sequence


def glob_with_multiple_patterns(
    *,
    folder: str,
    patterns: Sequence[str] = None,
) -> set[str]:
    """
    List all file and folders in a folder that simultaneously match a series of
    glob patterns

    Arguments:
        :folder: TBD
        :patterns: TBD
    """

    if folder.endswith("/"):
        actual_folder = folder[:-1]
    else:
        actual_folder = folder[:]

    if not patterns:
        patterns = ["*"]

    items = None
    for pattern in patterns:
        new_matches = glob(f"{actual_folder}/{pattern}")
        if items:
            items = items.intersection(new_matches)
        else:
            items = set(new_matches)

    if items:
        return items
    else:
        return set()
