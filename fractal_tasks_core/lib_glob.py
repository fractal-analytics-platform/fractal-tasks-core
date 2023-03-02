"""
Auxiliary functions related to globbing (i.e. listing some items from a
directory)
"""
import logging
from glob import glob
from typing import Sequence


def glob_with_multiple_patterns(
    *,
    folder: str,
    patterns: Sequence[str] = None,
) -> set[str]:
    """
    List all files and folders in a folder that simultaneously match a series
    of glob patterns

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
    logging.info(f"[glob_with_multiple_patterns] {patterns=}")

    items = None
    for pattern in patterns:
        new_matches = glob(f"{actual_folder}/{pattern}")
        if items is None:
            items = set(new_matches)
        else:
            items = items.intersection(new_matches)
    items = items or set()

    logging.info(f"[glob_with_multiple_patterns] Found {len(items)} items")
    return items
