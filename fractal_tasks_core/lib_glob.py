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
    List all the items (files and folders) in a given folder that
    simultaneously match a series of glob patterns

    Arguments:
        :folder: Base folder where items will be searched.
        :patterns: If specified, the list of patterns (defined as in
                   https://docs.python.org/3/library/fnmatch.html) that item
                   names will match with.
    """

    # Sanitize base-folder path
    if folder.endswith("/"):
        actual_folder = folder[:-1]
    else:
        actual_folder = folder[:]

    # If not pattern is specified, look for *all* items in the base folder
    if not patterns:
        patterns = ["*"]

    # Combine multiple glob searches (via set intersection)
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
