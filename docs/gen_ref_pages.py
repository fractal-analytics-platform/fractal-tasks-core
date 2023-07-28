from pathlib import Path
from typing import Iterable
from typing import Mapping

import mkdocs_gen_files
from mkdocs_gen_files import Nav


class CustomNav(Nav):
    """
    The original Nav class is part of mkdocs_gen_files
    (https://github.com/oprypin/mkdocs-gen-files)
    Original Copyright 2020 Oleh Prypin <oleh@pryp.in>
    License: MIT
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _items(cls, data: Mapping, level: int) -> Iterable[Nav.Item]:
        """
        Custom modification: rather than looping over data.items(), we loop
        over keys/values in a custom order (that is, we first include "tasks",
        then "dev", then all the rest)
        """
        sorted_keys = list(data.keys())
        if None in sorted_keys:
            sorted_keys.remove(None)
        sorted_keys = sorted(sorted_keys, key=str.casefold)
        if "dev" in sorted_keys:
            sorted_keys.remove("dev")
            sorted_keys = ["dev"] + sorted_keys
        if "tasks" in sorted_keys:
            sorted_keys.remove("tasks")
            sorted_keys = ["tasks"] + sorted_keys

        for key in sorted_keys:
            value = data[key]
            if key is not None:
                yield cls.Item(
                    level=level, title=key, filename=value.get(None)
                )
                yield from cls._items(value, level + 1)


nav = CustomNav()

for path in sorted(Path("fractal_tasks_core").rglob("*.py")):
    module_path = path.relative_to(".").with_suffix("")
    doc_path = path.relative_to(".").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    # Remove fractal_tasks_core from doc_path
    doc_path = Path("/".join(doc_path.as_posix().split("/")[1:]))

    # Remove fractal_tasks_core from parts, and skip the case where
    # parts=["fractal_tasks_core"]
    if parts[1:]:
        nav[parts[1:]] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"::: {identifier}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)


with mkdocs_gen_files.open(
    "reference/fractal_tasks_core/SUMMARY.md", "w"
) as nav_file:
    nav_file.writelines(nav.build_literate_nav())
