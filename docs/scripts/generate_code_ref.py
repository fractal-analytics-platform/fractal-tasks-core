"""Generate code reference."""

from pathlib import Path

PKG_NAME = "fractal_tasks_core"
BASE_CODE_DIR = Path("fractal_tasks_core")
BASE_DOCS_DIR = Path("docs/code_reference")

# ---------------------------


def get_subpackages(base_path: Path) -> list[Path]:
    """Get subpackages of `base_path`."""
    print(f"[get_subpackages] {base_path}")
    subpkgs = [
        subpkg_dir
        for subpkg_dir in base_path.glob("*")
        if (subpkg_dir.is_dir() and (subpkg_dir / "__init__.py").exists())
    ]
    subpkgs = sorted(subpkgs)
    return subpkgs


def _sort_key(path: Path) -> int:
    """List private modules first."""
    return 0 if path.name.startswith("_") else 1


def get_modules(base_path: Path) -> list[Path]:
    """Get modules in `base_path`."""
    print(f"[get_modules] {base_path}")
    modules = [m for m in base_path.glob("*.py") if m.name != "__init__.py"]
    modules = sorted(modules, key=_sort_key)
    return modules


def walk_and_build(subpkg_path: Path):
    """Recursively walk `subpkg_path`."""
    print(f"[walk_and_build {subpkg_path}] Start")
    relative_string_dots = (
        subpkg_path.relative_to(BASE_CODE_DIR).as_posix().replace("/", ".")
    ).strip(".")
    if relative_string_dots == ".":
        relative_string_dots = ""
    subpkg_docs_path = BASE_DOCS_DIR / subpkg_path.relative_to(BASE_CODE_DIR)
    subpkg_docs_path.mkdir(parents=True, exist_ok=True)
    index_docs_path = subpkg_docs_path / "index.md"

    with index_docs_path.open("w") as f:
        title = (".".join([PKG_NAME, relative_string_dots])).strip(".")
        f.write(f"# `{title}`\n\n")

        identifier = f"{PKG_NAME}.{relative_string_dots}".strip(".")
        f.write(f"::: {identifier}\n\n")

        subpkgs = get_subpackages(subpkg_path)
        if subpkgs:
            f.write("## Subpackages\n\n")
            for subpkg in subpkgs:
                relative_subpkg_path = subpkg.name
                f.write(f"### {relative_subpkg_path}\n\n")
                f.write(f"- [{relative_subpkg_path}](./{relative_subpkg_path})\n\n")
                walk_and_build(subpkg)
            f.write("\n")

        modules = get_modules(subpkg_path)
        for module in modules:
            relative_module_path = module.relative_to(BASE_CODE_DIR).with_suffix("")
            relative_module_string = relative_module_path.as_posix()
            relative_module_string_dots = relative_module_string.replace("/", ".")
            module_docs_path = BASE_DOCS_DIR / relative_module_path.with_suffix(".md")
            with module_docs_path.open("w") as f1:
                f1.write(f"::: {PKG_NAME}.{relative_module_string_dots}\n")

    print(f"[walk_and_build {subpkg_path}] End")


if __name__ == "__main__":
    walk_and_build(BASE_CODE_DIR)
