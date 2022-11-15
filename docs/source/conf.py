from pathlib import Path
from typing import Any
from typing import List

from sphinx.application import Sphinx
from sphinx.ext import apidoc

project = "Fractal Tasks Core"
copyright = (
    "2022, Friedrich Miescher Institute for Biomedical Research and "
    "University of Zurich"
)
version = "0.3.4"
language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx_autodoc_typehints",
    "autodocsumm",
    "sphinx_autodoc_defaultargs",
]

autodoc_default_options = {"autosummary": True}
autodata_content = "both"
source_suffix = ".rst"
exclude_patterns = []
gettext_compact = False

master_doc = "index"
github_url = "https://github.com/fractal-analytics-platform/fractal-tasks-core"

# sphinx_rtd_theme config
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "sticky_navigation": True,
    "titles_only": True,
    "navigation_depth": 5,
    "prev_next_buttons_location": None,
    "display_version": True,
    "style_external_links": True,
}
html_context = {}

# This prolog is useful for both sphinx_autodoc_defaultargs and the "Edit on
# github" button
rst_prolog = (
    """
:github_url: https://github.com/fractal-analytics-platform/fractal-tasks-core

.. |default| raw:: html

    <div class="default-value-section">"""
    + ' <span class="default-value-label">Default:</span>'
)


# Extensions to theme docs
def setup(app):

    # apidoc (see https://github.com/readthedocs/readthedocs.org/issues/1139)
    source_dir = Path(__file__).parent.absolute()
    package_dir = str(source_dir / "../../fractal_tasks_core")
    api_files_dir = str(source_dir / "api_files")
    templates_dir = str(source_dir / "../apidoc_templates")
    app.connect(
        "builder-inited",
        lambda _: apidoc.main(
            [
                "-o",
                api_files_dir,
                "-d2",
                "-feMT",
                f"--templatedir={templates_dir}",
                package_dir,
            ]
        ),
    )

    # What follows is taken from https://stackoverflow.com/a/68913808,
    # and used to remove each indented block following a line starting
    # with "Copyright"
    what = None

    def process(
        app: Sphinx,
        what_: str,
        name: str,
        obj: Any,
        options: Any,
        lines: List[str],
    ) -> None:
        if what and what_ not in what:
            return
        orig_lines = lines[:]
        ignoring = False
        new_lines = []
        for i, line in enumerate(orig_lines):
            if line.startswith("Copyright"):
                # We will start ignoring everything indented after this
                ignoring = True
            else:
                # if the line startswith anything but a space stop
                # ignoring the indented region.
                if ignoring and line and not line.startswith(" "):
                    ignoring = False
            if not ignoring:
                new_lines.append(line)
        lines[:] = new_lines
        # make sure there is a blank line at the end
        if lines and lines[-1]:
            lines.append("")

    app.connect("autodoc-process-docstring", process)
    return app
