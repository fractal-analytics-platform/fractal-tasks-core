import json
import sys
from pathlib import Path

import requests
from devtools import debug

import fractal_tasks_core


def test_valid_manifest(tmp_path):
    """
    NOTE: to avoid adding a fractal-server dependency, we simply download the
    relevant file.
    """

    url = (
        "https://raw.githubusercontent.com/fractal-analytics-platform/"
        "fractal-server/main/fractal_server/app/schemas/manifest.py"
    )
    r = requests.get(url)
    debug(tmp_path)
    with (tmp_path / "fractal_manifest.py").open("wb") as fout:
        fout.write(r.content)

    sys.path.append(str(tmp_path))
    from fractal_manifest import ManifestV1

    module_dir = Path(fractal_tasks_core.__file__).parent
    with (module_dir / "__FRACTAL_MANIFEST__.json").open("r") as fin:
        manifest_dict = json.load(fin)
    manifest = ManifestV1(**manifest_dict)
    debug(manifest)
