import json
from pathlib import Path

import pytest
import requests
from devtools import debug
from jsonschema import validate

import fractal_tasks_core


@pytest.mark.xfail(
    reason=(
        "See https://github.com/fractal-analytics-platform/"
        "fractal-server/issues/1347"
    )
)
def test_valid_manifest(tmp_path):
    """
    NOTE: to avoid adding a fractal-server dependency, we simply download the
    relevant file.
    """
    # Download JSON Schema for ManifestV1
    url = (
        "https://raw.githubusercontent.com/fractal-analytics-platform/"
        "fractal-server/main/"
        "fractal_server/app/schemas/json_schemas/manifest.json"
    )
    r = requests.get(url)
    with (tmp_path / "manifest_schema.json").open("wb") as f:
        f.write(r.content)
    with (tmp_path / "manifest_schema.json").open("r") as f:
        manifest_schema = json.load(f)

    module_dir = Path(fractal_tasks_core.__file__).parent
    with (module_dir / "__FRACTAL_MANIFEST__.json").open("r") as fin:
        manifest_dict = json.load(fin)

    debug(manifest_dict)
    validate(instance=manifest_dict, schema=manifest_schema)
