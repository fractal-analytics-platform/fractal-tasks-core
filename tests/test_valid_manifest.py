import json
from pathlib import Path

import requests
from jsonschema import validate

import fractal_tasks_core


def test_valid_manifest(tmp_path):
    """
    NOTE: to avoid adding a fractal-server dependency, we simply download the
    relevant file.
    """
    # Download JSON Schema for ManifestV2
    url = (
        "https://raw.githubusercontent.com/fractal-analytics-platform/"
        "fractal-server/main/"
        "fractal_server/json_schemas/manifest_v2.json"
    )
    r = requests.get(url)
    with (tmp_path / "manifest_schema_v2.json").open("wb") as f:
        f.write(r.content)
    with (tmp_path / "manifest_schema_v2.json").open("r") as f:
        manifest_schema = json.load(f)

    # Load tasks-package manifest
    module_dir = Path(fractal_tasks_core.__file__).parent
    with (module_dir / "__FRACTAL_MANIFEST__.json").open("r") as fin:
        manifest_dict = json.load(fin)

    # Validate manifest against JSON Schema
    validate(instance=manifest_dict, schema=manifest_schema)
