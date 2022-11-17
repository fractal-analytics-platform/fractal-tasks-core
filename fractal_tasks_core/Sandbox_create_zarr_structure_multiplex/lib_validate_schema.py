import json
import urllib
from typing import Dict

from devtools import debug
from jsonschema import validate

from fractal_tasks_core import __OME_NGFF_VERSION__


def validate_schema(*, path: str, type: str):
    url: str = (
        "https://raw.githubusercontent.com/ome/ngff/main/"
        f"{__OME_NGFF_VERSION__}/schemas/{type}.schema"
    )
    debug(url)
    with urllib.request.urlopen(url) as fin:
        schema: Dict = json.load(fin)
    debug(path)
    debug(type)
    with open(f"{path}/.zattrs", "r") as fin:
        zattrs = json.load(fin)
    validate(instance=zattrs, schema=schema)
