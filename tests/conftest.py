import json
import os
from pathlib import Path
from urllib.parse import unquote

import pytest
import requests  # type: ignore
import wget


@pytest.fixture(scope="session")
def testdata_path() -> Path:
    TEST_DIR = Path(__file__).parent
    return TEST_DIR / "data/"


@pytest.fixture(scope="session")
def zenodo_images(testdata_path):
    # Based on
    # https://github.com/dvolgyes/zenodo_get/blob/master/zenodo_get/zget.py

    url = "10.5281/zenodo.7059515"
    folder = str(testdata_path / (url.replace(".", "_").replace("/", "_")))
    if os.path.isdir(folder):
        print(f"{folder} already exists, skip")
        return Path(folder)
    os.makedirs(folder)
    url = "https://doi.org/" + url
    print(f"I will download {url} files to {folder}")

    r = requests.get(url)
    recordID = r.url.split("/")[-1]
    url = "https://zenodo.org/api/records/"
    r = requests.get(url + recordID)

    js = json.loads(r.text)
    files = js["files"]
    for f in files:
        fname = f["key"]
        link = f"https://zenodo.org/record/{recordID}/files/{fname}"
        print(link)
        link = unquote(link)
        wget.download(link, out=folder)
        print()
    return Path(folder)
