import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import TypeVar

from pydantic import BaseModel
from pydantic import validator


__all__ = ("TaskManifestV1", "ManifestV1")


class _TaskManifestBase(BaseModel):
    name: str
    input_type: str
    output_type: str
    default_args: Dict[str, Any]
    executable: Path
    meta: Dict[str, Any]


TaskManifestType = TypeVar("TaskManifestType", bound=_TaskManifestBase)


class _ManifestBase(BaseModel):
    manifest_version: str
    task_list: List[TaskManifestType]  # type: ignore


class TaskManifestV1(_TaskManifestBase):
    pass


class ManifestV1(_ManifestBase):
    task_list: List[TaskManifestV1]
    manifest_version: str

    @validator("manifest_version")
    def manifest_version_1(cls, value):
        if value != "1":
            raise ValueError("Wrong manifest version")


with open("__FRACTAL_MANIFEST__.json", "r") as fin:
    manifest_dict = json.load(fin)
print(json.dumps(manifest_dict, indent=4))
manifest = ManifestV1(**manifest_dict)
print(manifest)
