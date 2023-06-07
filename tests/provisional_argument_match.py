from inspect import signature
from typing import _UnionGenericAlias
from typing import Optional

import pydantic
from devtools import debug


def fun(
    *,
    x: int,
    y: Optional[int] = None,
    z: str = "asd",
    w,
):
    return


class Args(pydantic.BaseModel):
    x: int
    y: Optional[int]
    z: str = "xx"


debug(fun.__annotations__)
debug(fun.__defaults__)
debug(fun.__kwdefaults__)
sig = signature(fun)
debug(sig)
debug(sig.parameters)
debug(sig.parameters["x"])
debug(sig.parameters["x"].annotation)
debug(sig.parameters["y"])
debug(sig.parameters["z"])
debug(sig.parameters["w"])
debug(sig.parameters["w"].annotation)

debug(Args.schema())
schema = Args.schema()
schema_properties = schema["properties"]

required = []
for name, _type in fun.__annotations__.items():
    debug(name)
    debug(_type)
    assert name in schema_properties

    if isinstance(_type, _UnionGenericAlias):
        required.append(name)

    class SingleArg(pydantic.BaseModel):
        single_attribute: _type

    debug(SingleArg.schema()["properties"]["single_attribute"])
    debug(schema_properties[name])

    if (
        SingleArg.schema()["properties"]["single_attribute"]["type"]
        != schema_properties[name]["type"]
    ):
        raise ValueError()

debug(required)
debug(schema["required"])
assert set(required) == set(schema["required"])
