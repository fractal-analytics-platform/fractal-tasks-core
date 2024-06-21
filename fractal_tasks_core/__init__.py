import logging


def _check_pydantic_version():
    """
    Temporary check for pydantic version.
    To be removed after moving to pydantic v2 is complete.
    """
    import importlib.metadata
    from packaging import version

    pydantic_version = version.parse(importlib.metadata.version("pydantic"))
    pydantic_v1 = version.parse("1.10.16")
    pydantic_v2 = version.parse("2.6.3")
    if pydantic_version != pydantic_v1 and pydantic_version < pydantic_v2:
        raise ImportError(
            f"Pydantic version {pydantic_version} is not supported. "
            f"Please use version =={pydantic_v1} or  >={pydantic_v2}."
        )


_check_pydantic_version()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s; %(levelname)s; %(message)s"
)


__VERSION__ = "1.0.3a0"
__OME_NGFF_VERSION__ = "0.4"
__FRACTAL_TABLE_VERSION__ = "1"
