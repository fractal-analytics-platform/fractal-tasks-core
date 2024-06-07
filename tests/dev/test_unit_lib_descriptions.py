from devtools import debug

from fractal_tasks_core.dev.lib_descriptions import (
    _get_class_attrs_descriptions,
)
from fractal_tasks_core.dev.lib_descriptions import (
    _get_function_args_descriptions,
)


def test_get_function_args_descriptions():
    args_descriptions = _get_function_args_descriptions(
        package_name="fractal_tasks_core",
        module_relative_path="dev/lib_signature_constraints.py",
        function_name="_extract_function",
    )
    debug(args_descriptions)
    assert args_descriptions.keys() == set(
        ("package_name", "module_relative_path", "function_name", "verbose")
    )


def test_get_class_attrs_descriptions():
    attrs_descriptions = _get_class_attrs_descriptions(
        package_name="fractal_tasks_core",
        module_relative_path="channels.py",
        class_name="ChannelInputModel",
    )
    debug(attrs_descriptions)
    assert attrs_descriptions.keys() == set(("wavelength_id", "label"))
