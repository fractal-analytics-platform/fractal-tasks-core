from devtools import debug

from fractal_tasks_core.dev.lib_descriptions import (
    _get_attributes_models_descriptions,
)


def test_get_args_model_descriptions():
    debug(_get_attributes_models_descriptions())
