import pytest
from devtools import debug

from fractal_tasks_core.tasks._input_models import BaseChannel
from fractal_tasks_core.tasks._input_models import (
    NapariWorkflowsOutputSpecsItem,
)


def test_BaseChannel():

    # Valid

    c = BaseChannel(wavelength_id="wavelength_id")
    debug(c)
    assert c.wavelength_id
    assert not c.label

    c = BaseChannel(label="label")
    debug(c)
    assert not c.wavelength_id
    assert c.label

    # Invalid

    with pytest.raises(ValueError) as e:
        BaseChannel()
    debug(e.value)

    with pytest.raises(ValueError) as e:
        BaseChannel(label="label", wavelength_id="wavelength_id")
    debug(e.value)


def test_NapariWorkflowsOutputSpecsItem():

    # Invalid

    with pytest.raises(ValueError) as e:
        NapariWorkflowsOutputSpecsItem(type="invalid")
    debug(e.value)

    with pytest.raises(ValueError) as e:
        NapariWorkflowsOutputSpecsItem(
            type="label",
            table_name="something",
        )
    debug(e.value)

    with pytest.raises(ValueError) as e:
        NapariWorkflowsOutputSpecsItem(
            type="label",
        )
    debug(e.value)

    with pytest.raises(ValueError) as e:
        NapariWorkflowsOutputSpecsItem(
            type="dataframe",
            label_name="something",
        )
    debug(e.value)

    with pytest.raises(ValueError) as e:
        NapariWorkflowsOutputSpecsItem(
            type="dataframe",
        )
    debug(e.value)

    # Valid

    specs = NapariWorkflowsOutputSpecsItem(
        type="label",
        label_name="label_DAPI",
    )
    debug(specs)
    assert specs.type
    assert specs.label_name

    specs = NapariWorkflowsOutputSpecsItem(
        type="dataframe",
        table_name="table_DAPI",
    )
    debug(specs)
    assert specs.type
    assert specs.table_name
