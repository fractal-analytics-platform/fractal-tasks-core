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

    with pytest.raises(ValueError) as e:
        NapariWorkflowsOutputSpecsItem(type="invalid")
    debug(e.value)

    with pytest.raises(ValueError) as e:
        NapariWorkflowsOutputSpecsItem(
            type="label",
            channel=dict(label="DAPI"),
            table_name="something",
        )
    debug(e.value)

    with pytest.raises(ValueError) as e:
        NapariWorkflowsOutputSpecsItem(
            type="dataframe",
            channel=dict(label="DAPI"),
            label_name="something",
        )
    debug(e.value)

    specs = NapariWorkflowsOutputSpecsItem(
        type="label",
        channel=dict(label="DAPI"),
        label_name="label_DAPI",
    )
    debug(specs)

    specs = NapariWorkflowsOutputSpecsItem(
        type="dataframe",
        channel=dict(label="DAPI"),
        table_name="table_DAPI",
    )
    debug(specs)
