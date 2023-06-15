import pytest
from devtools import debug

from fractal_tasks_core.tasks._input_models import BaseChannel
from fractal_tasks_core.tasks._input_models import (
    NapariWorkflowsInput,
)
from fractal_tasks_core.tasks._input_models import (
    NapariWorkflowsOutput,
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


def test_NapariWorkflowsInput():

    # Invalid

    with pytest.raises(ValueError) as e:
        NapariWorkflowsInput(type="invalid")
    debug(e.value)

    with pytest.raises(ValueError) as e:
        NapariWorkflowsInput(type="image")
    debug(e.value)

    with pytest.raises(ValueError) as e:
        NapariWorkflowsInput(type="label")
    debug(e.value)

    # Valid

    spec = NapariWorkflowsInput(type="label", label_name="name")
    assert spec.type
    assert spec.label_name
    assert not spec.channel

    spec = NapariWorkflowsInput(type="image", channel=dict(label="something"))
    assert spec.type
    assert not spec.label_name
    assert spec.channel


def test_NapariWorkflowsOutput():

    # Invalid

    with pytest.raises(ValueError) as e:
        NapariWorkflowsOutput(type="invalid")
    debug(e.value)

    with pytest.raises(ValueError) as e:
        NapariWorkflowsOutput(
            type="label",
            table_name="something",
        )
    debug(e.value)

    with pytest.raises(ValueError) as e:
        NapariWorkflowsOutput(
            type="label",
        )
    debug(e.value)

    with pytest.raises(ValueError) as e:
        NapariWorkflowsOutput(
            type="dataframe",
            label_name="something",
        )
    debug(e.value)

    with pytest.raises(ValueError) as e:
        NapariWorkflowsOutput(
            type="dataframe",
        )
    debug(e.value)

    # Valid

    specs = NapariWorkflowsOutput(
        type="label",
        label_name="label_DAPI",
    )
    debug(specs)
    assert specs.type
    assert specs.label_name

    specs = NapariWorkflowsOutput(
        type="dataframe",
        table_name="table_DAPI",
    )
    debug(specs)
    assert specs.type
    assert specs.table_name
