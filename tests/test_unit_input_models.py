import pytest
from devtools import debug

from fractal_tasks_core.lib_input_models import Channel
from fractal_tasks_core.lib_input_models import (
    NapariWorkflowsInput,
)
from fractal_tasks_core.lib_input_models import (
    NapariWorkflowsOutput,
)


def test_Channel():

    # Valid

    c = Channel(wavelength_id="wavelength_id")
    debug(c)
    assert c.wavelength_id
    assert not c.label

    c = Channel(label="label")
    debug(c)
    assert not c.wavelength_id
    assert c.label

    # Invalid

    with pytest.raises(ValueError) as e:
        Channel()
    debug(e.value)

    with pytest.raises(ValueError) as e:
        Channel(label="label", wavelength_id="wavelength_id")
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

    with pytest.raises(ValueError) as e:
        NapariWorkflowsOutput(
            type="dataframe",
            table_name="something",
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
        label_name="label_DAPI",
        table_name="table_DAPI",
    )
    debug(specs)
    assert specs.type
    assert specs.table_name
