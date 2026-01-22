import pytest
from devtools import debug

from fractal_tasks_core.channels import ChannelInputModel


def test_Channel():
    # Valid

    c = ChannelInputModel(wavelength_id="wavelength_id")
    debug(c)
    assert c.wavelength_id
    assert not c.label

    c = ChannelInputModel(label="label")
    debug(c)
    assert not c.wavelength_id
    assert c.label

    # Invalid

    with pytest.raises(ValueError) as e:
        ChannelInputModel()
    debug(e.value)

    with pytest.raises(ValueError) as e:
        ChannelInputModel(label="label", wavelength_id="wavelength_id")
    debug(e.value)
