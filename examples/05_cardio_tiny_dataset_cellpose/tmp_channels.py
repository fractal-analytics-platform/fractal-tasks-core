from fractal_tasks_core.lib_channels import OmeroChannel
from fractal_tasks_core.lib_channels import Window

allowed_channels = [
    OmeroChannel(
        label="DAPI",
        wavelength_id="A01_C01",
        color="00FFFF",
        window=Window(start=0, end=700),
    ),
    OmeroChannel(
        wavelength_id="A01_C02",
        label="nanog",
        color="FF00FF",
        window=Window(start=0, end=180),
    ),
    OmeroChannel(
        wavelength_id="A02_C03",
        label="Lamin B1",
        color="FFFF00",
        window=Window(start=0, end=1500),
    ),
]
