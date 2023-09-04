from setuptools import setup

setup(
    name="napari-skimage-regionprops-mock",
    description=(
        "A custom mock of napari-skimage-regionprops, "
        "only used for testing in fractal-tasks-core"
    ),
    install_requires=["numpy", "pandas"],
    version="9.9.9",
)
