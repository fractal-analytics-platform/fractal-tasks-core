#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import os

from setuptools import setup


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding="utf-8").read()


# Add your dependencies in requirements.txt
# Note: you can add test-specific requirements in tox.ini
requirements = []
with open("requirements.txt") as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)


setup(
    name="napari-skimage-regionprops-mock",
    author="Marcelo Zoccoler, Robert Haase",
    author_email="robert.haase@tu-dresden.de",
    license="BSD-3",
    description='MOCK OF "A regionprops table widget plugin for napari"',
    python_requires=">=3.8",
    install_requires=requirements,
    version="9.9.9",
    setup_requires=["setuptools_scm"],
)
