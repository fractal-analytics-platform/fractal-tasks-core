Docstrings should default to these style guides

### Module docstrings

Module docstring includes the copyright notice and a minimalistic description. Note that the copyright notice has to be indented exactly as in this example (first line not indented, other lines indented, and then a new line), since this is used by sphinx to ignore that part.
```python
"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Tommaso Comparin <tommaso.comparin@exact-lab.it>
    Marco Franzon <marco.franzon@exact-lab.it>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

This is a very nice module.
"""
```


### Function docstrings

See example below, or also https://sphinx-rtd-theme.readthedocs.io/en/stable/_modules/test_py_module/test.html#Foo


```python

def parse_filename(filename: str) -> Dict[str, str]:
    """
    One-line description of the function

    Additional lines here, to describe the function more in detail, provide
    examples, ...

    1) Filenames from UZH:
       20200812-Cardio[...]Cycle1_B03_T0001F036L01A01Z18C01.png
       with plate name 20200812-Cardio[...]Cycle1
    2) Filenames from FMI, with successful barcode reading:
       210305NAR005AAN_210416_164828_B11_T0001F006L01A04Z14C01.tif
       with plate name 210305NAR005AAN
    3) Filenames from FMI, with failed barcode reading:
       yymmdd_hhmmss_210416_164828_B11_T0001F006L01A04Z14C01.tif
       with plate name RS{yymmddhhmmss}

    We can include python code as here::

        print(1)
        print(2)


    :param filename: name of the image
    :returns: metadata dictionary
    """
```
