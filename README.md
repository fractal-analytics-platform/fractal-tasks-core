# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| fractal\_tasks\_core/\_\_init\_\_.py                         |        4 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_ROI\_overlaps.py                   |      158 |        5 |       62 |        6 |     95% |134, 184, 224, 306, 317, 458->465 |
| fractal\_tasks\_core/lib\_channels.py                        |      108 |        1 |       58 |        1 |     99% |        29 |
| fractal\_tasks\_core/lib\_glob.py                            |       19 |        0 |        8 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_input\_models.py                   |       56 |        0 |       12 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_masked\_loading.py                 |       61 |        9 |       16 |        6 |     81% |93, 104, 110, 139-146, 160, 170 |
| fractal\_tasks\_core/lib\_metadata\_parsing.py               |       99 |        7 |       30 |        8 |     88% |96, 113, 226, 233, 235, 274, 318->326, 367 |
| fractal\_tasks\_core/lib\_parse\_filename\_metadata.py       |       33 |        2 |       12 |        2 |     91% |   98, 105 |
| fractal\_tasks\_core/lib\_pyramid\_creation.py               |       27 |        0 |       10 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_read\_fractal\_metadata.py         |       34 |        3 |       14 |        2 |     90% |48, 106-107 |
| fractal\_tasks\_core/lib\_regions\_of\_interest.py           |      136 |        3 |       42 |        3 |     97% |155, 349, 351 |
| fractal\_tasks\_core/lib\_upscale\_array.py                  |       73 |        7 |       48 |        6 |     88% |63, 91, 96->100, 113, 128, 190-195 |
| fractal\_tasks\_core/lib\_zattrs\_utils.py                   |       51 |        2 |       28 |        2 |     95% |72->71, 89, 138 |
| fractal\_tasks\_core/tasks/\_\_init\_\_.py                   |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/tasks/\_utils.py                        |       28 |        5 |        8 |        1 |     78% |31-33, 65-68 |
| fractal\_tasks\_core/tasks/cellpose\_segmentation.py         |      238 |       28 |       78 |       16 |     85% |103-144, 283, 291-294, 325-331, 339-340, 395, 403->419, 411, 426, 431, 438, 461, 512, 536->540, 617->635, 629, 705, 726-728 |
| fractal\_tasks\_core/tasks/compress\_tif.py                  |       35 |       35 |       14 |        0 |      0% |     17-88 |
| fractal\_tasks\_core/tasks/copy\_ome\_zarr.py                |       70 |        4 |       30 |        6 |     90% |107, 110, 170->158, 174->184, 196->201, 215-217 |
| fractal\_tasks\_core/tasks/create\_ome\_zarr.py              |      165 |       25 |       70 |       10 |     83% |126, 128, 181, 188-198, 206-207, 220-223, 273, 304, 320-321, 324, 460-462 |
| fractal\_tasks\_core/tasks/create\_ome\_zarr\_multiplex.py   |      191 |       19 |       78 |       13 |     88% |131, 138, 142, 151, 169, 200, 221-224, 251, 299, 305, 322, 335-336, 342, 508-510 |
| fractal\_tasks\_core/tasks/illumination\_correction.py       |       99 |       17 |       32 |       10 |     78% |64, 85-89, 171, 175, 178-183, 196-199, 238, 253, 266-267, 322-324 |
| fractal\_tasks\_core/tasks/maximum\_intensity\_projection.py |       51 |        5 |       14 |        4 |     86% |83, 117, 136, 159-161 |
| fractal\_tasks\_core/tasks/napari\_workflows\_wrapper.py     |      265 |       24 |      128 |       18 |     89% |160-162, 199, 213, 218, 316, 323, 329-334, 339, 370, 375, 415-419, 448, 462, 543->530, 582-587, 594->596, 651, 676-678 |
| fractal\_tasks\_core/tasks/yokogawa\_to\_ome\_zarr.py        |       80 |        5 |       20 |        4 |     91% |113, 160, 202, 252-254 |
|                                                    **TOTAL** | **2081** |  **206** |  **812** |  **118** | **88%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/fractal-analytics-platform/fractal-tasks-core/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/fractal-analytics-platform/fractal-tasks-core/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Ffractal-analytics-platform%2Ffractal-tasks-core%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.