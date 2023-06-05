# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                   |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| fractal\_tasks\_core/\_\_init\_\_.py                   |        4 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/\_utils.py                        |       34 |       34 |       10 |        0 |      0% |     19-98 |
| fractal\_tasks\_core/cellpose\_segmentation.py         |      224 |       29 |       74 |       21 |     82% |99-140, 243, 260-263, 294-301, 305->313, 309-310, 327->333, 360->369, 363, 371->387, 379, 394, 399, 414->418, 415, 422, 464, 488->492, 537, 555->573, 567, 644, 666-668 |
| fractal\_tasks\_core/compress\_tif.py                  |       35 |       30 |       14 |        1 |     12% |37-68, 72-88 |
| fractal\_tasks\_core/copy\_ome\_zarr.py                |       71 |        4 |       32 |        7 |     89% |88, 91, 93->98, 155->143, 159->169, 181->186, 200-202 |
| fractal\_tasks\_core/create\_ome\_zarr.py              |      162 |       25 |       70 |       11 |     83% |97, 102, 155, 162-172, 180-181, 194-197, 232->241, 247, 278, 294-295, 298, 433-435 |
| fractal\_tasks\_core/create\_ome\_zarr\_multiplex.py   |      195 |       21 |       86 |       16 |     87% |95, 104, 111, 116, 120, 127, 145, 176, 197-200, 227, 265->270, 276, 282, 299, 312-313, 319, 484-486 |
| fractal\_tasks\_core/illumination\_correction.py       |       99 |       18 |       34 |       11 |     77% |61, 82-86, 122, 126, 129-134, 147-150, 187, 195, 205, 218-219, 274-276 |
| fractal\_tasks\_core/lib\_ROI\_overlaps.py             |      119 |        5 |       38 |        5 |     94% |129, 179, 219, 301, 312 |
| fractal\_tasks\_core/lib\_channels.py                  |       84 |       10 |       48 |        5 |     89% |41-42, 48, 82-87, 162, 178, 234-235, 261 |
| fractal\_tasks\_core/lib\_glob.py                      |       19 |        1 |        8 |        1 |     93% |        28 |
| fractal\_tasks\_core/lib\_masked\_loading.py           |       61 |       42 |       16 |        2 |     27% |88-178, 205-206, 232-233, 243 |
| fractal\_tasks\_core/lib\_metadata\_parsing.py         |       99 |        7 |       30 |        8 |     88% |96, 113, 226, 233, 235, 274, 318->326, 367 |
| fractal\_tasks\_core/lib\_parse\_filename\_metadata.py |       33 |        2 |       12 |        2 |     91% |   98, 105 |
| fractal\_tasks\_core/lib\_pyramid\_creation.py         |       26 |        3 |       10 |        3 |     83% |62, 75, 90 |
| fractal\_tasks\_core/lib\_read\_fractal\_metadata.py   |       34 |        3 |       14 |        2 |     90% |48, 106-107 |
| fractal\_tasks\_core/lib\_regions\_of\_interest.py     |      138 |       27 |       34 |        6 |     80% |149, 236-238, 282-308, 385, 387, 395-396, 398 |
| fractal\_tasks\_core/lib\_upscale\_array.py            |       71 |       11 |       46 |        8 |     80% |61, 72, 89, 94-95, 111, 126, 156, 188-193 |
| fractal\_tasks\_core/lib\_zattrs\_utils.py             |       42 |        7 |       22 |        6 |     80% |43, 47, 58->57, 61, 66-73, 118 |
| fractal\_tasks\_core/maximum\_intensity\_projection.py |       49 |        5 |       14 |        4 |     86% |59, 93, 112, 135-137 |
| fractal\_tasks\_core/napari\_workflows\_wrapper.py     |      254 |       22 |      122 |       16 |     89% |160, 172, 177, 214, 284, 289-299, 312, 317, 357-361, 369, 390, 478->465, 512-517, 524->526, 581, 606-608 |
| fractal\_tasks\_core/tools/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/tools/lib\_metadata\_checks.py    |       52 |        1 |       30 |        1 |     98% |        28 |
| fractal\_tasks\_core/yokogawa\_to\_ome\_zarr.py        |       84 |       10 |       24 |        5 |     84% |85, 127, 169, 209-213, 219-221 |
|                                              **TOTAL** | **1989** |  **317** |  **788** |  **141** | **82%** |           |


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