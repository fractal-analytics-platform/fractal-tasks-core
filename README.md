# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                   |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| fractal\_tasks\_core/\_\_init\_\_.py                   |        4 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/\_utils.py                        |       34 |       34 |       10 |        0 |      0% |     19-96 |
| fractal\_tasks\_core/cellpose\_segmentation.py         |      228 |       52 |       68 |       17 |     75% |106-147, 241, 287-294, 298->306, 302-303, 347, 356->370, 362, 372-375, 385, 390, 405->409, 406, 413, 456, 480->484, 531->552, 546, 613-643 |
| fractal\_tasks\_core/compress\_tif.py                  |       35 |       30 |       14 |        1 |     12% |37-68, 72-88 |
| fractal\_tasks\_core/copy\_ome\_zarr.py                |       78 |       13 |       32 |        7 |     82% |88, 91, 93->97, 154->142, 157->167, 173->178, 190-202 |
| fractal\_tasks\_core/create\_ome\_zarr.py              |      173 |       37 |       70 |       11 |     79% |97, 102, 155, 162-172, 180-181, 194-197, 232->241, 247, 278, 294-295, 298, 432-447 |
| fractal\_tasks\_core/create\_ome\_zarr\_multiplex.py   |      204 |       33 |       86 |       16 |     83% |95, 104, 111, 116, 120, 127, 145, 176, 197-200, 224, 262->267, 273, 279, 296, 309-310, 313, 477-492 |
| fractal\_tasks\_core/illumination\_correction.py       |      110 |       29 |       34 |       11 |     71% |61, 82-86, 122, 126, 129-134, 147-150, 187, 195, 205, 218-219, 275-291 |
| fractal\_tasks\_core/lib\_channels.py                  |       84 |       10 |       48 |        5 |     89% |41-42, 48, 82-87, 162, 178, 234-235, 261 |
| fractal\_tasks\_core/lib\_glob.py                      |       19 |        1 |        8 |        1 |     93% |        28 |
| fractal\_tasks\_core/lib\_metadata\_parsing.py         |       99 |        7 |       30 |        8 |     88% |96, 113, 226, 233, 235, 274, 318->326, 367 |
| fractal\_tasks\_core/lib\_parse\_filename\_metadata.py |       33 |        2 |       12 |        2 |     91% |   98, 105 |
| fractal\_tasks\_core/lib\_pyramid\_creation.py         |       26 |        3 |       10 |        3 |     83% |62, 75, 90 |
| fractal\_tasks\_core/lib\_read\_fractal\_metadata.py   |       34 |        3 |       14 |        2 |     90% |48, 106-107 |
| fractal\_tasks\_core/lib\_regions\_of\_interest.py     |      115 |       21 |       20 |        3 |     81% |148, 170, 230-232, 276-299 |
| fractal\_tasks\_core/lib\_remove\_FOV\_overlaps.py     |      103 |        5 |       34 |        5 |     93% |125, 172, 212, 294, 305 |
| fractal\_tasks\_core/lib\_upscale\_array.py            |       42 |        4 |       28 |        4 |     89% |56, 67, 79, 100 |
| fractal\_tasks\_core/lib\_zattrs\_utils.py             |       42 |        7 |       22 |        6 |     80% |43, 47, 58->57, 61, 66-73, 118 |
| fractal\_tasks\_core/maximum\_intensity\_projection.py |       56 |       12 |       14 |        4 |     77% |59, 93, 112, 134-144 |
| fractal\_tasks\_core/napari\_workflows\_wrapper.py     |      260 |       35 |      118 |       15 |     86% |161, 173, 178, 214, 284, 289-299, 312, 317, 357-361, 369, 390, 478->465, 512-517, 524->526, 589-606 |
| fractal\_tasks\_core/tools/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/tools/lib\_metadata\_checks.py    |       52 |        1 |       30 |        1 |     98% |        28 |
| fractal\_tasks\_core/yokogawa\_to\_ome\_zarr.py        |       93 |       18 |       24 |        5 |     79% |86, 128, 170, 210-214, 220-231 |
|                                              **TOTAL** | **1924** |  **357** |  **726** |  **127** | **80%** |           |


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