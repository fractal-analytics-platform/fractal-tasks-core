# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                   |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| fractal\_tasks\_core/\_\_init\_\_.py                   |        4 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/\_utils.py                        |       34 |       34 |       10 |        0 |      0% |     19-96 |
| fractal\_tasks\_core/cellpose\_segmentation.py         |      250 |       56 |       74 |       22 |     75% |99-140, 239, 256-259, 290-297, 301->309, 305-306, 323->329, 343, 356->365, 359, 367->383, 375, 390, 395, 410->414, 411, 418, 460, 484->488, 533, 551->569, 563, 640, 662-693 |
| fractal\_tasks\_core/compress\_tif.py                  |       35 |       30 |       14 |        1 |     12% |37-68, 72-88 |
| fractal\_tasks\_core/copy\_ome\_zarr.py                |       80 |       13 |       32 |        7 |     82% |88, 91, 93->98, 155->143, 159->169, 181->186, 200-212 |
| fractal\_tasks\_core/create\_ome\_zarr.py              |      174 |       37 |       70 |       11 |     79% |97, 102, 155, 162-172, 180-181, 194-197, 232->241, 247, 278, 294-295, 298, 433-448 |
| fractal\_tasks\_core/create\_ome\_zarr\_multiplex.py   |      205 |       33 |       86 |       16 |     83% |95, 104, 111, 116, 120, 127, 145, 176, 197-200, 224, 262->267, 273, 279, 296, 309-310, 313, 478-493 |
| fractal\_tasks\_core/illumination\_correction.py       |      110 |       29 |       34 |       11 |     71% |61, 82-86, 122, 126, 129-134, 147-150, 187, 195, 205, 218-219, 275-291 |
| fractal\_tasks\_core/lib\_ROI\_overlaps.py             |      119 |        5 |       38 |        5 |     94% |129, 179, 219, 301, 312 |
| fractal\_tasks\_core/lib\_channels.py                  |       84 |       10 |       48 |        5 |     89% |41-42, 48, 82-87, 162, 178, 234-235, 261 |
| fractal\_tasks\_core/lib\_glob.py                      |       19 |        1 |        8 |        1 |     93% |        28 |
| fractal\_tasks\_core/lib\_masked\_loading.py           |       61 |       42 |       16 |        2 |     27% |88-178, 205-206, 232-233, 243 |
| fractal\_tasks\_core/lib\_metadata\_parsing.py         |       99 |        7 |       30 |        8 |     88% |96, 113, 226, 233, 235, 274, 318->326, 367 |
| fractal\_tasks\_core/lib\_parse\_filename\_metadata.py |       33 |        2 |       12 |        2 |     91% |   98, 105 |
| fractal\_tasks\_core/lib\_pyramid\_creation.py         |       26 |        3 |       10 |        3 |     83% |62, 75, 90 |
| fractal\_tasks\_core/lib\_read\_fractal\_metadata.py   |       34 |        3 |       14 |        2 |     90% |48, 106-107 |
| fractal\_tasks\_core/lib\_regions\_of\_interest.py     |      134 |       27 |       28 |        6 |     78% |149, 233-235, 279-305, 379, 381, 389-390, 392 |
| fractal\_tasks\_core/lib\_upscale\_array.py            |       71 |       11 |       46 |        8 |     80% |61, 72, 89, 94-95, 111, 126, 156, 188-193 |
| fractal\_tasks\_core/lib\_zattrs\_utils.py             |       42 |        7 |       22 |        6 |     80% |43, 47, 58->57, 61, 66-73, 118 |
| fractal\_tasks\_core/maximum\_intensity\_projection.py |       56 |       12 |       14 |        4 |     77% |59, 93, 112, 134-144 |
| fractal\_tasks\_core/napari\_workflows\_wrapper.py     |      265 |       36 |      120 |       16 |     86% |161, 173, 178, 215, 285, 290-300, 313, 318, 358-362, 370, 391, 479->466, 513-518, 525->527, 578, 601-618 |
| fractal\_tasks\_core/tools/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/tools/lib\_metadata\_checks.py    |       52 |        1 |       30 |        1 |     98% |        28 |
| fractal\_tasks\_core/yokogawa\_to\_ome\_zarr.py        |       93 |       18 |       24 |        5 |     79% |86, 128, 170, 210-214, 220-231 |
|                                              **TOTAL** | **2080** |  **417** |  **780** |  **142** | **79%** |           |


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