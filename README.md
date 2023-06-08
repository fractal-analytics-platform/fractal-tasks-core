# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| fractal\_tasks\_core/\_\_init\_\_.py                         |        4 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_ROI\_overlaps.py                   |      157 |        5 |       62 |        6 |     95% |131, 181, 221, 303, 314, 455->462 |
| fractal\_tasks\_core/lib\_channels.py                        |       84 |       10 |       48 |        5 |     89% |41-42, 48, 82-87, 162, 178, 234-235, 261 |
| fractal\_tasks\_core/lib\_glob.py                            |       19 |        1 |        8 |        1 |     93% |        28 |
| fractal\_tasks\_core/lib\_masked\_loading.py                 |       61 |       42 |       16 |        2 |     27% |88-178, 205-206, 232-233, 243 |
| fractal\_tasks\_core/lib\_metadata\_parsing.py               |       99 |        7 |       30 |        8 |     88% |96, 113, 226, 233, 235, 274, 318->326, 367 |
| fractal\_tasks\_core/lib\_parse\_filename\_metadata.py       |       33 |        2 |       12 |        2 |     91% |   98, 105 |
| fractal\_tasks\_core/lib\_pyramid\_creation.py               |       26 |        3 |       10 |        3 |     83% |62, 75, 90 |
| fractal\_tasks\_core/lib\_read\_fractal\_metadata.py         |       34 |        3 |       14 |        2 |     90% |48, 106-107 |
| fractal\_tasks\_core/lib\_regions\_of\_interest.py           |      138 |       27 |       34 |        6 |     80% |149, 236-238, 282-308, 385, 387, 395-396, 398 |
| fractal\_tasks\_core/lib\_upscale\_array.py                  |       71 |       11 |       46 |        8 |     80% |61, 72, 89, 94-95, 111, 126, 156, 188-193 |
| fractal\_tasks\_core/lib\_zattrs\_utils.py                   |       42 |        7 |       22 |        6 |     80% |43, 47, 58->57, 61, 66-73, 118 |
| fractal\_tasks\_core/tasks/\_\_init\_\_.py                   |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/tasks/\_utils.py                        |       28 |        5 |        8 |        1 |     78% |35-37, 69-72 |
| fractal\_tasks\_core/tasks/cellpose\_segmentation.py         |      226 |       29 |       74 |       21 |     82% |100-141, 245, 262-265, 296-303, 307->315, 311-312, 329->335, 362->371, 365, 373->389, 381, 396, 401, 416->420, 417, 424, 466, 490->494, 539, 557->575, 569, 646, 668-670 |
| fractal\_tasks\_core/tasks/compress\_tif.py                  |       35 |       30 |       14 |        1 |     12% |37-68, 72-88 |
| fractal\_tasks\_core/tasks/copy\_ome\_zarr.py                |       73 |        4 |       32 |        7 |     90% |90, 93, 95->100, 157->145, 161->171, 183->188, 202-204 |
| fractal\_tasks\_core/tasks/create\_ome\_zarr.py              |      164 |       25 |       70 |       11 |     83% |99, 104, 157, 164-174, 182-183, 196-199, 234->243, 249, 280, 296-297, 300, 435-437 |
| fractal\_tasks\_core/tasks/create\_ome\_zarr\_multiplex.py   |      197 |       21 |       86 |       16 |     87% |97, 106, 113, 118, 122, 129, 147, 178, 199-202, 229, 267->272, 278, 284, 301, 314-315, 321, 486-488 |
| fractal\_tasks\_core/tasks/illumination\_correction.py       |      101 |       18 |       34 |       11 |     77% |62, 83-87, 124, 128, 131-136, 149-152, 189, 197, 207, 220-221, 276-278 |
| fractal\_tasks\_core/tasks/maximum\_intensity\_projection.py |       51 |        5 |       14 |        4 |     86% |61, 95, 114, 137-139 |
| fractal\_tasks\_core/tasks/napari\_workflows\_wrapper.py     |      256 |       25 |      122 |       17 |     88% |125-127, 162, 174, 179, 216, 286, 291-301, 314, 319, 359-363, 371, 392, 480->467, 514-519, 526->528, 583, 608-610 |
| fractal\_tasks\_core/tasks/yokogawa\_to\_ome\_zarr.py        |       86 |       10 |       24 |        5 |     85% |87, 129, 171, 211-215, 221-223 |
|                                                    **TOTAL** | **1985** |  **290** |  **780** |  **143** | **83%** |           |


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