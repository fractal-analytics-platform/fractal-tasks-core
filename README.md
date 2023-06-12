# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| fractal\_tasks\_core/\_\_init\_\_.py                         |        4 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_ROI\_overlaps.py                   |      157 |        5 |       62 |        6 |     95% |131, 181, 221, 303, 314, 455->462 |
| fractal\_tasks\_core/lib\_channels.py                        |      108 |        8 |       58 |        6 |     92% |29, 106, 139-144, 239, 293-294, 297->275, 304 |
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
| fractal\_tasks\_core/tasks/cellpose\_segmentation.py         |      226 |       29 |       74 |       21 |     82% |100-141, 292, 309-312, 343-350, 354->362, 358-359, 376->382, 409->418, 412, 420->436, 428, 443, 448, 463->467, 464, 471, 513, 537->541, 586, 604->622, 616, 693, 715-717 |
| fractal\_tasks\_core/tasks/compress\_tif.py                  |       35 |       30 |       14 |        1 |     12% |37-68, 72-88 |
| fractal\_tasks\_core/tasks/copy\_ome\_zarr.py                |       73 |        4 |       32 |        7 |     90% |109, 112, 114->119, 176->164, 180->190, 202->207, 221-223 |
| fractal\_tasks\_core/tasks/create\_ome\_zarr.py              |      165 |       25 |       70 |       11 |     83% |128, 133, 186, 193-203, 211-212, 225-228, 263->272, 278, 309, 325-326, 329, 464-466 |
| fractal\_tasks\_core/tasks/create\_ome\_zarr\_multiplex.py   |      198 |       21 |       86 |       16 |     87% |133, 142, 149, 154, 158, 165, 183, 214, 235-238, 265, 303->308, 314, 320, 337, 350-351, 357, 522-524 |
| fractal\_tasks\_core/tasks/illumination\_correction.py       |      102 |       18 |       34 |       11 |     77% |63, 84-88, 163, 167, 170-175, 188-191, 230, 238, 248, 261-262, 317-319 |
| fractal\_tasks\_core/tasks/maximum\_intensity\_projection.py |       51 |        5 |       14 |        4 |     86% |83, 117, 136, 159-161 |
| fractal\_tasks\_core/tasks/napari\_workflows\_wrapper.py     |      256 |       25 |      122 |       17 |     88% |158-160, 195, 207, 212, 249, 319, 324-334, 347, 352, 392-396, 404, 425, 513->500, 547-552, 559->561, 616, 641-643 |
| fractal\_tasks\_core/tasks/yokogawa\_to\_ome\_zarr.py        |       87 |       10 |       24 |        5 |     85% |117, 162, 204, 244-248, 254-256 |
|                                                    **TOTAL** | **2013** |  **288** |  **790** |  **144** | **83%** |           |


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