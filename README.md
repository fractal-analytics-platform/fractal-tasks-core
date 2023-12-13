# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| fractal\_tasks\_core/\_\_init\_\_.py                                |        5 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/cellvoyager/\_\_init\_\_.py                    |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/cellvoyager/filenames.py                       |       52 |        1 |       20 |        1 |     97% |       148 |
| fractal\_tasks\_core/cellvoyager/metadata.py                        |       99 |        7 |       30 |        8 |     88% |94, 111, 225, 232, 234, 274, 319->327, 370 |
| fractal\_tasks\_core/channels.py                                    |      188 |        1 |       98 |        4 |     98% |28, 84->83, 119->118, 465->467 |
| fractal\_tasks\_core/labels.py                                      |       40 |        0 |       14 |        0 |    100% |           |
| fractal\_tasks\_core/masked\_loading.py                             |       62 |        8 |       14 |        5 |     83% |92, 108, 137-144, 158, 168 |
| fractal\_tasks\_core/ngff/\_\_init\_\_.py                           |        6 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/ngff/specs.py                                  |      122 |        0 |       48 |        8 |     95% |111->110, 155->154, 183->182, 199->198, 206->205, 210->209, 217->216, 258->257 |
| fractal\_tasks\_core/ngff/zarr\_utils.py                            |       47 |        3 |        6 |        0 |     94% |     77-82 |
| fractal\_tasks\_core/pyramids.py                                    |       33 |        0 |       10 |        0 |    100% |           |
| fractal\_tasks\_core/roi/\_\_init\_\_.py                            |        4 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/roi/\_overlaps\_common.py                      |       22 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/roi/load\_region.py                            |       16 |        0 |       10 |        0 |    100% |           |
| fractal\_tasks\_core/roi/v1.py                                      |      156 |        1 |       34 |        1 |     99% |       164 |
| fractal\_tasks\_core/roi/v1\_checks.py                              |       37 |        0 |       18 |        0 |    100% |           |
| fractal\_tasks\_core/roi/v1\_overlaps.py                            |      145 |        5 |       62 |        6 |     95% |72, 132, 173, 255, 266, 383->390 |
| fractal\_tasks\_core/tables/\_\_init\_\_.py                         |       17 |        0 |        4 |        0 |    100% |           |
| fractal\_tasks\_core/tables/v1.py                                   |       94 |        0 |       40 |        3 |     98% |33->32, 51->50, 110->113 |
| fractal\_tasks\_core/tasks/\_\_init\_\_.py                          |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/tasks/\_utils.py                               |       29 |        5 |        8 |        1 |     78% |33-35, 68-71 |
| fractal\_tasks\_core/tasks/apply\_registration\_to\_ROI\_tables.py  |       84 |        5 |       28 |        6 |     90% |37->36, 81->83, 124, 136, 250, 291-293 |
| fractal\_tasks\_core/tasks/apply\_registration\_to\_image.py        |      113 |       17 |       38 |        9 |     79% |46->45, 104, 153-154, 156->180, 186->185, 190-202, 204->229, 214-216, 240, 337-350, 375-377 |
| fractal\_tasks\_core/tasks/calculate\_registration\_image\_based.py |       98 |        6 |       22 |        6 |     90% |45->44, 170, 180, 199, 244, 360-362 |
| fractal\_tasks\_core/tasks/cellpose\_segmentation.py                |      208 |       23 |       68 |       12 |     86% |105-146, 150->149, 257, 265-268, 311-317, 325-326, 364, 372->377, 381, 453, 477->481, 558->574, 568, 659-661 |
| fractal\_tasks\_core/tasks/copy\_ome\_zarr.py                       |       71 |        4 |       32 |        7 |     89% |39->38, 102, 105, 171->159, 173->182, 198->203, 220-222 |
| fractal\_tasks\_core/tasks/create\_ome\_zarr.py                     |      162 |       25 |       72 |       11 |     83% |50->49, 122, 124, 177, 184-194, 202-203, 216-219, 273, 304, 320-321, 324, 468-470 |
| fractal\_tasks\_core/tasks/create\_ome\_zarr\_multiplex.py          |      189 |       19 |       80 |       14 |     88% |52->51, 127, 134, 138, 147, 165, 196, 217-220, 247, 298, 304, 321, 334-335, 341, 513-515 |
| fractal\_tasks\_core/tasks/illumination\_correction.py              |      105 |       14 |       34 |        9 |     82% |62, 83-87, 96->95, 151, 172-175, 221, 236, 249-250, 305-307 |
| fractal\_tasks\_core/tasks/import\_ome\_zarr.py                     |       93 |       10 |       36 |        9 |     85% |70, 72, 84->95, 95->110, 110->exit, 114-123, 156->155, 204, 253->270, 279-281 |
| fractal\_tasks\_core/tasks/maximum\_intensity\_projection.py        |       48 |        3 |        8 |        3 |     89% |32->31, 67, 129-131 |
| fractal\_tasks\_core/tasks/napari\_workflows\_wrapper.py            |      244 |       20 |      118 |       15 |     90% |65->64, 153-155, 192, 297, 304, 310-315, 320, 351, 356, 396-400, 423, 526->513, 565-570, 577->579, 661-663 |
| fractal\_tasks\_core/tasks/napari\_workflows\_wrapper\_models.py    |       31 |        0 |       12 |        3 |     93% |27->26, 54->53, 66->65 |
| fractal\_tasks\_core/tasks/yokogawa\_to\_ome\_zarr.py               |       94 |        5 |       22 |        5 |     91% |64->63, 113, 170, 221, 271-273 |
| fractal\_tasks\_core/upscale\_array.py                              |       74 |        7 |       48 |        6 |     88% |63, 91, 96->100, 113, 128, 192-197 |
| fractal\_tasks\_core/utils.py                                       |       68 |        1 |       28 |        1 |     98% |        71 |
| fractal\_tasks\_core/zarr\_utils.py                                 |       33 |        0 |        6 |        1 |     97% |    81->85 |
|                                                           **TOTAL** | **2889** |  **190** | **1068** |  **154** | **91%** |           |


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