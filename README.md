# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                      |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| fractal\_tasks\_core/\_\_init\_\_.py                                      |        5 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/cellvoyager/\_\_init\_\_.py                          |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/cellvoyager/filenames.py                             |       61 |        1 |       26 |        1 |     98% |       167 |
| fractal\_tasks\_core/cellvoyager/metadata.py                              |      125 |        8 |       38 |        9 |     90% |113, 130, 310, 317, 329, 338, 379, 424->432, 475 |
| fractal\_tasks\_core/cellvoyager/wells.py                                 |       18 |        0 |        8 |        0 |    100% |           |
| fractal\_tasks\_core/channels.py                                          |      190 |        1 |       70 |        2 |     99% |30, 473->475 |
| fractal\_tasks\_core/labels.py                                            |       39 |        0 |       14 |        0 |    100% |           |
| fractal\_tasks\_core/masked\_loading.py                                   |       62 |        8 |       16 |        5 |     83% |92, 108, 139-146, 160, 170 |
| fractal\_tasks\_core/ngff/\_\_init\_\_.py                                 |        6 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/ngff/specs.py                                        |      157 |        0 |       32 |        0 |    100% |           |
| fractal\_tasks\_core/ngff/zarr\_utils.py                                  |       61 |       15 |        6 |        0 |     78% |78-83, 96-113 |
| fractal\_tasks\_core/pyramids.py                                          |       43 |        0 |       14 |        0 |    100% |           |
| fractal\_tasks\_core/roi/\_\_init\_\_.py                                  |        4 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/roi/\_overlaps\_common.py                            |       21 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/roi/load\_region.py                                  |       16 |        0 |       10 |        0 |    100% |           |
| fractal\_tasks\_core/roi/v1.py                                            |      170 |        2 |       34 |        2 |     98% |  164, 622 |
| fractal\_tasks\_core/roi/v1\_checks.py                                    |       36 |        0 |       12 |        0 |    100% |           |
| fractal\_tasks\_core/roi/v1\_overlaps.py                                  |      148 |        5 |       50 |        6 |     94% |72, 132, 173, 254, 265, 382->389 |
| fractal\_tasks\_core/tables/\_\_init\_\_.py                               |       16 |        0 |        4 |        0 |    100% |           |
| fractal\_tasks\_core/tables/v1.py                                         |      122 |       16 |       52 |        2 |     83% |112->115, 299, 310-327 |
| fractal\_tasks\_core/tasks/\_\_init\_\_.py                                |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/tasks/\_registration\_utils.py                       |       62 |        1 |       16 |        1 |     97% |       153 |
| fractal\_tasks\_core/tasks/\_utils.py                                     |       31 |       31 |        4 |        0 |      0% |     14-89 |
| fractal\_tasks\_core/tasks/\_zarr\_utils.py                               |       65 |        0 |       14 |        1 |     99% | 185->exit |
| fractal\_tasks\_core/tasks/apply\_registration\_to\_image.py              |      128 |       18 |       40 |        8 |     81% |104, 150-151, 153->175, 181->180, 185-197, 199->247, 218-230, 360-373, 398-400 |
| fractal\_tasks\_core/tasks/calculate\_registration\_image\_based.py       |       87 |        6 |       20 |        6 |     89% |68->exit, 190, 200, 218, 279, 312-314 |
| fractal\_tasks\_core/tasks/cellpose\_segmentation.py                      |      183 |       15 |       58 |       11 |     88% |156, 172->188, 182, 272-273, 304, 311-312, 318-320, 360, 368->373, 378, 452, 612-614 |
| fractal\_tasks\_core/tasks/cellpose\_utils.py                             |      137 |       12 |       46 |        7 |     90% |267, 285-291, 406-408, 422-424, 431-435 |
| fractal\_tasks\_core/tasks/cellvoyager\_to\_ome\_zarr\_compute.py         |       87 |        4 |       20 |        3 |     93% |118, 178, 247-249 |
| fractal\_tasks\_core/tasks/cellvoyager\_to\_ome\_zarr\_init.py            |      167 |       17 |       54 |       10 |     88% |128, 130, 186, 196-197, 212, 225-228, 283, 318, 335-336, 339, 485-487 |
| fractal\_tasks\_core/tasks/cellvoyager\_to\_ome\_zarr\_init\_multiplex.py |      206 |       21 |       54 |       15 |     86% |125, 132, 136, 145, 167, 169, 202, 204, 223-226, 255, 308, 314, 332, 346-347, 353, 532-534 |
| fractal\_tasks\_core/tasks/copy\_ome\_zarr\_hcs\_plate.py                 |       71 |        5 |       14 |        2 |     92% |48-49, 193, 227-229 |
| fractal\_tasks\_core/tasks/find\_registration\_consensus.py               |       47 |        4 |       16 |        4 |     87% |77->79, 107, 119, 168-170 |
| fractal\_tasks\_core/tasks/illumination\_correction.py                    |       94 |        7 |       26 |        5 |     90% |62, 83-87, 194, 210, 287-289 |
| fractal\_tasks\_core/tasks/image\_based\_registration\_hcs\_init.py       |       21 |        2 |       10 |        1 |     90% |     93-95 |
| fractal\_tasks\_core/tasks/import\_ome\_zarr.py                           |       93 |        8 |       30 |        6 |     89% |69, 107->154, 111-120, 155->158, 156->158, 268->290, 295-297 |
| fractal\_tasks\_core/tasks/init\_group\_by\_well\_for\_multiplexing.py    |       22 |        3 |       10 |        2 |     84% | 61, 86-88 |
| fractal\_tasks\_core/tasks/io\_models.py                                  |       80 |        0 |       14 |        0 |    100% |           |
| fractal\_tasks\_core/tasks/napari\_workflows\_wrapper.py                  |      235 |       19 |       98 |       13 |     90% |137-139, 274, 281, 287-292, 297, 328, 333, 373-377, 399, 500->487, 539-544, 551->553, 633-635 |
| fractal\_tasks\_core/tasks/projection.py                                  |       55 |        3 |       10 |        2 |     92% |46, 147-149 |
| fractal\_tasks\_core/tasks/projection\_utils.py                           |       29 |        0 |        4 |        0 |    100% |           |
| fractal\_tasks\_core/upscale\_array.py                                    |       73 |        7 |       48 |        6 |     88% |63, 91, 96->100, 113, 128, 192-197 |
| fractal\_tasks\_core/utils.py                                             |       88 |        3 |       36 |        6 |     93% |73, 141->149, 142->141, 178->181, 183, 230 |
| fractal\_tasks\_core/zarr\_utils.py                                       |       32 |        0 |        6 |        1 |     97% |    81->85 |
|                                                                 **TOTAL** | **3393** |  **242** | **1034** |  **137** | **91%** |           |


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