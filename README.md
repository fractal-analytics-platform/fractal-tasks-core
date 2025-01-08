# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                      |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| .venv/lib/python3.10/site-packages/cellpose/dynamics.py                   |      366 |      338 |      142 |       12 |      6% |5-25, 28, 52->53, 52->58, 62-408, 412, 440->442, 440->450, 442->440, 442->443, 453, 481->482, 481->490, 482->481, 482->484, 488->482, 488->489, 492-784 |
| .venv/lib/python3.10/site-packages/numba/cpython/old\_builtins.py         |      644 |      632 |      194 |        4 |      1% |1-953, 956-997, 1000-1005, 1008->1009, 1013-1018, 1021->1022, 1024-1025 |
| .venv/lib/python3.10/site-packages/numba/cpython/unicode.py               |     1723 |     1602 |      724 |       56 |      5% |1-278, 282, 291, 292->293, 292->294, 294->295, 294->296, 296->297, 296->301, 306-333, 337, 344->345, 344->346, 346->347, 346->348, 348->349, 348->351, 355-370, 372, 373->374, 373->376, 376->377, 376->378, 378->379, 378->383, 379->380, 379->382, 383->384, 383->386, 389, 393, 394->395, 394->396, 399-410, 412, 413->414, 413->415, 415->416, 415->417, 417->418, 417->419, 419->420, 419->422, 424-478, 481-1487, 1491, 1493->1494, 1493->1497, 1501->1502, 1501->1506, 1512->1513, 1512->1519, 1522-1526, 1529-1764, 1768, 1769->1770, 1769->1777, 1777->1778, 1777->exit, 1782-1884, 1890->1891, 1890->1892, 1892->1893, 1892->1894, 1895-2509, 2512->2514, 2512->2515, 2516-2556, 2560-2561, 2567-2570, 2574->2575, 2574->2577, 2577->2578, 2577->2579, 2583->2584, 2583->2585, 2586->2587, 2586->2591, 2592-2596, 2600-2675 |
| .venv/lib/python3.10/site-packages/numba/np/arrayobj.py                   |     4036 |     4030 |     1068 |        0 |      1% |6-1660, 1667-4287, 4293, 4302-6995 |
| .venv/lib/python3.10/site-packages/numba/np/npyimpl.py                    |      434 |      418 |      124 |       10 |      3% |6-334, 352->355, 352->357, 359->360, 359->375, 364->368, 364->370, 368->369, 368->373, 370->372, 370->373, 377-878 |
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
| fractal\_tasks\_core/ngff/zarr\_utils.py                                  |       61 |       10 |        6 |        0 |     85% |78-83, 98-104, 108-113 |
| fractal\_tasks\_core/pyramids.py                                          |       32 |        0 |       10 |        0 |    100% |           |
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
| fractal\_tasks\_core/tasks/\_utils.py                                     |       28 |        5 |        4 |        1 |     75% |33-35, 68-71 |
| fractal\_tasks\_core/tasks/\_zarr\_utils.py                               |       65 |        0 |       14 |        1 |     99% | 185->exit |
| fractal\_tasks\_core/tasks/apply\_registration\_to\_image.py              |      126 |       17 |       40 |        7 |     82% |104, 150-151, 153->175, 181->180, 185-197, 199->236, 216-222, 349-362, 387-389 |
| fractal\_tasks\_core/tasks/calculate\_registration\_image\_based.py       |       87 |        6 |       20 |        6 |     89% |68->exit, 190, 200, 218, 279, 312-314 |
| fractal\_tasks\_core/tasks/cellpose\_segmentation.py                      |      183 |       15 |       58 |       11 |     88% |156, 172->188, 182, 272-273, 304, 311-312, 318-320, 360, 368->373, 378, 452, 612-614 |
| fractal\_tasks\_core/tasks/cellpose\_utils.py                             |      137 |       12 |       46 |        7 |     90% |267, 285-291, 406-408, 422-424, 431-435 |
| fractal\_tasks\_core/tasks/cellvoyager\_to\_ome\_zarr\_compute.py         |       84 |        4 |       20 |        3 |     93% |111, 164, 233-235 |
| fractal\_tasks\_core/tasks/cellvoyager\_to\_ome\_zarr\_init.py            |      167 |       17 |       54 |       10 |     88% |132, 134, 190, 200-201, 216, 229-232, 287, 322, 339-340, 343, 489-491 |
| fractal\_tasks\_core/tasks/cellvoyager\_to\_ome\_zarr\_init\_multiplex.py |      206 |       21 |       54 |       15 |     86% |129, 136, 140, 149, 171, 173, 206, 208, 227-230, 259, 312, 318, 336, 350-351, 357, 536-538 |
| fractal\_tasks\_core/tasks/copy\_ome\_zarr\_hcs\_plate.py                 |      105 |        2 |       24 |        1 |     98% |   297-299 |
| fractal\_tasks\_core/tasks/find\_registration\_consensus.py               |       47 |        4 |       16 |        4 |     87% |77->79, 107, 119, 168-170 |
| fractal\_tasks\_core/tasks/illumination\_correction.py                    |       94 |        7 |       26 |        5 |     90% |62, 83-87, 194, 210, 287-289 |
| fractal\_tasks\_core/tasks/image\_based\_registration\_hcs\_init.py       |       21 |        2 |       10 |        1 |     90% |     93-95 |
| fractal\_tasks\_core/tasks/import\_ome\_zarr.py                           |       95 |        9 |       32 |        7 |     87% |69, 107->154, 111-120, 155->158, 156->158, 213, 282->304, 309-311 |
| fractal\_tasks\_core/tasks/init\_group\_by\_well\_for\_multiplexing.py    |       22 |        3 |       10 |        2 |     84% | 61, 86-88 |
| fractal\_tasks\_core/tasks/io\_models.py                                  |       60 |        0 |        6 |        0 |    100% |           |
| fractal\_tasks\_core/tasks/napari\_workflows\_wrapper.py                  |      235 |       19 |       98 |       13 |     90% |137-139, 274, 281, 287-292, 297, 328, 333, 373-377, 399, 500->487, 539-544, 551->553, 633-635 |
| fractal\_tasks\_core/tasks/projection.py                                  |       56 |        3 |        8 |        2 |     92% |76, 142-144 |
| fractal\_tasks\_core/tasks/projection\_utils.py                           |       29 |        0 |        4 |        0 |    100% |           |
| fractal\_tasks\_core/upscale\_array.py                                    |       73 |        7 |       48 |        6 |     88% |63, 91, 96->100, 113, 128, 192-197 |
| fractal\_tasks\_core/utils.py                                             |       88 |        3 |       36 |        6 |     93% |73, 141->149, 142->141, 178->181, 183, 230 |
| fractal\_tasks\_core/zarr\_utils.py                                       |       32 |        0 |        6 |        1 |     97% |    81->85 |
|                                                                 **TOTAL** | **10594** | **7228** | **3284** |  **219** | **31%** |           |


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