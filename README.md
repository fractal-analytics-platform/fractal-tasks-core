# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                      |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| fractal\_tasks\_core/\_\_init\_\_.py                                      |        5 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/cellvoyager/\_\_init\_\_.py                          |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/cellvoyager/filenames.py                             |       61 |        1 |       26 |        1 |     98% |       167 |
| fractal\_tasks\_core/cellvoyager/metadata.py                              |      125 |        8 |       38 |        9 |     90% |113, 130, 310, 317, 329, 338, 379, 424->432, 475 |
| fractal\_tasks\_core/cellvoyager/wells.py                                 |       18 |        0 |        8 |        0 |    100% |           |
| fractal\_tasks\_core/channels.py                                          |      190 |        1 |       78 |        5 |     98% |30, 85->87, 86->85, 87->86, 124->123 |
| fractal\_tasks\_core/labels.py                                            |       39 |        0 |       14 |        0 |    100% |           |
| fractal\_tasks\_core/masked\_loading.py                                   |       62 |        8 |       16 |        5 |     83% |92, 108, 137-144, 158, 168 |
| fractal\_tasks\_core/ngff/\_\_init\_\_.py                                 |        6 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/ngff/specs.py                                        |      157 |        0 |       52 |       10 |     95% |122->121, 166->168, 167->166, 168->167, 200->199, 216->215, 223->222, 227->226, 234->233, 275->274 |
| fractal\_tasks\_core/ngff/zarr\_utils.py                                  |       61 |       10 |        6 |        0 |     85% |78-83, 98-104, 108-113 |
| fractal\_tasks\_core/pyramids.py                                          |       32 |        0 |       10 |        0 |    100% |           |
| fractal\_tasks\_core/roi/\_\_init\_\_.py                                  |        4 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/roi/\_overlaps\_common.py                            |       21 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/roi/load\_region.py                                  |       16 |        0 |       10 |        0 |    100% |           |
| fractal\_tasks\_core/roi/v1.py                                            |      170 |        2 |       34 |        2 |     98% |  164, 622 |
| fractal\_tasks\_core/roi/v1\_checks.py                                    |       36 |        0 |       18 |        0 |    100% |           |
| fractal\_tasks\_core/roi/v1\_overlaps.py                                  |      144 |        5 |       50 |        6 |     94% |72, 132, 173, 255, 266, 383->390 |
| fractal\_tasks\_core/tables/\_\_init\_\_.py                               |       16 |        0 |        4 |        0 |    100% |           |
| fractal\_tasks\_core/tables/v1.py                                         |      122 |        5 |       70 |       10 |     90% |32->34, 33->32, 34->33, 51->53, 52->51, 53->52, 112->115, 317->312, 319-326 |
| fractal\_tasks\_core/tasks/\_\_init\_\_.py                                |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/tasks/\_registration\_utils.py                       |       62 |        1 |       16 |        1 |     97% |       153 |
| fractal\_tasks\_core/tasks/\_utils.py                                     |       28 |        5 |        8 |        1 |     78% |33-35, 68-71 |
| fractal\_tasks\_core/tasks/\_zarr\_utils.py                               |       65 |        0 |       20 |        1 |     99% | 185->exit |
| fractal\_tasks\_core/tasks/apply\_registration\_to\_image.py              |      126 |       17 |       42 |        8 |     82% |51->50, 104, 150-151, 153->175, 181->180, 185-197, 199->236, 216-222, 349-362, 387-389 |
| fractal\_tasks\_core/tasks/calculate\_registration\_image\_based.py       |       84 |        6 |       22 |        7 |     88% |67->exit, 72->71, 181, 191, 209, 254, 287-289 |
| fractal\_tasks\_core/tasks/cellpose\_segmentation.py                      |      187 |       15 |       62 |       12 |     88% |157, 173->189, 183, 193->192, 273-274, 305, 312-313, 319-321, 361, 369->374, 379, 453, 622-624 |
| fractal\_tasks\_core/tasks/cellpose\_utils.py                             |      137 |       12 |       52 |       10 |     88% |74->73, 134->133, 260->259, 267, 285-291, 406-408, 422-424, 431-435 |
| fractal\_tasks\_core/tasks/cellvoyager\_to\_ome\_zarr\_compute.py         |       84 |        4 |       22 |        4 |     92% |57->56, 111, 164, 233-235 |
| fractal\_tasks\_core/tasks/cellvoyager\_to\_ome\_zarr\_init.py            |      167 |       17 |       56 |       11 |     87% |53->52, 132, 134, 190, 200-201, 216, 229-232, 287, 322, 339-340, 343, 489-491 |
| fractal\_tasks\_core/tasks/cellvoyager\_to\_ome\_zarr\_init\_multiplex.py |      206 |       21 |       56 |       16 |     86% |56->55, 129, 136, 140, 149, 171, 173, 206, 208, 227-230, 259, 312, 318, 336, 350-351, 357, 536-538 |
| fractal\_tasks\_core/tasks/copy\_ome\_zarr\_hcs\_plate.py                 |      106 |        3 |       28 |        3 |     96% |190->189, 232, 296-298 |
| fractal\_tasks\_core/tasks/find\_registration\_consensus.py               |       47 |        4 |       18 |        5 |     86% |42->41, 77->79, 107, 119, 168-170 |
| fractal\_tasks\_core/tasks/illumination\_correction.py                    |       94 |        7 |       28 |        6 |     89% |62, 83-87, 96->95, 194, 210, 287-289 |
| fractal\_tasks\_core/tasks/image\_based\_registration\_hcs\_init.py       |       21 |        2 |       12 |        2 |     88% |28->27, 93-95 |
| fractal\_tasks\_core/tasks/import\_ome\_zarr.py                           |       95 |        9 |       34 |        8 |     87% |69, 107->154, 111-120, 155->158, 156->158, 163->162, 213, 282->304, 309-311 |
| fractal\_tasks\_core/tasks/init\_group\_by\_well\_for\_multiplexing.py    |       22 |        3 |       12 |        3 |     82% |27->26, 61, 86-88 |
| fractal\_tasks\_core/tasks/io\_models.py                                  |       57 |        0 |       12 |        3 |     96% |126->125, 156->155, 169->168 |
| fractal\_tasks\_core/tasks/maximum\_intensity\_projection.py              |       63 |        6 |       10 |        3 |     88% |39->38, 151-164, 186-188 |
| fractal\_tasks\_core/tasks/napari\_workflows\_wrapper.py                  |      235 |       19 |      100 |       14 |     90% |63->62, 137-139, 274, 281, 287-292, 297, 328, 333, 373-377, 399, 500->487, 539-544, 551->553, 633-635 |
| fractal\_tasks\_core/upscale\_array.py                                    |       73 |        7 |       48 |        6 |     88% |63, 91, 96->100, 113, 128, 192-197 |
| fractal\_tasks\_core/utils.py                                             |       88 |        3 |       38 |        6 |     93% |73, 141->149, 142->141, 178->181, 183, 230 |
| fractal\_tasks\_core/zarr\_utils.py                                       |       32 |        0 |        6 |        1 |     97% |    81->85 |
|                                                                 **TOTAL** | **3364** |  **201** | **1136** |  **179** | **91%** |           |


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