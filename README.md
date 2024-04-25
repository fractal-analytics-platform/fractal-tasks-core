# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                      |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| fractal\_tasks\_core/\_\_init\_\_.py                                      |        5 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/cellvoyager/\_\_init\_\_.py                          |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/cellvoyager/filenames.py                             |       52 |        1 |       20 |        1 |     97% |       148 |
| fractal\_tasks\_core/cellvoyager/metadata.py                              |       99 |        7 |       26 |        8 |     88% |94, 111, 225, 232, 234, 274, 319->327, 370 |
| fractal\_tasks\_core/channels.py                                          |      188 |        1 |       74 |        3 |     98% |28, 84->83, 119->118 |
| fractal\_tasks\_core/labels.py                                            |       40 |        0 |       14 |        0 |    100% |           |
| fractal\_tasks\_core/masked\_loading.py                                   |       63 |        8 |       16 |        5 |     84% |92, 108, 137-144, 158, 168 |
| fractal\_tasks\_core/ngff/\_\_init\_\_.py                                 |        6 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/ngff/specs.py                                        |      145 |        0 |       44 |        8 |     96% |111->110, 155->154, 183->182, 199->198, 206->205, 210->209, 217->216, 258->257 |
| fractal\_tasks\_core/ngff/zarr\_utils.py                                  |       61 |       10 |        6 |        0 |     85% |78-83, 98-104, 108-113 |
| fractal\_tasks\_core/pyramids.py                                          |       33 |        0 |       10 |        0 |    100% |           |
| fractal\_tasks\_core/roi/\_\_init\_\_.py                                  |        4 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/roi/\_overlaps\_common.py                            |       22 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/roi/load\_region.py                                  |       16 |        0 |       10 |        0 |    100% |           |
| fractal\_tasks\_core/roi/v1.py                                            |      156 |        1 |       30 |        1 |     99% |       164 |
| fractal\_tasks\_core/roi/v1\_checks.py                                    |       37 |        0 |       18 |        0 |    100% |           |
| fractal\_tasks\_core/roi/v1\_overlaps.py                                  |      145 |        5 |       50 |        6 |     94% |72, 132, 173, 255, 266, 383->390 |
| fractal\_tasks\_core/tables/\_\_init\_\_.py                               |       17 |        0 |        4 |        0 |    100% |           |
| fractal\_tasks\_core/tables/v1.py                                         |      120 |        6 |       62 |        7 |     91% |33->32, 51->50, 110->113, 290, 315->310, 317-324 |
| fractal\_tasks\_core/tasks/\_\_init\_\_.py                                |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/tasks/\_registration\_utils.py                       |       69 |        2 |       24 |        2 |     96% |   53, 200 |
| fractal\_tasks\_core/tasks/\_utils.py                                     |       29 |        5 |        8 |        1 |     78% |33-35, 68-71 |
| fractal\_tasks\_core/tasks/\_zarr\_utils.py                               |       69 |        0 |       20 |        1 |     99% | 195->exit |
| fractal\_tasks\_core/tasks/apply\_registration\_to\_image.py              |      127 |       17 |       42 |        8 |     82% |51->50, 104, 150-151, 153->175, 181->180, 185-197, 199->236, 216-222, 349-362, 387-389 |
| fractal\_tasks\_core/tasks/calculate\_registration\_image\_based.py       |       71 |        6 |       14 |        6 |     86% |48->47, 142, 152, 170, 216, 261-263 |
| fractal\_tasks\_core/tasks/cellpose\_segmentation.py                      |      202 |       23 |       62 |       10 |     85% |131-191, 195->194, 321-322, 363-369, 377-378, 416, 424->429, 433, 505, 529->533, 618->634, 628, 717-719 |
| fractal\_tasks\_core/tasks/cellpose\_transforms.py                        |       78 |        8 |       36 |        8 |     86% |68->67, 128->127, 181-183, 197-199, 206-210 |
| fractal\_tasks\_core/tasks/cellvoyager\_to\_ome\_zarr\_compute.py         |       79 |        4 |       20 |        4 |     92% |57->56, 111, 158, 222-224 |
| fractal\_tasks\_core/tasks/cellvoyager\_to\_ome\_zarr\_init.py            |      163 |       25 |       54 |       11 |     82% |51->50, 124, 126, 177, 184-194, 202-203, 216-219, 273, 304, 320-321, 324, 470-472 |
| fractal\_tasks\_core/tasks/cellvoyager\_to\_ome\_zarr\_init\_multiplex.py |      197 |       20 |       54 |       15 |     86% |54->53, 122, 129, 133, 142, 157, 188, 190, 209-212, 242, 293, 299, 316, 329-330, 336, 519-521 |
| fractal\_tasks\_core/tasks/copy\_ome\_zarr\_hcs\_plate.py                 |      108 |        3 |       28 |        3 |     96% |191->190, 233, 297-299 |
| fractal\_tasks\_core/tasks/find\_registration\_consensus.py               |       48 |        4 |       18 |        5 |     86% |42->41, 77->79, 107, 119, 167-169 |
| fractal\_tasks\_core/tasks/illumination\_correction.py                    |       95 |        7 |       28 |        6 |     89% |62, 83-87, 96->95, 195, 211, 288-290 |
| fractal\_tasks\_core/tasks/image\_based\_registration\_hcs\_init.py       |       22 |        2 |       12 |        2 |     88% |28->27, 93-95 |
| fractal\_tasks\_core/tasks/import\_ome\_zarr.py                           |       97 |       10 |       36 |       11 |     84% |69, 71, 83->94, 94->109, 109->156, 113-122, 157->160, 158->160, 165->164, 215, 283->305, 311-313 |
| fractal\_tasks\_core/tasks/init\_group\_by\_well\_for\_multiplexing.py    |       23 |        3 |       12 |        3 |     83% |27->26, 61, 86-88 |
| fractal\_tasks\_core/tasks/io\_models.py                                  |       52 |        0 |       12 |        3 |     95% |123->122, 150->149, 162->161 |
| fractal\_tasks\_core/tasks/maximum\_intensity\_projection.py              |       64 |        6 |       10 |        3 |     88% |39->38, 151-164, 187-189 |
| fractal\_tasks\_core/tasks/napari\_workflows\_wrapper.py                  |      236 |       19 |      100 |       14 |     90% |63->62, 137-139, 274, 281, 287-292, 297, 328, 333, 373-377, 400, 500->487, 539-544, 551->553, 633-635 |
| fractal\_tasks\_core/upscale\_array.py                                    |       74 |        7 |       48 |        6 |     88% |63, 91, 96->100, 113, 128, 192-197 |
| fractal\_tasks\_core/utils.py                                             |       68 |        2 |       28 |        5 |     93% |71, 139->147, 140->139, 176->179, 181 |
| fractal\_tasks\_core/zarr\_utils.py                                       |       33 |        0 |        6 |        1 |     97% |    81->85 |
|                                                                 **TOTAL** | **3213** |  **212** | **1056** |  **167** | **91%** |           |


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