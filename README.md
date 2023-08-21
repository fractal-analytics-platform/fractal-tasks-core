# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| fractal\_tasks\_core/\_\_init\_\_.py                         |        4 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_ROI\_overlaps.py                   |      159 |        5 |       62 |        6 |     95% |139, 199, 240, 322, 333, 490->497 |
| fractal\_tasks\_core/lib\_channels.py                        |       95 |        1 |       58 |        1 |     99% |        25 |
| fractal\_tasks\_core/lib\_glob.py                            |       20 |        0 |        8 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_input\_models.py                   |       49 |        0 |       22 |        5 |     93% |34->33, 67->66, 79->78, 104->103, 116->115 |
| fractal\_tasks\_core/lib\_masked\_loading.py                 |       62 |        9 |       16 |        6 |     81% |90, 101, 107, 136-143, 157, 167 |
| fractal\_tasks\_core/lib\_metadata\_parsing.py               |       99 |        7 |       30 |        8 |     88% |94, 111, 225, 232, 234, 274, 319->327, 370 |
| fractal\_tasks\_core/lib\_parse\_filename\_metadata.py       |       33 |        1 |       12 |        1 |     96% |       104 |
| fractal\_tasks\_core/lib\_pyramid\_creation.py               |       28 |        0 |       10 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_read\_fractal\_metadata.py         |       35 |        0 |       14 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_regions\_of\_interest.py           |      136 |        3 |       42 |        3 |     97% |154, 357, 359 |
| fractal\_tasks\_core/lib\_upscale\_array.py                  |       74 |        7 |       48 |        6 |     88% |63, 91, 96->100, 113, 128, 192-197 |
| fractal\_tasks\_core/lib\_zattrs\_utils.py                   |       50 |        2 |       28 |        2 |     95% |71->70, 88, 138 |
| fractal\_tasks\_core/tasks/\_\_init\_\_.py                   |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/tasks/\_utils.py                        |       29 |        5 |        8 |        1 |     78% |33-35, 68-71 |
| fractal\_tasks\_core/tasks/cellpose\_segmentation.py         |      238 |       28 |       80 |       17 |     85% |101-142, 146->145, 256, 264-267, 298-304, 312-313, 368, 376->392, 384, 399, 404, 411, 434, 485, 509->513, 590->608, 602, 678, 699-701 |
| fractal\_tasks\_core/tasks/copy\_ome\_zarr.py                |       70 |        4 |       32 |        7 |     89% |38->37, 99, 102, 162->150, 166->176, 188->193, 207-209 |
| fractal\_tasks\_core/tasks/create\_ome\_zarr.py              |      164 |       25 |       72 |       11 |     83% |46->45, 116, 118, 171, 178-188, 196-197, 210-213, 263, 294, 310-311, 314, 450-452 |
| fractal\_tasks\_core/tasks/create\_ome\_zarr\_multiplex.py   |      190 |       19 |       80 |       14 |     88% |46->45, 119, 126, 130, 139, 157, 188, 209-212, 239, 287, 293, 310, 323-324, 330, 496-498 |
| fractal\_tasks\_core/tasks/illumination\_correction.py       |       99 |       17 |       34 |       11 |     77% |61, 82-86, 95->94, 154, 158, 161-166, 179-182, 221, 236, 249-250, 305-307 |
| fractal\_tasks\_core/tasks/maximum\_intensity\_projection.py |       51 |        5 |       16 |        5 |     85% |34->33, 75, 109, 128, 151-153 |
| fractal\_tasks\_core/tasks/napari\_workflows\_wrapper.py     |      264 |       24 |      130 |       19 |     89% |60->59, 148-150, 187, 201, 206, 304, 311, 317-322, 327, 358, 363, 403-407, 436, 450, 531->518, 570-575, 582->584, 639, 664-666 |
| fractal\_tasks\_core/tasks/yokogawa\_to\_ome\_zarr.py        |       80 |        5 |       22 |        5 |     90% |59->58, 106, 153, 195, 245-247 |
|                                                    **TOTAL** | **2029** |  **167** |  **824** |  **128** | **89%** |           |


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