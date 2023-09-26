# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| fractal\_tasks\_core/\_\_init\_\_.py                                |        4 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_ROI\_overlaps.py                          |      159 |        5 |       62 |        6 |     95% |139, 199, 240, 322, 333, 490->497 |
| fractal\_tasks\_core/lib\_channels.py                               |      107 |        1 |       68 |        2 |     98% |26, 83->82 |
| fractal\_tasks\_core/lib\_glob.py                                   |       20 |        0 |        8 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_input\_models.py                          |       49 |        0 |       22 |        5 |     93% |34->33, 67->66, 79->78, 104->103, 116->115 |
| fractal\_tasks\_core/lib\_masked\_loading.py                        |       62 |        9 |       16 |        6 |     81% |90, 101, 107, 136-143, 157, 167 |
| fractal\_tasks\_core/lib\_metadata\_parsing.py                      |       99 |        7 |       30 |        8 |     88% |94, 111, 225, 232, 234, 274, 319->327, 370 |
| fractal\_tasks\_core/lib\_parse\_filename\_metadata.py              |       33 |        1 |       12 |        1 |     96% |       104 |
| fractal\_tasks\_core/lib\_pyramid\_creation.py                      |       28 |        0 |       10 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_read\_fractal\_metadata.py                |       35 |        0 |       14 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_regions\_of\_interest.py                  |      163 |        1 |       54 |        1 |     99% |       167 |
| fractal\_tasks\_core/lib\_upscale\_array.py                         |       74 |        7 |       48 |        6 |     88% |63, 91, 96->100, 113, 128, 192-197 |
| fractal\_tasks\_core/lib\_write.py                                  |       97 |        0 |       42 |        2 |     99% |83->87, 171->174 |
| fractal\_tasks\_core/lib\_zattrs\_utils.py                          |       79 |        2 |       40 |        2 |     97% |72->71, 89, 139 |
| fractal\_tasks\_core/tasks/\_\_init\_\_.py                          |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/tasks/\_utils.py                               |       29 |        5 |        8 |        1 |     78% |33-35, 68-71 |
| fractal\_tasks\_core/tasks/apply\_registration\_to\_ROI\_tables.py  |       86 |        5 |       30 |        6 |     91% |38->37, 82->84, 122, 133, 248, 289-291 |
| fractal\_tasks\_core/tasks/apply\_registration\_to\_image.py        |      115 |       17 |       40 |        9 |     79% |48->47, 111, 157-158, 160->184, 190->189, 194-206, 208->233, 222-224, 244, 341-354, 379-381 |
| fractal\_tasks\_core/tasks/calculate\_registration\_image\_based.py |       92 |        5 |       20 |        5 |     91% |45->44, 160, 180, 225, 350-352 |
| fractal\_tasks\_core/tasks/cellpose\_segmentation.py                |      215 |       26 |       72 |       15 |     84% |103-144, 148->147, 260, 268-271, 302-308, 316-317, 365, 373->389, 381, 396, 401, 408, 480, 504->508, 585->603, 597, 685-687 |
| fractal\_tasks\_core/tasks/copy\_ome\_zarr.py                       |       71 |        4 |       32 |        7 |     89% |39->38, 102, 105, 171->159, 175->185, 197->202, 216-218 |
| fractal\_tasks\_core/tasks/create\_ome\_zarr.py                     |      162 |       25 |       72 |       11 |     83% |46->45, 118, 120, 173, 180-190, 198-199, 212-215, 269, 300, 316-317, 320, 464-466 |
| fractal\_tasks\_core/tasks/create\_ome\_zarr\_multiplex.py          |      189 |       19 |       80 |       14 |     88% |48->47, 123, 130, 134, 143, 161, 192, 213-216, 243, 294, 300, 317, 330-331, 337, 509-511 |
| fractal\_tasks\_core/tasks/illumination\_correction.py              |      101 |       14 |       34 |        9 |     81% |62, 83-87, 96->95, 156, 181-184, 224, 239, 252-253, 308-310 |
| fractal\_tasks\_core/tasks/maximum\_intensity\_projection.py        |       60 |        5 |       16 |        5 |     87% |37->36, 81, 116, 135, 167-169 |
| fractal\_tasks\_core/tasks/napari\_workflows\_wrapper.py            |      251 |       22 |      122 |       17 |     90% |62->61, 155-157, 194, 208, 213, 312, 319, 325-330, 335, 366, 371, 412-416, 439, 540->527, 579-584, 591->593, 671-673 |
| fractal\_tasks\_core/tasks/yokogawa\_to\_ome\_zarr.py               |       90 |        5 |       22 |        5 |     91% |62->61, 111, 160, 211, 261-263 |
|                                                           **TOTAL** | **2470** |  **185** |  **974** |  **143** | **90%** |           |


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