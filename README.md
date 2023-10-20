# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| fractal\_tasks\_core/\_\_init\_\_.py                                |        4 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_ROI\_overlaps.py                          |      159 |        5 |       62 |        6 |     95% |139, 199, 240, 322, 333, 490->497 |
| fractal\_tasks\_core/lib\_channels.py                               |      176 |        1 |       92 |        3 |     99% |28, 85->84, 432->434 |
| fractal\_tasks\_core/lib\_glob.py                                   |       20 |        0 |        8 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_input\_models.py                          |       49 |        0 |       22 |        5 |     93% |34->33, 67->66, 79->78, 104->103, 116->115 |
| fractal\_tasks\_core/lib\_masked\_loading.py                        |       62 |        9 |       16 |        6 |     81% |90, 101, 107, 136-143, 157, 167 |
| fractal\_tasks\_core/lib\_metadata\_parsing.py                      |       99 |        7 |       30 |        8 |     88% |94, 111, 225, 232, 234, 274, 319->327, 370 |
| fractal\_tasks\_core/lib\_ngff.py                                   |      152 |        3 |       54 |        8 |     95% |111->110, 155->154, 183->182, 199->198, 206->205, 210->209, 217->216, 258->257, 417-422 |
| fractal\_tasks\_core/lib\_parse\_filename\_metadata.py              |       33 |        1 |       12 |        1 |     96% |       104 |
| fractal\_tasks\_core/lib\_pyramid\_creation.py                      |       33 |        0 |       10 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_read\_fractal\_metadata.py                |       35 |        0 |       14 |        0 |    100% |           |
| fractal\_tasks\_core/lib\_regions\_of\_interest.py                  |      202 |        1 |       66 |        1 |     99% |       168 |
| fractal\_tasks\_core/lib\_upscale\_array.py                         |       74 |        7 |       48 |        6 |     88% |63, 91, 96->100, 113, 128, 192-197 |
| fractal\_tasks\_core/lib\_write.py                                  |       97 |        0 |       42 |        2 |     99% |83->87, 171->174 |
| fractal\_tasks\_core/lib\_zattrs\_utils.py                          |       38 |        1 |       14 |        1 |     96% |        69 |
| fractal\_tasks\_core/tasks/\_\_init\_\_.py                          |        0 |        0 |        0 |        0 |    100% |           |
| fractal\_tasks\_core/tasks/\_utils.py                               |       29 |        5 |        8 |        1 |     78% |33-35, 68-71 |
| fractal\_tasks\_core/tasks/apply\_registration\_to\_ROI\_tables.py  |       84 |        5 |       28 |        6 |     90% |37->36, 81->83, 119, 130, 245, 286-288 |
| fractal\_tasks\_core/tasks/apply\_registration\_to\_image.py        |      116 |       17 |       38 |        9 |     79% |46->45, 104, 153-154, 156->180, 186->185, 190-202, 204->229, 218-220, 240, 337-350, 375-377 |
| fractal\_tasks\_core/tasks/calculate\_registration\_image\_based.py |       94 |        5 |       20 |        5 |     91% |45->44, 158, 177, 222, 347-349 |
| fractal\_tasks\_core/tasks/cellpose\_segmentation.py                |      208 |       23 |       68 |       12 |     86% |103-144, 148->147, 255, 263-266, 309-315, 323-324, 362, 370->375, 379, 451, 475->479, 556->572, 566, 658-660 |
| fractal\_tasks\_core/tasks/copy\_ome\_zarr.py                       |       72 |        4 |       32 |        7 |     89% |39->38, 102, 105, 171->159, 175->184, 196->201, 215-217 |
| fractal\_tasks\_core/tasks/create\_ome\_zarr.py                     |      162 |       25 |       72 |       11 |     83% |46->45, 118, 120, 173, 180-190, 198-199, 212-215, 269, 300, 316-317, 320, 464-466 |
| fractal\_tasks\_core/tasks/create\_ome\_zarr\_multiplex.py          |      189 |       19 |       80 |       14 |     88% |48->47, 123, 130, 134, 143, 161, 192, 213-216, 243, 294, 300, 317, 330-331, 337, 509-511 |
| fractal\_tasks\_core/tasks/illumination\_correction.py              |      105 |       14 |       34 |        9 |     82% |62, 83-87, 96->95, 151, 172-175, 221, 236, 249-250, 305-307 |
| fractal\_tasks\_core/tasks/import\_ome\_zarr.py                     |       93 |       10 |       36 |        9 |     85% |70, 72, 84->95, 95->110, 110->exit, 114-123, 156->155, 204, 253->270, 279-281 |
| fractal\_tasks\_core/tasks/maximum\_intensity\_projection.py        |       48 |        3 |        8 |        3 |     89% |32->31, 67, 129-131 |
| fractal\_tasks\_core/tasks/napari\_workflows\_wrapper.py            |      243 |       20 |      118 |       15 |     90% |61->60, 149-151, 188, 293, 300, 306-311, 316, 347, 352, 392-396, 419, 522->509, 561-566, 573->575, 653-655 |
| fractal\_tasks\_core/tasks/yokogawa\_to\_ome\_zarr.py               |       94 |        5 |       22 |        5 |     91% |62->61, 111, 168, 219, 269-271 |
|                                                           **TOTAL** | **2770** |  **190** | **1054** |  **153** | **91%** |           |


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