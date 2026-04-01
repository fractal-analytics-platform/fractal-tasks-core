# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/fractal-analytics-platform/fractal-tasks-core/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                        |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| fractal\_tasks\_core/\_\_init\_\_.py                        |        5 |        2 |        0 |        0 |     60% |       8-9 |
| fractal\_tasks\_core/\_illumination\_correction\_utils.py   |       23 |        0 |        2 |        0 |    100% |           |
| fractal\_tasks\_core/\_measure\_features\_utils.py          |       93 |        1 |       30 |        2 |     98% |228, 258-\>261 |
| fractal\_tasks\_core/\_projection\_utils.py                 |       80 |        1 |       12 |        2 |     97% |176, 252-\>254 |
| fractal\_tasks\_core/\_registration\_utils.py               |       29 |        0 |        8 |        1 |     97% | 72-\>exit |
| fractal\_tasks\_core/\_threshold\_segmentation\_utils.py    |       57 |        1 |        2 |        1 |     97% |       206 |
| fractal\_tasks\_core/\_utils.py                             |       56 |        0 |       14 |        0 |    100% |           |
| fractal\_tasks\_core/apply\_registration\_to\_image.py      |      106 |        9 |       38 |        6 |     90% |166-167, 200, 237-\>248, 250-\>278, 262-269, 309-311 |
| fractal\_tasks\_core/compute\_image\_based\_registration.py |       47 |        3 |       14 |        2 |     92% |113, 160-162 |
| fractal\_tasks\_core/compute\_projection\_hcs.py            |       11 |        2 |        2 |        1 |     77% |     45-47 |
| fractal\_tasks\_core/compute\_registration\_consensus.py    |      110 |        6 |       38 |        6 |     92% |49-\>exit, 72, 167, 210, 242, 268-270 |
| fractal\_tasks\_core/illumination\_correction.py            |      134 |        4 |       64 |        3 |     96% |238, 290, 340-342 |
| fractal\_tasks\_core/import\_ome\_zarr.py                   |      107 |        4 |       26 |        3 |     95% |39, 335, 347-349 |
| fractal\_tasks\_core/init\_image\_based\_registration.py    |       30 |        3 |       12 |        2 |     88% | 72, 95-97 |
| fractal\_tasks\_core/init\_projection\_hcs.py               |       68 |        5 |       16 |        3 |     90% |24-25, 138-\>140, 166, 200-202 |
| fractal\_tasks\_core/init\_registration\_consensus.py       |       27 |        3 |        8 |        2 |     86% | 60, 81-83 |
| fractal\_tasks\_core/measure\_features.py                   |       35 |        3 |        8 |        1 |     91% |29, 121-123 |
| fractal\_tasks\_core/projection.py                          |       19 |        2 |        6 |        2 |     84% |54-\>56, 66-68 |
| fractal\_tasks\_core/threshold\_segmentation.py             |       39 |        2 |        6 |        1 |     93% |   160-162 |
| **TOTAL**                                                   | **1076** |   **51** |  **306** |   **38** | **94%** |           |


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