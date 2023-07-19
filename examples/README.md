This `examples` folder offers a few examples of how to run `fractal-tasks-core`
tasks as part of a Python script.

**NOTE**: this folder is not always kept up-to-date. If you encounter any
unexpected problem, please feel free to [open an issue on the
`fractal-tasks-core` GitHub
repository](https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/new/choose).

**NOTE**: Only examples from 01 to 04 are currently up-to-date.

General instructions:

1. Set up the correct environment via
```bash
pip install fractal-tasks-core[fractal-tasks]
```
(note this can be done e.g. from a venv or from a conda environment).

2. Download the example data from Zenodo, if necessary, via
```bash
pip install zenodo-get
./fetch_test_data_from_zenodo.sh
```

3. Browse to one of the examples (currently up-to-date are examples from 01 to
   04), remove the `tmp_out` temporary output folder (if present), and run one
   of the `run_workflow` Python scripts.

4. View the output OME-Zarr in the `tmp_out` folder with napari (which can be
   installed via `pip install napari[pyqt5] napari-ome-zarr`).
