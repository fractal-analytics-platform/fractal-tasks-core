# Run Fractal tasks from the command line

This example just works for the test dataset at `10.5281_zenodo.7057076`. Download it first by installing `zenodo-get` in your python environment and running `fetch_test_data_from_zenodo_2x2.sh`.

To run a Fractal task from the command line, call `python Path/To/Task/File/Python.py -j Path/to/args.json --metadata-out Path/to/metadata_out.json`

1. The Python environment needs to have `fractal-tasks-core[fractal-tasks]` installed.
2. The path to the python file for each task needs to be known
3. The args.json file needs to be prepared (see below)
4. An output path for the output metadata needs to be provided

### 3. Creating the args.json files
The args.json files need to contain input and output paths (currently, relative paths are set). The input to the create_ome_zarr task is the folder with the raw images and the output is the folder where Zarr files are saved. For all other tasks, input and output are the Zarr folder.

In normal Fractal usage, the server handles getting the metadata output from one task and using it as input for the next task. Currently, the relevant metadata for the example at `10.5281_zenodo.7057076` is hard-coded in this example
To generalize, one would have to update the metadata after each task with the new entries that were produced (sometimes no new entries), e.g. set the plate, well & image entries for each metadata entry of the args.json files. Additionally, one needs to set the relevant component (one run with its custom args.json file needed for every OME-Zarr image that should be processed, e.g. one per well in the Fractal approach and the well is a component as in "myfile.ome.zarr/B/03/0/").

Commands to be run from the 07_use_task_from_command_line folder once everything is set (can be used directly for the test dataset at `10.5281_zenodo.7057076` once it's downloaded)

```
python ../../fractal_tasks_core/create_ome_zarr.py -j create_ome_zarr.args.json --metadata-out 0_metadata_out.json
python ../../fractal_tasks_core/yokogawa_to_ome_zarr.py  -j yokogawa_to_ome_zarr_B03_0.args.json --metadata-out 1_metadata_out.json
python ../../fractal_tasks_core/copy_ome_zarr.py -j copy_ome_zarr_structure.args.json --metadata-out 2_metadata_out.json
python ../../fractal_tasks_core/maximum_intensity_projection.py -j maximum_intensity_projection_B03_0.args.json --metadata-out 3_metadata_out.json
python ../../fractal_tasks_core/cellpose_segmentation.py -j cellpose.json --metadata-out 4_metadata_out.json

```

### Limitations
1. This example currently hard-codes the output path and can only be run if there is nothing in this output path
2. metadata_out.json files need to be unique. To rerun, delete the old ones or name the output differently.
3. The newly generated metadata is supposed to be added to the input for the next task. Currently, it's hard-coded for the example dataset. Fractal-server or the python example scripts in the other examples take care of that, while running the tasks from the command line don't make this as easy.

Tested with fractal-tasks-core 0.8.0 dev branch (parameters changed since 0.7.5)
