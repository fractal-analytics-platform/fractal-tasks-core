rm -r data/*.zarr
cp -r Backup_data/one_well_9x8_fovs_10_z_planes_3_channels.zarr/ data/plate.zarr
mprof run run_example_v8_12G_overwrite_pyramid.py
