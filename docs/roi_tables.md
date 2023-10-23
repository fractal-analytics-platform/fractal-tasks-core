# ROI tables

## Scope

In `fractal-tasks-core`, regions of interest (ROIs) are three-dimensional
regions of space delimited by orthogonal planes. ROI tables are stored as
AnnData tables, within OME-NGFF Zarr images.

We have several use cases for tables:

1. We keep track of the positions of the Field of Views (FOVs) within a well, after stitching the corresponding FOV images into a single whole-well array.
2. We keep track of the original state before some transformations are applied - e.g. shifting FOVs to avoid overlaps, or shifting a multiplexing cycle during registration.
3. Several tasks in `fractal-tasks-core` take an existing ROI table as an input and then loop over the ROIs defined in the table. Such tasks have more flexibility, as they can process e.g. a whole well, a set of FOVs, or a set of custom regions of the array.
4. We store ROIs associated to segmented objects, for instance the bounding boxes of organoids/nuclei.
5. We store measurements associated to segmented objects, e.g. as computed via `regionprops` from `scikit-image` (as wrapped in [napari-skimage-regionprops](https://github.com/haesleinhuepf/napari-skimage-regionprops)).


## Table specifications

The current section describes the first version (V1) of `fractal-tasks-core`
tables, which is based on [a proposed update to NGFF
specs](https://github.com/ome/ngff/pull/64); this update is currently on hold,
and `fractal-tasks-core` will evolve as soon as the NGFF specs will adopt a
definition of tables.
As in the original proposed NGFF update, the current specifications are
specifically based on AnnData tables.

### Zarr structure

The structure of Zarr groups is based on the [`image` specification in NGFF 0.4](https://ngff.openmicroscopy.org/0.4/index.html#image-layout), with an additional `tables` group and the corresponding subgroups (similar to `labels`):
```
image.zarr        # Zarr group for a NGFF image
|
├── 0             # Zarr array for multiscale level 0
├── ...
├── N             # Zarr array for multiscale level N
|
├── labels        # Zarr subgroup with a list of labels associated to this image
|   ├── label_A   # Zarr subgroup for a given label
|   ├── label_B   # Zarr subgroup for a given label
|   └── ...
|
├── tables        # Zarr subgroup with a list of tables associated to this image
|   ├── table_1   # Zarr subgroup for a given table
|   ├── table_2   # Zarr subgroup for a given table
|   └── ...
|

```

### Zarr attributes

#### Tables container

The Zarr attributes of the `tables` group must include the key `tables`,
pointing to the list of all tables; this simplifies the discovery of image
tables.

Here is an example of `image.zarr/tables/.zattrs`:
```json
{
    "tables": ["table_1", "table_2"]
}
```

#### Single table (default)

For each table, the Zarr attributes must include the key
`fractal_roi_table_version`, pointing to the string version of this
specification (e.g. `1`).

Here is an example of `image.zarr/tables/table1/.zattrs`
```json
{
    "fractal_roi_table_version": "1",
    "encoding-type": "anndata",      # Automatically added by anndata
    "encoding-version": "0.1.0",     # Automatically added by anndata
}
```

This is the kind of tables that are used in `fractal-tasks-core` to store ROIs
like a whole well, or the list of field of views.

#### Single table (segmented objects)

When each table row corresponds to (the bounding box of) a segmented object,
`fractal-tasks-core` follows more closely the [proposed NGFF update mentioned
above](https://github.com/ome/ngff/pull/64), with the following additional
requirements on the Zarr group of a given table:

* Attributes must contain a `type` key, with value `ngff:region_table`.
* Attributes must contain a `region` key; the corresponding value must be an
  object with a `path` key and a string value (i.e. the path to the data the
  table is annotating).
* Attributes may include a key `instance_key`, which is the key in `obs` that
  denotes which instance in `region` the row corresponds to. If `instance_key`
  is not provided, the values from the `_index` Zarr attribute of `obs` is used.

Here is an example of `image.zarr/tables/table1/.zattrs`
```json
{
    "fractal_roi_table_version": "1",
    "type": "ngff:region_table",
    "region": {
        "path": "../labels/label_DAPI",
    },
    "instance_key": "label",
    "encoding-type": "anndata",      # Automatically added by anndata
    "encoding-version": "0.1.0",     # Automatically added by anndata
}
```

### AnnData table format

Data of a table are stored into a Zarr group as AnnData ("Annotated Data")
objects; the [`anndata` Python library](anndata.readthedocs.io) provides the
definition of this format and the relevant tools.

Quoting from `anndata` documentation:

> AnnData is specifically designed for matrix-like data. By this we mean that
> we have $n$ observations, each of which can be represented as $d$-dimensional
> vectors, where each dimension corresponds to a variable or feature. Both the
> rows and columns of this $n \times d$ matrix are special in the sense that
> they are indexed.
>
> (https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html)

Note that AnnData tables are easily transformed from/into `pandas.DataFrame`
objects - see e.g. the [`AnnData.to_df`
method](https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.to_df.html#anndata.AnnData.to_df).

### Table columns (WIP)

TODO: list here all required/optional columns and their meaning.

## Default tables (WIP)

When parsing Yokogawa images into OME-Zarr (via the
[`create_ome_zarr`](../reference/fractal_tasks_core/tasks/create_ome_zarr/#fractal_tasks_core.tasks.create_ome_zarr.create_ome_zarr)
or
[`create_ome_zarr_multiplex`](../reference/fractal_tasks_core/tasks/create_ome_zarr_multiplex/#fractal_tasks_core.tasks.create_ome_zarr_multiplex.create_ome_zarr_multiplex)
tasks), we always create some default ROI tables.

When importing a Zarr with the import-ome-zarr task...

FIXME

## Handling tables

The `anndata` library offers a set of functions for input/output of AnnData
tables, including functions specifically targeting the Zarr format.

### Reading a table

To read an AnnData table from a Zarr group, one may use the [`read_zarr`
function](https://anndata.readthedocs.io/en/latest/generated/anndata.read_zarr.html).
In the following example a NGFF image was created by sticthing together two
field of views, and the `FOV_ROI_table` has information on the position of the
two original FOVs (named `FOV_1` and `FOV_2`):
```python
import anndata as ad

table = ad.read_zarr("/somewhere/image.zarr/tables/FOV_ROI_table")

print(table)
# AnnData object with n_obs × n_vars = 2 × 8

print(table.obs_names)
# Index(['FOV_1', 'FOV_2'], dtype='object', name='FieldIndex')

print(table.var_names)
# Index([
#        'x_micrometer',
#        'y_micrometer',
#        'z_micrometer',
#        'len_x_micrometer',
#        'len_y_micrometer',
#        'len_z_micrometer',
#        'x_micrometer_original',
#        'y_micrometer_original'
#       ],
#       dtype='object')

print(table.X)
# [[    0.      0.      0.    416.    351.      2.  -1448.3 -1517.7]
#  [  416.      0.      0.    416.    351.      2.  -1032.3 -1517.7]]
```

### Writing a table

The `anndata.experimental.write_elem` function provides the required
functionality to write an AnnData object to a Zarr group. In
`fractal-tasks-core`, the `write_table` helper function wraps the `anndata`
function and includes additional functionalities -- see [its
documentation](../reference/fractal_tasks_core/lib_write/#fractal_tasks_core.lib_write.write_table).


```python
def write_table(
    image_group: zarr.hierarchy.Group,
    table_name: str,
    table: ad.AnnData,
    overwrite: bool = False,
    table_attrs: Optional[dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
)
```


## Future updates

These specifications may evolve (especially based on the future NGFF updates), eventually leading to breaking changes in V2.
Development of `fractal-tasks-core` will mantain backwards-compatibility with V1 for a reasonable amount of time.

Some aspects that most likely will require a review are:

1. We aim at removing the use of hard-coded units from the column names (e.g. `x_micrometer`), in favor of a more general definition of units.
2. We may re-assess whether AnnData tables are the right tool for our scopes, or whether simpler dataframes (e.g. from `pandas`) are sufficient. Not clear whether this is easily doable with zarr though.
parquet in zarr?

https://github.com/zarr-developers/community/issues/31
https://github.com/zarr-developers/numcodecs/issues/452
