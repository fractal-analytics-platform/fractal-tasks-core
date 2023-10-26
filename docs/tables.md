# Fractal tables

Within `fractal-tasks-core` we make use of tables stored as `AnnData` objects
within OME-Zarr image groups. This page describes the specifications for
different kinds of tables:

* A core [table specification](#core-tables), common to all cases;
* Two levels of specifications for tables that describe regions of interest (ROIs):
    * [Basic ROI tables](#basic-roi-tables); (FIXME: better naming?)
    * [Advanced ROI tables](#advanced-roi-tables); (FIXME: better naming?)
* A [feature-table specification](#feature-tables). (FIXME: specify this is in progress)

These different specifications correspond to different use cases in `fractal-tasks-core`:

* Basic ROI tables:
    * We keep track of the positions of the Field of Views (FOVs) within a well, after stitching the corresponding FOV images into a single whole-well array.
    * We keep track of the original state before some transformations are applied - e.g. shifting FOVs to avoid overlaps, or shifting a multiplexing cycle during registration.
    * Several tasks in `fractal-tasks-core` take an existing ROI table as an input and then loop over the ROIs defined in the table. Such tasks have more flexibility, as they can process e.g. a whole well, a set of FOVs, or a set of custom regions of the array.
* Advanced ROI tables:
    * We store ROIs associated to segmented objects, for instance the bounding boxes of organoids/nuclei.
* Feature tables:
    * We store measurements associated to segmented objects (e.g. as computed via `regionprops` from `scikit-image`, as wrapped in [napari-skimage-regionprops](https://github.com/haesleinhuepf/napari-skimage-regionprops)).

> **Note**: The specifications below are largely based on [a proposed update to
> NGFF specs](https://github.com/ome/ngff/pull/64). This update is currently on
> hold, and `fractal-tasks-core` will evolve as soon as the NGFF specs will
> adopt a definition of tables - see also the [Outlook](#outlook) section.

## Specifications

### Core tables

The core-table specification consists in the definition of the required Zarr
structure and attributes, and of the `AnnData` table format.

#### `AnnData` table format

Data of a table are stored into a Zarr group as `AnnData` ("Annotated Data")
objects; the [`anndata` Python library](https://anndata.readthedocs.io) provides the
definition of this format and the relevant tools.

Quoting from `anndata` documentation:

> `AnnData` is specifically designed for matrix-like data. By this we mean that
> we have $n$ observations, each of which can be represented as $d$-dimensional
> vectors, where each dimension corresponds to a variable or feature. Both the
> rows and columns of this $n \times d$ matrix are special in the sense that
> they are indexed.
>
> (https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html)

Note that `AnnData` tables are easily transformed from/into `pandas.DataFrame`
objects - see e.g. the [`AnnData.to_df`
method](https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.to_df.html#anndata.AnnData.to_df).

#### Zarr structure

The structure of Zarr groups is based on the [`image` specification in NGFF
0.4](https://ngff.openmicroscopy.org/0.4/index.html#image-layout), with an
additional `tables` group and the corresponding subgroups (similar to
`labels`):
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
└── tables        # Zarr subgroup with a list of tables associated to this image
    ├── table_1   # Zarr subgroup for a given table
    ├── table_2   # Zarr subgroup for a given table
    └── ...
```

#### Zarr attributes

The Zarr attributes of the `tables` group must include the key `tables`,
pointing to the list of all tables; this simplifies the discovery of image
tables.

Here is an example of `image.zarr/tables/.zattrs`:
```json
{
    "tables": ["table_1", "table_2"]
}
```

The Zarr attributes of each specific-table group have no required properties,
but writing an `AnnData` object to that group typically sets some default
attributes. For anndata 0.11, for instance, the attributes in
`image.zarr/tables/table1/.zattrs` would be
```json
{
    "encoding-type": "anndata",   # Automatically added by anndata
    "encoding-version": "0.1.0",  # Automatically added by anndata
}
```

### ROI tables

The current section describes the first version (V1) of `fractal-tasks-core`
tables, which is based on [a proposed update to NGFF
specs](https://github.com/ome/ngff/pull/64); this update is currently on hold,
and `fractal-tasks-core` will evolve as soon as the NGFF specs will adopt a
definition of tables.
As in the original proposed NGFF update, the current specifications are
specifically based on `AnnData` tables.

In Fractal, regions of interest (ROIs) are three-dimensional
regions of space delimited by orthogonal planes. ROI tables are stored as

#### Basic ROI tables

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

##### Required columns

The [`var` attribute of AnnData
objects](https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.var.html#anndata.AnnData.var)
indexes the columns of the table. A `fractal-tasks-core` ROI table must include
the following six columns:

* `x_micrometer`, `y_micrometer`: the lower bounds of the XY intervals defining the ROI, in micrometers;
* `z_micrometer`: the lower bound of the Z interval defining the ROI, in arbitrary units or in micrometers;
* `len_x_micrometer`, `len_y_micrometer`: the XY edge lenghts, in micrometers;
* `len_z_micrometer`: the Z edge lenght in arbitrary units (corresponding to the number of Z planes) or in micrometers.

> Notes:
>
> 1. The **axes origin** for the ROI positions (e.g. for `x_micrometer`) is set
>    to coincide with the top-left corner of a well (for the YX axes) and with
>    the lowest Z plane.
> 2. ROIs are defined in **physical coordinates**, and they do not store
>    information on the number or size of pixels.
> 3. The current version of `fractal-tasks-core` only uses **arbitrary units**
>    for `z_micrometer` and `len_z_micrometer` columns, where a single unit
>    corresponds to the distance between two subsequent Z planes.

##### Other columns

ROI tables may also include abitrary columns. Here are the ones that are
typically used in `fractal-tasks-core`:

* `x_micrometer_original` and `y_micrometer_original`, which are a copy of `x_micrometer` and `y_micrometer` taken before applying some transformation;
* `label`, which is used within measurement tables as a reference to the labels corresponding to a row of measurements (see [description of `instance_key` below](#single-table-segmented-objects)).
* FIXME: add `translation_x/y/z` columns






#### Advanced ROI tables (FIXME: rename?)

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

### Feature tables

FIXME: to do

## Examples

### Default ROI tables

OME-Zarrs created via `fractal-tasks-core` (e.g. by parsing Yokogawa images via
the
[`create_ome_zarr`](../reference/fractal_tasks_core/tasks/create_ome_zarr/#fractal_tasks_core.tasks.create_ome_zarr.create_ome_zarr)
or
[`create_ome_zarr_multiplex`](../reference/fractal_tasks_core/tasks/create_ome_zarr_multiplex/#fractal_tasks_core.tasks.create_ome_zarr_multiplex.create_ome_zarr_multiplex)
tasks) always include two specific ROI tables:

* The table named `FOV_ROI_table`, which lists all original FOVs;
* The table named `well_ROI_table`, which covers the NGFF image corresponding to the whole well (formed by all the original FOVs stiched together)

Each one of these two tables includes ROIs that are only defined in the XY
plane, and span the whole set of Z planes. Note that this differs, e.g., from
the case of bounding-box ROIs based on three-dimensional segmented objects,
which may have a non-trivial Z size.

When working with an externally-generated OME-Zarr, one may use the
[`import_ome_zarr`
task](../reference/fractal_tasks_core/tasks/import_ome_zarr/#fractal_tasks_core.tasks.import_ome_zarr.import_ome_zarr)
to make it compatible with `fractal-tasks-core`. This task optionally adds two
ROI tables to the NGFF images:

* The table named `image_ROI_table`, which simply covers the whole image.
* A table named `grid_ROI_table`, which splits the whole-image ROI into a YX
  rectangular grid of smaller ROIs. This may correspond to original FOVs, or it
  may simply be useful for applying downstream processing to smaller arrays and
  avoid large memory requirements.


### Reading/writing tables

The `anndata` library offers a set of functions for input/output of AnnData
tables, including functions specifically targeting the Zarr format.

#### Reading a table

To read an `AnnData` table from a Zarr group, one may use the [`read_zarr`
function](https://anndata.readthedocs.io/en/latest/generated/anndata.read_zarr.html).
In the following example a NGFF image was created by sticthing together two
field of views, where each one is made of a stack of five Z planes.
The `FOV_ROI_table` has information on the XY position and size of the two
original FOVs (named `FOV_1` and `FOV_2`):
```python
import anndata as ad

table = ad.read_zarr("/somewhere/image.zarr/tables/FOV_ROI_table")

print(table)
# `AnnData` object with n_obs × n_vars = 2 × 8

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
# [[    0.      0.      0.    416.    351.      5.  -1448.3 -1517.7]
#  [  416.      0.      0.    416.    351.      5.  -1032.3 -1517.7]]
```

In this case, the second FOV (labeled `FOV_2`) is defined as the three-dimensional region such that

* X is between 416 and 832 micrometers;
* Y is between 0 and 351 micrometers;
* Z is between 0 and 5 - which means that all the five available Z planes are included.

#### Writing a table

The `anndata.experimental.write_elem` function provides the required
functionality to write an `AnnData` object to a Zarr group. In
`fractal-tasks-core`, the `write_table` helper function wraps the `anndata`
function and includes additional functionalities -- see [its
documentation](../reference/fractal_tasks_core/lib_write/#fractal_tasks_core.lib_write.write_table).

With respect to the wrapped `anndata` function, the main additional features of `write_table` are

* The boolean parameter `overwrite` (defaulting to `False`), that determines the behavior in case of an already-existing table at the given path.
* The `table_attrs` parameter, as a shorthand for updating the Zarr attributes of the table group after its creation.

Here is an example of how to use `write_table`:
```python
import numpy as np
import zarr
import anndata as ad
from fractal_tasks_core.lib_write import write_table

table = ad.AnnData(X=np.ones((10, 10)))  # Generate a dummy `AnnData` object
image_group = zarr.open_group("/tmp/image.zarr")
table_name = "MyTable"
table_attrs = {
    "type": "ngff:region_table",
    "region": {"path": "../labels/MyLabel"},
    "instance_key": "label",
}

write_table(
    image_group,
    table_name,
    table,
    overwrite=True,
    table_attrs=table_attrs,
)
```
After running this Python code snippet, the on-disk output  is as follows:
```console
$ tree /tmp/image.zarr/tables/                  # View folder structure
/tmp/image.zarr/tables/
└── MyTable
    ├── layers
    ├── obs
    │   └── _index
    │       └── 0
    ├── obsm
    ├── obsp
    ├── uns
    ├── var
    │   └── _index
    │       └── 0
    ├── varm
    ├── varp
    └── X
        └── 0.0

12 directories, 3 files

$ cat /tmp/image.zarr/tables/.zattrs            # View tables atributes
{
    "tables": [
        "MyTable"
    ]
}

$ cat /tmp/image.zarr/tables/MyTable/.zattrs    # View single-table attributes
{
    "encoding-type": "anndata",
    "encoding-version": "0.1.0",
    "fractal_roi_table_version": "1",
    "instance_key": "label",
    "region": {
        "path": "../labels/MyLabel"
    },
    "type": "ngff:region_table"
}
```


## Outlook


These specifications may evolve (especially based on the future NGFF updates),
eventually leading to breaking changes in V2. Development of
`fractal-tasks-core` will aim at mantaining backwards-compatibility with V1 for
a reasonable amount of time.

An in-progress list of aspects that may be reviewed:

1. We aim at removing the use of hard-coded units from the column names (e.g.
   `x_micrometer`), in favor of a more general definition of units. This will
   also fix the current misleading names for the Z position/length columns
   (`z_micrometer` and `len_z_micrometer`, even though corresponding data are
   in arbitrary units).
2. We may re-evaluate whether `AnnData` tables are the most appropriate tool. For
   the record, Zarr does not natively support storage of dataframes (see e.g.
   https://github.com/zarr-developers/numcodecs/issues/452), which is one
   aspect in favor of sticking with the `anndata` library.

---
