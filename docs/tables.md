# Tables

Within `fractal-tasks-core`, we make use of tables which are `AnnData` objects
stored within OME-Zarr image groups. This page describes the different kinds of
tables we use, and it includes:

* A core [table specification](#core-tables), valid for all tables;
* The definition of [tables for regions of interests (ROIs)](#roi-tables);
* The definition of [masking ROI tables](#masking-roi-tables), namely ROI tables that are linked e.g. to labels;
* A [feature-table specification](#feature-tables), to store measurements.

> ⚠️  **Warning**: As of version 0.13 of `fractal-tasks-core`, the
> specifications below are not yet fully implemented (see issue
> [602](https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/602)
> and
> [593](https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/593)).
<div></div>
> **Note**: The specifications below are largely inspired by [a proposed update
> to OME-NGFF specs](https://github.com/ome/ngff/pull/64). This update is currently
> on hold, and `fractal-tasks-core` will evolve as soon as an official NGFF
> table specs is adopted - see also the [Outlook](#outlook) section.

## Specifications (V1)

In this section we describe verion 1 of the Fractal table specification, which
is currently the only one.

### Core tables

The core-table specification consists in the definition of the required Zarr
structure and attributes, and of the `AnnData` table format.

**`AnnData` table format**

We store tabular data into Zarr groups as `AnnData` ("Annotated Data") objects;
the [`anndata` Python library](https://anndata.readthedocs.io) provides the
definition of this format and the relevant tools. Quoting from the `anndata`
documentation:

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

**Zarr structure and attributes**

The structure of Zarr groups is based on the [`image` specification in NGFF
0.4](https://ngff.openmicroscopy.org/0.4/index.html#image-layout), with an
additional `tables` group and the corresponding subgroups (similar to
`labels`):
```hl_lines="12 13 14 15"
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

The Zarr attributes of the `tables` group must include the key `tables`,
pointing to the list of all tables (this simplifies discovery of tables
associated to the current NGFF image), as in
```json title="image.zarr/tables/.zattrs"
{
    "tables": ["table_1", "table_2"]
}
```

The Zarr attributes of each specific-table group must include the version of
the table specification (currently version 1), through the
`fractal_table_version` attribute. Also note that the `anndata` function to
write an `AnnData` object into a Zarr group automatically sets additional
attributes. Here is an example of the resulting Zarr attributes:
```json title="image.zarr/tables/table_1/.zattrs"
{
    "fractal_table_version": "1",
    "encoding-type": "anndata",    // Automatically added by anndata 0.11
    "encoding-version": "0.1.0",   // Automatically added by anndata 0.11
}
```

### ROI tables

In `fractal-tasks-core`, a ROI table defines regions of space which are
three-dimensional (see also the [Outlook section](#outlook) about
dimensionality flexibility) and box-shaped.
Typical use cases are described [here](#use-cases-for-roi-tables).

**Zarr attributes**

The specification of a ROI table is a subset of the [core table
one](#core-tables). Moreover, the table-group Zarr attributes must include the
`type` attribute with value `roi_table`, as in
```json title="image.zarr/tables/table_1/.zattrs" hl_lines="3"
{
    "fractal_table_version": "1",
    "type": "roi_table",
    "encoding-type": "anndata",
    "encoding-version": "0.1.0",
}
```

**Table columns**

The [`var`
attribute](https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.var.html#anndata.AnnData.var)
of a given `AnnData` object indexes the columns of the table. A
`fractal-tasks-core` ROI table must include the following six columns:

* `x_micrometer`, `y_micrometer`, `z_micrometer`:
  the lower bounds of the XYZ intervals defining the ROI, in micrometers;
* `len_x_micrometer`, `len_y_micrometer`, `len_z_micrometer`:
  the XYZ edge lengths, in micrometers.

> Notes:
>
> 1. The **axes origin** for the ROI positions (e.g. for `x_micrometer`)
>    corresponds to the top-left corner of the image (for the YX axes) and to
>    the lowest Z plane.
> 2. ROIs are defined in **physical coordinates**, and they do not store
>    information on the number or size of pixels.

ROI tables may also include other columns, beyond the required ones. Here are
the ones that are typically used in `fractal-tasks-core` (see also the [Use
cases](#roi-tables-use-cases) section):

* `x_micrometer_original` and `y_micrometer_original`, which are a copy of
  `x_micrometer` and `y_micrometer` taken before applying some transformation;
* `translation_x`, `translation_y` and `translation_z`, which are used during
  registration of multiplexing cycles;
* `label`, which is used to link a ROI to a label (either for
  [masking ROI tables](#masking-roi-tables) or for
  [feature tables](#feature-tables)).

### Masking ROI tables

Masking ROI tables are a specific instance of the basic ROI tables described
above, where each ROI must also be associated to a specific label of a label
image.

**Motivation**

The motivation for this association is based on the following use case:

* By performing segmentation of a NGFF image, we identify N objects and we
  store them as a label image (where the value at each pixel correspond to the
  label index);
* We also compute the three-dimensional bounding box of each segmented object,
  and store these bounding boxes into a `masking` ROI table;
* For each one of these ROIs, we also include information that link it to both
  the label image and a specific label index;
* During further processing we can load/modify specific sub-regions of the ROI,
  based on information contained in the label image. This kind of operations
  are `masked`, as they only act on the array elements that match a certain
  condition on the label value.

**Zarr attributes**

For this kind of tables, `fractal-tasks-core` closely follows the [proposed
NGFF update mentioned above](https://github.com/ome/ngff/pull/64). The
requirements on the Zarr attributes of a given table are:

* Attributes must contain a `type` key, with value `masking_roi_table`[^2].
* Attributes must contain a `region` key; the corresponding value must be an
  object with a `path` key and a string value (i.e. the path to the data the
  table is annotating).
* Attributes must include a key `instance_key`, which is the key in `obs` that
  denotes which instance in `region` the row corresponds to.

Here is an example of valid Zarr attributes
```json title="image.zarr/tables/table_1/.zattrs" hl_lines="3 4 5"
{
    "fractal_table_version": "1",
    "type": "masking_roi_table",
    "region": { "path": "../labels/label_DAPI" },
    "instance_key": "label",
    "encoding-type": "anndata",
    "encoding-version": "0.1.0",
}
```

**Table columns**

On top of the required ROI-table colums, a masking ROI table must include the
column which is defined in its `instance_key` attribute (e.g. the `label`
column, for the example above).

### Feature tables

**Motivation**

The typical use case for feature tables is to store measurements related to
segmented objects, while mantaining a link to the original instances (e.g.
labels). Note that the current specification is aligned to the one of [masking
ROI tables](#masking-roi-tables), since they both need to relate a table to a
label image, but the two may diverge in the future.

As part of the current `fractal-tasks-core` tasks, measurements can be
performed e.g. via `regionprops` from `scikit-image`, as wrapped in
[napari-skimage-regionprops](https://github.com/haesleinhuepf/napari-skimage-regionprops)).

**Zarr attributes**

For this kind of tables, `fractal-tasks-core` closely follows the [proposed
NGFF update mentioned above](https://github.com/ome/ngff/pull/64). The
requirements on the Zarr attributes of a given table are:

* Attributes must contain a `type` key, with value `feature_table`[^2].
* Attributes must contain a `region` key; the corresponding value must be an
  object with a `path` key and a string value (i.e. the path to the data the
  table is annotating).
* Attributes must include a key `instance_key`, which is the key in `obs` that
  denotes which instance in `region` the row corresponds to.

Here is an example of valid Zarr attributes
```json title="image.zarr/tables/table_1/.zattrs" hl_lines="3 4 5"
{
    "fractal_table_version": "1",
    "type": "feature_table",
    "region": { "path": "../labels/label_DAPI" },
    "instance_key": "label",
    "encoding-type": "anndata",
    "encoding-version": "0.1.0",
}
```

**Table columns**

A feature table must include the column which is defined in its `instance_key`
attribute (e.g. the `label` column, for the example above).

## Examples

### Use cases for ROI tables

#### OME-Zarr creation

OME-Zarrs created via `fractal-tasks-core` (e.g. by parsing Yokogawa images via
the
[`create_ome_zarr`](../reference/fractal_tasks_core/tasks/create_ome_zarr/#fractal_tasks_core.tasks.create_ome_zarr.create_ome_zarr)
or
[`create_ome_zarr_multiplex`](../reference/fractal_tasks_core/tasks/create_ome_zarr_multiplex/#fractal_tasks_core.tasks.create_ome_zarr_multiplex.create_ome_zarr_multiplex)
tasks) always include two specific ROI tables:

* The table named `well_ROI_table`, which covers the NGFF image corresponding to the whole well[^1];
* The table named `FOV_ROI_table`, which lists all original fields of view (FOVs).

Each one of these two tables includes ROIs that span the whole image size along
the Z axis. Note that this differs, e.g., from ROIs which are the bounding
boxes of three-dimensional segmented objects, and which may cover only a part
of the image Z size.

#### OME-Zarr import

When working with an externally-generated OME-Zarr, one may use the
[`import_ome_zarr`
task](../reference/fractal_tasks_core/tasks/import_ome_zarr/#fractal_tasks_core.tasks.import_ome_zarr.import_ome_zarr)
to make it compatible with `fractal-tasks-core`. This task optionally adds two
ROI tables to the NGFF images:

* The table named `image_ROI_table`, which covers the whole image;
* A table named `grid_ROI_table`, which splits the whole-image ROI into a YX
  rectangular grid of smaller ROIs. This may correspond to original FOVs (in
  case the image is a tiled well[^1]), or it may simply be useful for applying
  downstream processing to smaller arrays and avoid large memory requirements.

As for the case of `well_ROI_table` and `FOV_ROI_table` described
[above](#ome-zarr-creation), also these two tables include ROIs spanning the
whole image extension along the Z axis.

#### OME-Zarr processing

ROI tables are also used and updated during image processing, e.g as in:

* The FOV ROI table may undergo transformations during processing, e.g. FOV
  ROIs may be shifted to avoid overlaps; in this case, we use the optional
  columns `x_micrometer_original` and `y_micrometer_original` to store the values
  before the transformation.
* The FOV ROI table is also used to store information on the registration of
  multiplexing cycles, via the `translation_x`, `translation_y` and
  `translation_z` optional columns.
* Several tasks in `fractal-tasks-core` take an existing ROI table as an input
  and then loop over the ROIs defined in the table. This makes the task more
  flexible, as it can be used to process e.g. a whole well, a set of FOVs, or a
  set of custom regions of the array.

### Reading/writing tables

The `anndata` library offers a set of functions for input/output of AnnData
tables, including functions specifically targeting the Zarr format.

#### Reading a table

To read an `AnnData` table from a Zarr group, one may use the [`read_zarr`
function](https://anndata.readthedocs.io/en/latest/generated/anndata.read_zarr.html).
In the following example a NGFF image was created by stitching together two
field of views, where each one is made of a stack of five Z planes with 1 um
spacing between the planes.
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

df = table.to_df()  # Convert to pandas DataFrame
print(df)
#             x_micrometer  y_micrometer  z_micrometer  ...  len_z_micrometer  x_micrometer_original  y_micrometer_original
# FieldIndex                                            ...
# FOV_1                0.0           0.0           0.0  ...               2.0           -1448.300049           -1517.699951
# FOV_2              416.0           0.0           0.0  ...               2.0           -1032.300049           -1517.699951
#
# [2 rows x 8 columns]
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
from fractal_tasks_core.lib_tables import write_table

table = ad.AnnData(X=np.ones((10, 10)))  # Generate a dummy `AnnData` object
image_group = zarr.open_group("/tmp/image.zarr")
table_name = "MyTable"
table_attrs = {
    "type": "feature_table",
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
    "fractal_table_version": "1",
    "instance_key": "label",
    "region": {
        "path": "../labels/MyLabel"
    },
    "type": "feature_table"
}
```

## Outlook

These specifications may evolve (especially based on the future NGFF updates),
eventually leading to breaking changes in future versions.
`fractal-tasks-core` will aim at mantaining backwards-compatibility with V1 for
a reasonable amount of time.

Here is an in-progress list of aspects that may be reviewed:

* We aim at removing the use of hard-coded units from the column names (e.g.
  `x_micrometer`), in favor of a more general definition of units.
* The `z_micrometer` and `len_z_micrometer` columns are currently required in
  all ROI tables, even when the ROIs actually define a two-dimensional XY
  region; in that case, we set `z_micrometer=0` and `len_z_micrometer` is such
  that the whole Z size is covered (that is, `len_z_micrometer` is the product
  of the spacing between Z planes and the number of planes). In a future
  version, we may introduce more flexibility and also accept ROI tables which
  only include X and Y axes, and adapt the relevant tools so that they
  automatically expand these ROIs into three-dimensions when appropriate.
* Concerning the use of `AnnData` tables or other formats for tabular data, our
  plan is to follow whatever serialised table specification becomes part of the
  NGFF standard. For the record, Zarr does not natively support storage of
  dataframes (see e.g.
  https://github.com/zarr-developers/numcodecs/issues/452), which is one aspect
  in favor of sticking with the `anndata` library.

[^1]:
Within `fractal-tasks-core`, NGFF images represent whole wells; this still
complies with the NGFF specifications, as of an [approved clarification in the
specs](https://github.com/ome/ngff/pull/137). This explains the reason for
storing the regions corresponding to the original FOVs in a specific ROI table,
since one NGFF image includes a collection of FOVs. Note that this approach
does not rely on the assumption that the FOVs constitute a regular tiling of
the well, but it also covers the case of irregularly placed FOVs.

[^2]:
Note that the table types `masking_roi_table` and `feature_table` closely
resemble the `type="ngff:region_table"` specification in the previous [proposed
NGFF table specs](https://github.com/ome/ngff/pull/64).
