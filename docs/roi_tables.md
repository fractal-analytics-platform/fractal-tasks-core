# ROI tables

We need to store tables as part of NGFF groups for multiple reasons:

1. Our image-to-OME-Zarr converters stitch all the field of views (FOV) of a given well together in a single NGFF image, and we keep a trace of the original FOV positions in a ROI table.
2. Several tasks in `fractal-tasks-core` take a ROI table as an input, an loop over the ROIs defined in the table rows. This offers some flexibility to the tasks, as they can process a well, a set of FOVs, or a set of custom regions of the array.
3. We store ROIs associated to segmeneted objects, for instance the bounding boxes of organoid/nuclear
4. We store measurements associated to segmented objects (e.g. as computed via `regionprops` from `scikit-image`, as wrapped in [napari-skimage-regionprops](https://github.com/haesleinhuepf/napari-skimage-regionprops)).


## Specifications

The current section describes the first version (V1) of `fractal-tasks-core`
tables, which is based on [a proposed update to NGFF
specs](https://github.com/ome/ngff/pull/64); this update is currently on hold,
and `fractal-tasks-core` will evolve as soon as the official specs will adopt a
new definition.
As in the original NGFF proposed update, the current specifications are
specifically based on AnnData tables -- see [section below](#anndata-tables).

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

The Zarr attributes of the `tables` group must include the key `"tables"`,
pointing to the list of all tables; this simplifies the discovery of image
tables.
Here is an example of `image.zarr/tables/.zattrs`:
```json
{
    "tables": [
        "table_1",
        "table_2",
    ]
}
```

#### Single table (standard)

For each table, the Zarr attributes must include the key
`"fractal_roi_table_version"`, pointing to the version of this specification
(e.g. `"1"`).

Here is an example of `image.zarr/tables/table1/.zattrs`
```json
{
    "fractal_roi_table_version": "1",
    "encoding-type": "anndata",      # Automatically added by AnnData
    "encoding-version": "0.1.0",     # Automatically added by AnnData
}
```

This is the kind of tables that are used in `fractal-tasks-core` to store ROIs
like the whole well or the list of field of views.

#### Single table (advanced)

When table rows correspond to segmented objects.

Moreover, they must include the key-value pairs proposed in https://github.com/ome/ngff/pull/64, that is:

> * Attributes MUST contain `"type"`, which is set to `"ngff:region_table"`.
> * Attributes MUST contain `"region"`, which is the path to the data the table is annotating.
> * `"region"` MUST be a single path (single region) or an array of paths (multiple regions).
> * `"region"` paths MUST be objects with a key "path" and the path value MUST be a string.
> * Attributes MUST contain `"region_key"` if `"region"` is an array. `"region_key"` is the key in `obs` denoting which region a given row corresponds to.


Here is an example of `image.zarr/tables/table1/.zattrs`
```json
{
    "fractal_roi_table_version": "1",
    "type": "ngff:region_table",
    "instance_key": "label",
    "region": {
        "path": "../labels/label_DAPI",
    },
    "encoding-type": "anndata",      # Automatically added by AnnData
    "encoding-version": "0.1.0",     # Automatically added by AnnData
}
```

### AnnData tables

On-disk (zarr), see https://anndata.readthedocs.io/en/latest/fileformat-prose.html

## Example

the

which may act for instance on FOVs or on pre-computed

makes the

tbaleiterate


tra Therefore we use
 perform s

https://github.com/ome/ngff/pull/64

in progress


```python
    ROI_table = ad.read_zarr(ROI_table_path)
    attrs = zarr.group(ROI_table_path).attrs
    if not attrs["type"] == "ngff:region_table":
        raise ValueError("Wrong attributes for {ROI_table_path}:\n{attrs}")
    label_relative_path = attrs["region"]["path"]
    column_name = attrs["instance_key"]

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
