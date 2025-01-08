import logging
from pathlib import Path

import anndata as ad
import pandas as pd
import zarr
from anndata._io.specs import write_elem


def _add_empty_ROI_table(
    image_zarr_path: Path,
    table_name: str = "empty_ROI_table",
):
    # Define an empty ROI table
    df_columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
        "label",
    ]
    empty_df = pd.DataFrame(columns=df_columns)
    empty_ROI_table = ad.AnnData(empty_df)

    # Write the ROI table into the tables group
    group_tables = zarr.group(str(image_zarr_path / "tables"))
    write_elem(group_tables, table_name, empty_ROI_table)
    logging.info(
        f"Writing empty ROI table '{table_name}' "
        f"into {str(image_zarr_path)}/tables"
    )
