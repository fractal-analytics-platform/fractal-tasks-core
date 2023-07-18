from typing import Sequence

import anndata as ad

from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)


def _inspect_ROI_table(
    path: str,
    full_res_pxl_sizes_zyx: Sequence[float],
    level: int = 0,
    coarsening_xy: int = 2,
) -> None:
    """
    Description

    :param dummy: this is just a placeholder
    :type dummy: int
    """

    print(f"{full_res_pxl_sizes_zyx=}")

    adata = ad.read_zarr(path)
    df = adata.to_df()
    print("table")
    print(df)
    print()

    try:
        list_indices = convert_ROI_table_to_indices(
            adata,
            level=level,
            coarsening_xy=coarsening_xy,
            full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
            # verbose=True,
        )
        print()
        print(f"level:         {level}")
        print(f"coarsening_xy: {coarsening_xy}")
        print("list_indices:")
        for indices in list_indices:
            print(indices)
        print()
    except KeyError as e:
        print("Something went wrong in convert_ROI_table_to_indices\n", str(e))

    return df
