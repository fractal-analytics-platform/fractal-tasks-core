import sys

from fractal_tasks_core.lib_regions_of_interest import _inspect_ROI_table


table = sys.argv[1]
level = int(sys.argv[2])
full_res_pxl_sizes_zyx = [5.0, 0.325, 0.325]
_inspect_ROI_table(
    table,
    level=level,
    coarsening_xy=2,
    full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
)
