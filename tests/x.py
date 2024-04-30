import time

from fractal_tasks_core.cellvoyager.wells import generate_row_col_split

start = time.perf_counter()
well_ids = ["B03.d1" for _ in range(1536)]
generate_row_col_split(well_ids)
end = time.perf_counter()
print(f"Elapsed time: {end - start} s")
