import time

from fractal_tasks_core.cellvoyager.wells import generate_row_col_split

well_ids = []
for row_base in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
    for col_base in ["01", "02", "03", "04", "05", "06"]:
        for row_suff in ["a", "b", "c", "d", "e", "f"]:
            for col_suff in ["1", "2", "3", "4", "5", "6"]:
                well_ids.append(f"{row_base}{col_base}.{row_suff}{col_suff}")

start = time.perf_counter()
output = generate_row_col_split(well_ids)
end = time.perf_counter()

print(f"{len(well_ids)=}")
print(f"{well_ids[:2]=}")
print(f"{output[:2]=}")
print(f"Elapsed time: {end - start:e} s")
