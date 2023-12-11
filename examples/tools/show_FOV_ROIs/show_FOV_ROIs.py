"""
An example of visualizing FOV ROIs and their overlaps.
"""
import matplotlib.pyplot as plt

from fractal_tasks_core.lib_ROI_overlaps import run_overlap_check
from fractal_tasks_core.yokogawa.metadata import (
    parse_yokogawa_metadata,
)


def _plot_rectangle(min_x, min_y, max_x, max_y, overlapping):
    x = [min_x, max_x, max_x, min_x, min_x]
    y = [min_y, min_y, max_y, max_y, min_y]
    if overlapping:
        plt.plot(x, y, ",-", lw=2.5, zorder=2)
    else:
        plt.plot(x, y, ",-", lw=0.3, c="k", zorder=1)


def _plotting_function(
    xmin, xmax, ymin, ymax, list_overlapping_FOVs, selected_well
):
    plt.figure()
    num_lines = len(xmin)
    for line in range(num_lines):
        min_x, max_x = [a[line] for a in [xmin, xmax]]
        min_y, max_y = [a[line] for a in [ymin, ymax]]

        _plot_rectangle(
            min_x, min_y, max_x, max_y, line in list_overlapping_FOVs
        )
        plt.text(
            0.5 * (min_x + max_x),
            0.5 * (min_y + max_y),
            f"{line + 1}",
            ha="center",
            va="center",
            fontsize=14,
        )

    plt.gca().set_aspect(1)
    plt.xlabel("x (um)", fontsize=12)
    plt.ylabel("y (um)", fontsize=12)
    plt.title(f"Well {selected_well}")

    plt.figure()
    for line in range(num_lines):
        min_x, max_x = [a[line] for a in [xmin, xmax]]
        min_y, max_y = [a[line] for a in [ymin, ymax]]

        _plot_rectangle(
            min_x, min_y, max_x, max_y, line in list_overlapping_FOVs
        )
        plt.text(
            0.5 * (min_x + max_x),
            0.5 * (min_y + max_y),
            f"{line + 1}",
            ha="center",
            va="center",
            fontsize=14,
        )

    plt.gca().set_aspect(1)
    plt.xlabel("x (um)", fontsize=12)
    plt.ylabel("y (um)", fontsize=12)
    plt.title(f"Well {selected_well}")


if __name__ == "__main__":
    mlf_path = "MeasurementData_2x2_well.mlf"
    mrf_path = "MeasurementDetail_2x2_well.mrf"
    site_metadata, total_files = parse_yokogawa_metadata(mrf_path, mlf_path)

    plt.close()
    run_overlap_check(
        site_metadata, tol=0, plotting_function=_plotting_function
    )
    plt.savefig("fig_tol_0.pdf")

    plt.close()
    run_overlap_check(
        site_metadata, tol=1e-10, plotting_function=_plotting_function
    )
    plt.savefig("fig_tol_1e-10.pdf")
