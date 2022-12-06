"""
Copyright 2022 (C)
    Friedrich Miescher Institute for Biomedical Research and
    University of Zurich

    Original authors:
    Joel Lüthi  <joel.luethi@fmi.ch>

    This file is part of Fractal and was originally developed by eXact lab
    S.r.l.  <exact-lab.it> under contract with Liberali Lab from the Friedrich
    Miescher Institute for Biomedical Research and Pelkmans Lab from the
    University of Zurich.

Functions to create a metadata dataframe from Yokogawa files
"""
import matplotlib.pyplot as plt

from fractal_tasks_core.lib_remove_FOV_overlaps import is_overlapping_2D


def _plot_rectangle(min_x, min_y, max_x, max_y, overlapping):
    x = [min_x, max_x, max_x, min_x, min_x]
    y = [min_y, min_y, max_y, max_y, min_y]
    if overlapping:
        plt.plot(x, y, ",-", lw=2.5, zorder=2)
    else:
        plt.plot(x, y, ",-", lw=0.3, c="k", zorder=1)


def check_well_for_FOV_overlap(
    site_metadata, selected_well, always_plot=False
):
    df = site_metadata.loc[selected_well].copy()
    df["xmin"] = df["x_micrometer"]
    df["ymin"] = df["y_micrometer"]
    df["xmax"] = df["x_micrometer"] + df["pixel_size_x"] * df["x_pixel"]
    df["ymax"] = df["y_micrometer"] + df["pixel_size_y"] * df["y_pixel"]

    xmin = list(df.loc[:, "xmin"])
    ymin = list(df.loc[:, "ymin"])
    xmax = list(df.loc[:, "xmax"])
    ymax = list(df.loc[:, "ymax"])
    num_lines = len(xmin)

    list_overlapping_FOVs = []
    for line_1 in range(num_lines):
        min_x_1, max_x_1 = [a[line_1] for a in [xmin, xmax]]
        min_y_1, max_y_1 = [a[line_1] for a in [ymin, ymax]]
        for line_2 in range(line_1):
            min_x_2, max_x_2 = [a[line_2] for a in [xmin, xmax]]
            min_y_2, max_y_2 = [a[line_2] for a in [ymin, ymax]]
            overlap = is_overlapping_2D(
                (min_x_1, min_y_1, max_x_1, max_y_1),
                (min_x_2, min_y_2, max_x_2, max_y_2),
            )
            if overlap:
                list_overlapping_FOVs.append(line_1)
                list_overlapping_FOVs.append(line_2)

    if always_plot or (len(list_overlapping_FOVs) > 0):
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

        # Increase values by one to switch from index to the label plotted
        return {selected_well: [x + 1 for x in list_overlapping_FOVs]}


def run_overlap_check(site_metadata):
    """
    Runs an overlap check over all wells, plots overlaps & returns
    """

    wells = site_metadata.index.unique(level="well_id")
    overlapping_FOVs = []
    for selected_well in wells:
        overlap_curr_well = check_well_for_FOV_overlap(
            site_metadata, selected_well=selected_well
        )
        if overlap_curr_well:
            overlapping_FOVs.append(overlap_curr_well)

    return overlapping_FOVs
