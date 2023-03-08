# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def heatmap(
    data: np.array,
    ax: Optional[Any] = None,
    xtick_labels: Optional[List[str]] = None,
    ytick_labels: Optional[List[str]] = None,
    cbar_kwargs: Optional[Dict[str, Any]] = None,
    cbar_label: Optional[str] = None,
    fmt="{x:.2f}",
    **kwargs,
) -> None:
    """"""

    # Create the axis and plot the heatmap
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(data, **kwargs)

    # Create the color bar
    if cbar_kwargs is None:
        cbar_kwargs = {}
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kwargs)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    # Display all ticks
    if xtick_labels is None:
        xtick_labels = [i for i in range(data.shape[1])]
    ax.set_xticks(np.arange(data.shape[1]), labels=xtick_labels)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)

    if ytick_labels is None:
        ytick_labels = [i for i in range(data.shape[0])]
    ax.set_yticks(np.arange(data.shape[0]), labels=ytick_labels)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

    # Adjust the grid layout and ticks positioning
    ax.spines[:].set_visible(False)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(
        which="minor",
        top=False,
        bottom=False,
        left=False,
        labeltop=True,
        labelbottom=False
    )

    # Annotate the heatmap
    if isinstance(fmt, str):
        fmt = matplotlib.ticker.StrMethodFormatter(fmt)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            im.axes.text(
                j,
                i,
                fmt(data[i, j], None),
                horizontalalignment="center",
                verticalalignment="center"
            )
