from typing import Optional, Tuple

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qdax.utils.plotting import get_voronoi_finite_polygons_2d


def plot_2d_map_elites_repertoire(
    centroids: jnp.ndarray,
    repertoire_fitnesses: jnp.ndarray,
    minval: jnp.ndarray,
    maxval: jnp.ndarray,
    repertoire_descriptors: Optional[jnp.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_centroids: Optional[bool] = True,
) -> Tuple[Optional[Figure], Axes]:
    # TODO: check it and fix it if needed
    repertoire_empty = repertoire_fitnesses == -jnp.inf
    num_descriptors = centroids.shape[1]
    if num_descriptors != 2 and use_centroids:
        raise NotImplementedError(
            "repertoire plot supports 2 descriptors only for now."
        )

    my_cmap = cm.viridis

    fitnesses = repertoire_fitnesses
    if vmin is None:
        vmin = float(jnp.min(fitnesses[~repertoire_empty]))
    if vmax is None:
        vmax = float(jnp.max(fitnesses[~repertoire_empty]))

    # set the parameters
    font_size = 12
    params = {
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "figure.figsize": [10, 10],
    }

    mpl.rcParams.update(params)

    # create the plot object
    fig = None
    if ax is None:
        fig, ax = plt.subplots(facecolor="white", edgecolor="white")

    assert (
        len(np.array(minval).shape) < 2
    ), f"minval : {minval} should be float or couple of floats"
    assert (
        len(np.array(maxval).shape) < 2
    ), f"maxval : {maxval} should be float or couple of floats"

    if len(np.array(minval).shape) == 0 and len(np.array(maxval).shape) == 0:
        ax.set_xlim(minval, maxval)
        ax.set_ylim(minval, maxval)
    else:
        ax.set_xlim(minval[0], maxval[0])
        ax.set_ylim(minval[1], maxval[1])

    ax.set(adjustable="box", aspect="equal")

    # create the regions and vertices from centroids
    if use_centroids:
        regions, vertices = get_voronoi_finite_polygons_2d(centroids)

    norm = Normalize(vmin=vmin, vmax=vmax)

    if use_centroids:
        # fill the plot with contours
        for region in regions:
            polygon = vertices[region]
            ax.fill(
                *zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1
            )

        # fill the plot with the colors
        for idx, fitness in enumerate(fitnesses):
            if fitness > -jnp.inf:
                region = regions[idx]
                polygon = vertices[region]

                ax.fill(*zip(*polygon), alpha=0.8, color=my_cmap(norm(fitness)))

    # if descriptors are specified, add points location
    if repertoire_descriptors is not None:
        descriptors = repertoire_descriptors[~repertoire_empty]
        ax.scatter(
            descriptors[:, 0],
            descriptors[:, 1],
            c=fitnesses[~repertoire_empty],
            cmap=my_cmap,
            s=10,
            zorder=0,
        )

    # aesthetic
    # ax.set_xlabel("Behavior Dimension 1")
    # ax.set_ylabel("Behavior Dimension 2")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
    cbar.ax.tick_params(labelsize=font_size)

    # ax.set_title("MAP-Elites repertoire")
    ax.set_aspect("equal")

    return fig, ax
