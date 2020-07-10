import numpy as np
from matplotlib import pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
from matplotlib import cm

"""
EXAMPLE CODE: Plot astro data

    from manifolder.plotting import plot_clusters
    import pandas as pd

    df = pd.read_csv('manifolder/code/astro/astro_subset_clustering_k=7.csv')
    df.columns = ['Bx', 'By', 'Bz', 'BL', 'BM', 'BN',
                  'Bmaд', 'Vx', 'Vy', 'Vz', 'Vmaд',
                  'np', 'Tpar', 'Tper', 'Tp', 'Snip Label',
                  'Snip Number', 'Cluster Index']

    snip_label = 1
    for snip_number in df[df['Snip Label'] == snip_label]['Snip Number'].unique():

        selected = df[(df['Snip Label'] == snip_label)  & (df['Snip Number'] == snip_number)]
        data = selected['BN']
        labels = selected['Cluster Index']
        n_clusters = max(df['Cluster Index']) + 1

        fig, ax = plt.subplots(figsize=(15, 8))

        plot_clusters(
            data.index, data.values, labels,
            n_clusters=n_clusters, cbar_label='Cluster Index',
            ax=ax, lw=2
        )

        ax.set_ylabel(data.name)
        ax.set_title(f"Snip Label = {snip_label}, Snip Number = {snip_number}")
"""


def plot_segments(x, y, colors=None, cmap=cm.viridis, ax=None, **kwargs):
    """
        Draws line-segments (a LineCollection object) to the current axis.
        Very similar to plt.plot(...), except that each line segment
        (joining consecutive points in the plot) can have its own color!
        Specify a sequence of numeric values (of the same length as x and y)
        to the colors kwarg, additionally specifying a colormap if desired.

        TODO: I am unsure whether the colors are 'pre' or 'post', i.e.
        whether they are based on the color associated with the first
        or second point in the segment. It would be nice to expose this
        to the user and allow them to specify which behavior they want
        in a kwarg.
    """
    if ax is None:
        ax = plt.gca()

    points = np.stack((x, y), -1)
    segments = np.stack([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, array=colors, capstyle='round', cmap=cmap, **kwargs)
    if 'capstyle' not in kwargs:
        lc.set_capstyle('round')

    ax.add_collection(lc)
    ax.autoscale_view(tight=True)

    return ax


def plot_clusters(x, y, labels, n_clusters=None, cmap=cm.tab10, cbar_label=None, ax=None, **kwargs):
    """
        Uses plot_segments to plot a sequence of connected line-segments with
        categorical colorings.

        Specify n_clusters if there are more clusters than are being represented
        in the current plot.
    """
    if ax is None:
        ax = plt.gca()

    if n_clusters is None:
        # Try to infer; assume smallest label is 0
        n_clusters = max(labels)+1

    norm = BoundaryNorm(np.arange(-.5, n_clusters+.5), n_clusters)
    ax = plot_segments(x, y, labels, cmap=cmap, norm=norm, ax=ax, **kwargs)
    mappable = ax.collections[0]  # Extract our  LineCollection
    cb = plt.colorbar(mappable, ticks=np.arange(0, n_clusters), ax=ax)
    cb.set_label(cbar_label)

    return ax
