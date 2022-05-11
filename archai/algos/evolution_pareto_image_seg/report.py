from pathlib import Path
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

from archai.common.common import get_expdir
from archai.nas.arch_meta import ArchWithMetaData
from archai.nas.constraints.pareto_frontier import find_pareto_frontier_points


def get_search_status_df(all_pop: List[ArchWithMetaData], pareto: List[ArchWithMetaData], iter_nb: int, 
                         fields: List[str]):
    """Gets a dataframe with the search status of the current iteration.

    Args:
        all_pop (List[ArchWithMetaData]): All visited architectures.
        pareto (List[ArchWithMetaData]): Pareto architectures.
        iter_nb (int): Current iteration number.
        fields: (List[str]): List of fields to include in the dataframe.

    Returns:
        pd.DataFrame: A dataframe with the search status of the current iteration.
    """
    assert 'archid' in fields, 'archid must be included in the fields'

    all_data = {
        k: [p.metadata[k] if k in p.metadata else None for p in all_pop]
        for k in fields
    }
    pareto_ids = [p.metadata['archid'] for p in pareto]

    status_df = pd.DataFrame(all_data)
    status_df['nb_iterations'] = iter_nb
    status_df['is_pareto'] = False
    status_df.loc[status_df['archid'].isin(pareto_ids), 'is_pareto'] = True

    return status_df


def save_3d_pareto_plot(all_pop: List[ArchWithMetaData], pareto: List[ArchWithMetaData],
                        dimensions: List[str], iter_nb: int, save_dir: str):
    """Saves a 3D plot of the pareto front.

    Args:
        all_pop (List[ArchWithMetaData]): List of all visited architectures.
        pareto (List[ArchWithMetaData]): List of pareto architectures.
        dimensions (List[str]): List of three dimensions to plot.
        iter_nb (int): Current iteration number.
        save_dir (str): Directory to save the plot.
    """
    assert len(dimensions) == 3
    x, y, z = dimensions

    all_data, pareto_data = [
        {k: [p.metadata[k] for p in pop] for k in dimensions + ['archid'] }
        for pop in [all_pop, pareto]
    ]

    # Converts archids to str
    all_data['archid'] = list(map(str, all_data['archid']))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=all_data[x],
                               y=all_data[y], 
                               z=all_data[z],
                               text=all_data['archid'],
                               mode='markers',
                               marker_color='blue',
                               showlegend=True,
                               name='All visited architectures'))

    fig.add_trace(go.Scatter3d(x=pareto_data[x], 
                               y=pareto_data[y], 
                               z=pareto_data[z],
                               text=pareto_data['archid'],
                               mode='markers',
                               marker_color='red',
                               showlegend=True,
                               name='Pareto architectures'))

    title_text = f'Search State Iter {iter_nb}'
    xaxis_title = 'Accuracy (validation f1)'
    yaxis_title = 'Latency (microseconds)'
    zaxis_title = 'Memory'

    fig.update_layout(title_text=title_text,
                      scene=dict(xaxis_title=xaxis_title,
                                 yaxis_title=yaxis_title,
                                 zaxis_title=zaxis_title))

    html_path = Path(save_dir) / f'search_state_{iter_nb}.html'
    fig.write_html(html_path)

    png_path = Path(save_dir) / f'search_state_{iter_nb}.png'
    fig.write_image(png_path, engine='kaleido', width=1500, height=1500, scale=1) 


def save_2d_pareto_evolution_plot(status_df: pd.DataFrame, x: str, y: str,
                                  save_path: str,
                                  x_increasing: bool = False, y_increasing: bool = False,
                                  max_x: Optional[float] = None,
                                  max_y: Optional[float] = None):
    """Saves a 2D pareto front evolution plot for two dimensions from a search status dataframe generated
    by `get_search_status_df`.

    Args:
        status_df (pd.DataFrame): Search status dataframe.
        x (str): First dimension.
        y (str): Second dimension.
        save_path (str): Path to save the plot.
        x_increasing (bool): Whether the first dimension is increasing.
        y_increasing (bool): Whether the second dimension is increasing.
        max_x (Optional[float]): Maximum value for the first dimension.
        max_y (Optional[float]): Maximum value for the second dimension.
    """
    status_df = status_df.copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    nb_iter = status_df['nb_iterations'].max()
    status_range = range(0, nb_iter + 1)
    
    # Transforms dimensions to be decreasing if necessary
    max_x, max_y = max_x or status_df[x].max(), max_y or status_df[y].max()
    status_df['x'] = status_df[x] if not x_increasing else (max_x - status_df[x])
    status_df['y'] = status_df[y] if not y_increasing else (max_y - status_df[y])

    colors = plt.cm.plasma(np.linspace(0, 1, nb_iter + 1))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=nb_iter + 1))

    for s in status_range:
        generation_df = status_df.query(f'generation <= {s}').copy()
        points = generation_df[['x', 'y']].values
        pareto_df = generation_df.iloc[find_pareto_frontier_points(points)].copy()
        pareto_df = pareto_df.query(f'{x} <= {max_x} and {y} <= {max_y}').sort_values(x)
        ax.step(pareto_df[x], pareto_df[y], where='post', color=colors[s])
        ax.plot(pareto_df[x], pareto_df[y], 'o', color=colors[s])
    
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.colorbar(sm, ax=ax)
    fig.savefig(save_path)


