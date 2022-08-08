# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Plotting functions that visualizes Pareto-frontiers.
"""

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go


def plot_2d_pareto(visited: Dict[str, List[Any]],
                   pareto: Dict[str, List[Any]],
                   parents: Optional[Dict[str, List[Any]]] = None,
                   hover_template: Optional[str] = None,
                   title_text: Optional[str] = None,
                   xaxis_title: Optional[str] = None,
                   yaxis_title: Optional[str] = None,
                   output_path: Optional[str] = None) -> None:
    """Plots a 2-dimensional visualization of the Pareto-frontier.

    Args:
        visited: Visited architectures points.
        pareto: Pareto-frontier architectures points.
        parents: Parents architectures points.
        hover_template: Template used when hovering over point (HTML-only).
        title_text: Title of the plot.
        xaxis_title: Title of the `x` axis.
        yaxis_title: Title of the `y` axis.
        output_path: Path to the plot output files (HTML and PNG files).

    """

    fig = go.Figure()

    x_visited, y_visited, config_visited = visited['x'], visited['y'], visited['config']
    fig.add_trace(go.Scatter(x=x_visited, 
                             y=y_visited, 
                             mode='markers',
                             marker_color='blue',
                             showlegend=True,
                             name='All visited architectures',
                             hovertemplate=hover_template,
                             text=[repr(config) for config in config_visited]))

    x_pareto, y_pareto, config_pareto = pareto['x'], pareto['y'], pareto['config']
    fig.add_trace(go.Scatter(x=x_pareto,
                             y=y_pareto,
                             mode='markers',
                             marker_color='red',
                             showlegend=True,
                             name='Pareto architectures',
                             hovertemplate=hover_template,
                             text=[repr(config) for config in config_pareto]))

    if parents:
        x_parents, y_parents, config_parents = parents['x'], parents['y'], parents['config']
        fig.add_trace(go.Scatter(x=x_parents,
                                 y=y_parents,
                                 mode='markers',
                                 marker_color='green',
                                 showlegend=True,
                                 name='Parent architectures',
                                 hovertemplate=hover_template,
                                 text=[repr(config) for config in config_parents]))

    fig.update_layout(title_text=title_text,
                      xaxis_title=xaxis_title,
                      yaxis_title=yaxis_title)

    html_path = f'{output_path}.html'
    fig.write_html(html_path)

    png_path = f'{output_path}.png'
    fig.write_image(png_path, engine='kaleido', width=1500, height=1500, scale=1)


def plot_3d_pareto(visited: Dict[str, List[Any]],
                   pareto: Dict[str, List[Any]],
                   parents: Optional[Dict[str, List[Any]]] = None,
                   hover_template: Optional[str] = None,
                   title_text: Optional[str] = None,
                   xaxis_title: Optional[str] = None,
                   yaxis_title: Optional[str] = None,
                   zaxis_title: Optional[str] = None,
                   output_path: Optional[str] = None) -> None:
    """Plots a 3-dimensional visualization of the Pareto-frontier.

    Args:
        visited: Visited architectures points.
        pareto: Pareto-frontier architectures points.
        parents: Parents architectures points.
        hover_template: Template used when hovering over point (HTML-only).
        title_text: Title of the plot.
        xaxis_title: Title of the `x` axis.
        yaxis_title: Title of the `y` axis.
        zaxis_title: Title of the `z` axis.
        output_path: Path to the plot output files (HTML and PNG files).
        
    """

    fig = go.Figure()

    x_visited, y_visited, z_visited, config_visited = visited['x'], visited['y'], visited['z'], visited['config']
    fig.add_trace(go.Scatter3d(x=x_visited, 
                               y=y_visited, 
                               z=z_visited,
                               mode='markers',
                               marker_color='blue',
                               showlegend=True,
                               name='All visited architectures',
                               hovertemplate=hover_template,
                               text=[repr(config) for config in config_visited]))

    x_pareto, y_pareto, z_pareto, config_pareto = pareto['x'], pareto['y'], pareto['z'], pareto['config']
    fig.add_trace(go.Scatter3d(x=x_pareto,
                               y=y_pareto,
                               z=z_pareto,
                               mode='markers',
                               marker_color='red',
                               showlegend=True,
                               name='Pareto architectures',
                               hovertemplate=hover_template,
                               text=[repr(config) for config in config_pareto]))

    if parents:
        x_parents, y_parents, z_parents, config_parents = parents['x'], parents['y'], parents['z'], parents['config']
        fig.add_trace(go.Scatter3d(x=x_parents,
                                   y=y_parents,
                                   z=z_parents,
                                   mode='markers',
                                   marker_color='green',
                                   showlegend=True,
                                   name='Parent architectures',
                                   hovertemplate=hover_template,
                                   text=[repr(config) for config in config_parents]))

    fig.update_layout(title_text=title_text,
                      scene=dict(xaxis_title=xaxis_title,
                                 yaxis_title=yaxis_title,
                                 zaxis_title=zaxis_title))

    html_path = f'{output_path}.html'
    fig.write_html(html_path)

    png_path = f'{output_path}.png'
    fig.write_image(png_path, engine='kaleido', width=1500, height=1500, scale=1)
