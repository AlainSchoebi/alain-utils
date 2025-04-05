# Numpy
import numpy as np
from numpy.typing import NDArray

# Python
from typing import Dict, Any, List, List, Optional, Tuple
from pathlib import Path
import math

# Plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    pass

# Shapely
try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
except ImportError:
    pass

# Utils
from .bbox import BBox
from .decorators import requires_package

# Logging
from .loggers import get_logger
logger = get_logger(__name__)

@requires_package('plotly')
def build_plotly_plot(
    plot: List[List[Dict[str, Any]]],
    title: Optional[str] = "",
    height: Optional[int | None] = None,
    open_browser: Optional[bool] = True,
    output_html: Optional[str | Path | None] = None) -> None:
    """
    Builds a plotly plot from a 2D list of dictionaries. Each dictionary
    describes a subplot of the plot.

    Inputs:
    - plot: `List[List[Dict[str, Any]]]` the 2D list of dictionaries describing
            the plot.

    Each entry in the 2D list of dictionaries MUST have the following keys:
    - title: `str` the title of the subplot.
    - traces: `go.Scatter | go.Image ... | List[go.Scatter | go.Image | ...]`
               the trace(s) of the subplot.

    Each entry in the 2D list of dictionaries CAN have the optional keys:
    - rowspan: `int` the number of rows spanned by the subplot.
    - colspan: `int` the number of columns spanned by the subplot.
    - xlim: `List[float, float]` the x-axis limits.
    - ylim: `List[float, float]` the y-axis limits.
    - secondary_ylim: `List[float, float]` the secondary y-axis limits.
    - viewpoint: `Dict[str, float]` the viewpoint of the 3D plot.
    - xlabel: `str` the label of the x-axis.
    - ylabel: `str` the label of the y-axis.
    - secondary_ylabel: `str` the label of the secondary y-axis.
    - secondary_y_axis_trace_idx: `List[int]` the indices of which traces should
                                  be plotted on the secondary y-axis.

    Optional Inputs:
    - title: `str` the title of the plot.
    - open_browser: `bool` whether to open the plot in the browser. Default is
                    `True`.
    - output_html: `str | Path | None` the path to save the plot as an HTML
                   file. Default is `None`.
    """

    rows = len(plot)
    assert rows > 0 and "Plot cannot be empty"

    # Count number of columns (not very clean...)
    cols = []
    for i, row in enumerate(plot):
        col = len(row)
        for j in range(len(row)):
            if row[j] is not None and 'colspan' in row[j]:
                col += row[j]['colspan'] - 1
                for jj in range(j + 1, j + row[j]['colspan']):
                    if jj >= len(row):
                        break
                    if row[jj] is None:
                        col -= 1
                    else:
                        raise ValueError(
                            f"Invalid use of 'colspan'. The entry of the " +
                            f"plot at `plot[{i}][{jj}] should be `None` or " +
                            f"inexistent."
                        )
            if row[j] is not None and 'rowspan' in row[j]:
                rows = max(rows, i + row[j]['rowspan'])

        cols.append(col)

    cols = max(cols)

    # Boolean matrix indicating whether each subplot is "overwritten" by a
    # colspan/rowspan or not
    is_span = np.full((rows, cols), fill_value=False, dtype=bool)
    for i in range(rows):
        for j in range(cols):
            if j >= len(plot[i]) or plot[i][j] is None:
                continue
            rowspan, colspan = 1, 1
            if 'rowspan' in plot[i][j]:
                rowspan = plot[i][j]['rowspan']
            if 'colspan' in plot[i][j]:
                colspan = plot[i][j]['colspan']

            # Check validity
            for ii in range(i, i + rowspan):
                for jj in range(j, j + colspan):
                    if ii == i and jj == j:
                        continue

                    if ii >= len(plot) or \
                       jj >= len(plot[ii]) or \
                       plot[ii][jj] is None:
                       continue

                    raise ValueError(
                        f"Invalid use of 'rowspan' or 'colspan'. The entry " +
                        f"of the plot at `plot[{i}][{jj}] should be `None` " +
                        f"or inexistent."
                    )

            # Set span
            is_span[i:, j:][:rowspan, :colspan] = True
            is_span[i, j] = False


    # Build specifications and titles of the plot
    specs, titles = [], []
    scatter_3d_viewpoints, scatter_3d_counter = {}, 0
    for i in range(rows):
        specs_row = []
        for j in range(cols):

            # If subplot is "overwritten" by colspan/rowspan
            if is_span[i, j]:
                specs_row.append(None)
                continue

            # If subplot is not directly defined
            if i >= len(plot) or \
               j >= len(plot[i]):
                specs_row.append({})
                titles.append("?")
                continue

            if plot[i][j] is None:
                specs_row.append({})
                titles.append("")
                continue

            # Acces entry
            entry = plot[i][j]
            if not "title" in entry or not "traces" in entry:
                raise ValueError(f"The entry must have a 'title' and a " +
                                 f"'traces' key.")

            # Subplot title
            titles.append(entry["title"])

            # Check if any traces should be plotted
            if entry["traces"] is None or \
               isinstance(entry["traces"], list) and len(entry["traces"]) == 0:
               specs_row.append({})
               continue

            # Pick first trace
            trace = entry["traces"]
            if isinstance(entry["traces"], list):
                trace = entry["traces"][0]

            # Type of subplot
            if isinstance(trace, go.Contour):
                specs_row.append({"type": "contour"})
            elif isinstance(trace, go.Scatter):
                specs_row.append({"type": "xy"})
            elif isinstance(trace, go.Image):
                specs_row.append({"type": "image"})
            elif isinstance(trace, go.Scatter3d) or \
                 isinstance(trace, go.Mesh3d):
                specs_row.append({"type": "scatter3d"})
                scatter_3d_counter += 1
                if 'viewpoint' in entry:
                    scene = f"scene{scatter_3d_counter}_camera"
                    scatter_3d_viewpoints[scene] = entry['viewpoint']
            elif isinstance(trace, go.Table):
                specs_row.append({"type": "table"})
            else:
                raise NotImplementedError(f"Type '{type(trace)} is not " +
                                          f"implemented")

            # Secondary y-axis
            if "secondary_y_axis_trace_idx" in entry:
                specs_row[-1].update({"secondary_y": True})

            # Rowspan and colspan
            rowspan, colspan = 0, 0
            if 'rowspan' in entry:
                rowspan = entry['rowspan']
                specs_row[-1]['rowspan'] = rowspan

            if 'colspan' in entry:
                colspan = entry['colspan']
                specs_row[-1]['colspan'] = colspan

        specs.append(specs_row)

    # Make plot
    fig = make_subplots(
        rows=rows, cols=cols, specs=specs, subplot_titles=titles
    )

    # Build plot
    axes_counter = 0
    for i, row in enumerate(plot):
        for j, entry in enumerate(row):

            if entry is None or entry["traces"] is None:
               continue

            # Add subplots (multiple traces)
            if isinstance(entry["traces"], list):
                if len(entry["traces"]) == 0:
                    continue

                # Handle secondary y axes
                if "secondary_y_axis_trace_idx" in entry:
                    for idx, trace in enumerate(entry["traces"]):
                        secondary_y = idx in entry["secondary_y_axis_trace_idx"]
                        fig.add_trace(trace, row=i+1, col=j+1,
                                      secondary_y=secondary_y)
                else:
                     for trace in entry["traces"]:
                        fig.add_trace(trace, row=i+1, col=j+1)
            # Add subplots (single trace)
            else:
                fig.add_trace(entry["traces"], row=i+1, col=j+1)


            # Pick first trace
            trace = entry["traces"]
            if isinstance(entry["traces"], list):
                trace = entry["traces"][0]

            # Count number of usual x-y axes
            if not isinstance(trace, go.Mesh3d):
                axes_counter += 1

            # Axes labels
            if 'xlabel' in entry:
                fig.update_xaxes(title_text=entry['xlabel'], row=i+1, col=j+1)
            if 'ylabel' in entry:
                fig.update_yaxes(title_text=entry['ylabel'], row=i+1, col=j+1)
            if 'secondary_ylabel' in entry:
                fig.update_yaxes(title_text=entry['secondary_ylabel'],
                                 row=i+1, col=j+1, secondary_y=True)

            # Axes limits
            if not isinstance(trace, (go.Scatter, go.Contour, go.Image)) and \
                any([k in entry for k in ['xlim', 'ylim', 'secondary_ylim']]):
                raise ValueError(
                    f"Using properties `xlim`, `ylim` or `secondary_ylim` is " +
                    f"only supported when using `Scatter`, `Contour` or " +
                    f"`Image` plots. Found plot type `{type(trace)}`.")

            if 'xlim' in entry:
                xlim = entry['xlim']
                if not isinstance(xlim, list) or not len(xlim) == 2 or \
                   None in xlim:
                    raise ValueError(f"The provided xlim='{xlim}' is " +
                                     f"invalid.")
                fig.update_xaxes(range=xlim, row=i+1, col=j+1)
            if 'ylim' in entry:
                ylim = entry['ylim']
                if not isinstance(ylim, list) or not len(ylim) == 2 or \
                    None in ylim:
                    raise ValueError(f"The provided ylim='{ylim}' is " +
                                     f"invalid.")
                fig.update_yaxes(range=ylim, row=i+1, col=j+1,
                                 secondary_y=False)
            if 'secondary_ylim' in entry:
                secondary_ylim = entry['secondary_ylim']
                if not isinstance(secondary_ylim, list) or \
                    not len(secondary_ylim) == 2 or None in secondary_ylim:
                    raise ValueError(f"The provided secondary_ylim='" +
                                     f"{secondary_ylim}' is invalid.")
                fig.update_yaxes(range=secondary_ylim,
                                 row=i+1, col=j+1, secondary_y=True)

            # Automatic axes bounds for go.Image as first trace
            if isinstance(trace, go.Image):
                H, W = trace.z.shape[:2]
                if not 'xlim' in entry and not 'ylim' in entry and \
                    not 'secondary_ylim' in entry:
                    fig.update_xaxes(range=[0, W], row=i+1, col=j+1)
                    fig.update_yaxes(range=[H, 0], row=i+1, col=j+1)

            # Automatic aspect ratio 1:1 for go.Contour as first trace
            if isinstance(trace, go.Contour):
                fig.update_xaxes(constrain='domain', row=i+1, col=j+1)
                fig.update_yaxes(constrain='domain',
                                 scaleanchor=f'x{axes_counter}',
                                 row=i+1, col=j+1,)

    # Layout
    if height is None:
        height = 400 * len(plot)
    fig.update_layout(title=title, showlegend=False, height=height,
                      **scatter_3d_viewpoints)

    # Save HTML file
    if output_html is not None:
        fig.write_html(str(output_html), include_mathjax='cdn')

    # Open browser
    if open_browser:
        fig.show()

    return


@requires_package('shapely')
def get_2d_boundary(vertices: NDArray, faces: NDArray ) -> List[NDArray]:
    """
    Computes the 2D boundary of a 2D mesh defined by its vertices and faces.

    Inputs
    - vertices: `NDArray(N, 2)` the 2D vertices of the mesh.
    - faces: `NDArray(M, 3)` of integers representing the faces of the mesh.

    Returns
    - boundaries: `List[NDArray(N, 2)]` the 2D boundaries of the mesh. Since the
                  mesh can be composed of multiple disjoint parts, the
                  boundaries are returned as a list of 2D vertices.
    """
    assert isinstance(vertices, np.ndarray) and isinstance(faces, np.ndarray)
    if len(vertices) == 0 or len(faces) == 0:
        return []
    assert vertices.ndim == 2 and vertices.shape[-1] == 2
    assert faces.ndim == 2 and faces.shape[-1] == 3 and faces.dtype == np.int32

    triangles = []
    for face in faces:
        triangle = Polygon([vertices[face[i]] for i in range(3)])
        triangles.append(triangle)

    try:
        merged_polygon = unary_union(triangles)
    except Exception as e:
        logger.error(f"Error while computing the 2D boundary of the mesh: {e}")
        return []

    boundaries = []
    if isinstance(merged_polygon, Polygon):
        boundaries.append(np.array(merged_polygon.exterior.coords))
    elif isinstance(merged_polygon, MultiPolygon):
        for polygon in merged_polygon.geoms:
            boundaries.append(np.array(polygon.exterior.coords))

    return boundaries


@requires_package('plotly')
def gaussian_1d_traces(
    mu: float, var: float, *values: float, S: int = 100,
    color: Optional[str] = 'cyan') -> List[go.Contour | go.Scatter]:
    """
    Generates the traces for a 1D Gaussian distribution. It plots the PDF.

    The function only draws the contour on a square image of size `H x W`, and
    `S` points are used to draw the ellipses.

    Inputs:
    - mu: `float` the mean of the Gaussian distribution.
    - cov: `float` the variance of the Gaussian distribution.
    - *values: `float` any number of x-values to plot on the PDF.
    - S: `int` the number of points used to draw the PDF.

    Returns:
    - traces: `List[go.Scatter]` the traces of the Gaussian distribution.
    """

    if var <= 0:
        raise ValueError(f"Variance of the Gaussian distribution must be " +
                         f"striclty postive to be plotted, found '{var}'.")

    traces = []

    # PDF of the Gaussian distribution
    if var > 1e-8:
        std = math.sqrt(var)
        x = np.linspace(mu - 3 * std, mu + 3 * std, S) # (S, )
        y = 1/math.sqrt(2 * math.pi * var) * np.exp(-(x-mu)**2 / (2 * var))

        pdf_trace = go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(
                width=1,
                color=color,
            ),
            name="PDF",
        )
        traces.append(pdf_trace)

    # Mean bar
    mean_trace = go.Scatter(
        x=[mu, mu],
        y=[0, 1/math.sqrt(2 * math.pi * var)],
        mode='lines',
        line=dict(
            color=color,
            dash='dash'
        ),
        name=f"mean {mu}",
    )
    traces.append(mean_trace)

    # Bars for every provided value to plot
    for value in values:
        value_trace = go.Scatter(
            x=[value, value],
            y=[0, 1/math.sqrt(2 * math.pi * var)],
            mode='lines',
            line=dict(
                color='lime',
                dash='dash'
            ),
            name=f"x-value {value}",
        )
        traces.append(value_trace)

    return traces


@requires_package('plotly')
def gaussian_2d_traces(
    mu: NDArray, cov: NDArray, output_size: BBox, S: int = 100,
    primary_color: str = "cyan", secondary_color: str = "blue",
    colorscale: str | List = "Viridis"
    ) -> List[go.Contour | go.Scatter]:
    """
    Generates the traces for a 2D Gaussian distribution. It plots the mean, the
    1 standard deviation and the 2 standard deviation ellipses, and the contour
    of the Gaussian distribution.

    The function only draws the contour on a square image of size `H x W`, and
    `S` points are used to draw the ellipses.

    Inputs:
    - mu: `NDArray(2,)` the mean of the Gaussian distribution.
    - cov: `NDArray(2, 2)` the covariance matrix of the Gaussian distribution.
    - output_size: `BBox` defining the size of the output image.
    - S: `int` the number of points used to draw the ellipses.

    Optional inputs:
    - primary_color:   `str` the primary color used for drawing the curves.
    - secondary_color: `str` the secondary color used for drawing the curves.
    - colorscale:     `str | List[...]` the colorscale used for drawing the
                       contours.

    Returns:
    - traces: `List[go.Contour | go.Scatter]` the traces of the Gaussian
               distribution.
    """

    assert mu.shape == (2,) and cov.shape == (2, 2)
    if np.linalg.det(cov) < 1e-15:
        #TODO IMPROVE
        return [
            go.Scatter(
               x=[mu[0]],
               y=[mu[1]],
               mode="markers",
               marker=dict(
                   size=5,
                   color="red",
                   opacity=0.7,
                   symbol="cross",
               ),
               name="mu",
            )
            ]

    # Covariance matrix
    cov_inv = np.linalg.inv(cov) # (2, 2)
    eigvals, eigvecs = np.linalg.eigh(cov_inv)

    # Compute the principal vectors of the ellipse x^T COV^{-1} x = 1
    P = eigvecs / np.sqrt(eigvals) # (2, 2)

    # Generate points on the ellipse
    thetas = np.linspace(0, 2 * np.pi, S) # (S,)
    cos_sin = np.c_[np.cos(thetas), np.sin(thetas)][..., None] # (S, 2, 1)
    points_1std = mu[:, None] + P @ cos_sin # (S, 2, 1)
    points_2std = mu[:, None] + 2 * P @ cos_sin # (S, 2, 1)
    points_1std = points_1std[..., 0] # (S, 2)
    points_2std = points_2std[..., 0] # (S, 2)

    # Contour of the Gaussian distribution
    x = np.linspace(output_size.x1, output_size.x2, S) # (S,)
    y = np.linspace(output_size.y1, output_size.y2, S) # (S,)
    X, Y = np.meshgrid(x, y) # (S, S, 2)
    XY = np.stack([X, Y], axis=-1)[..., None] # (S, S, 2, 1)
    XY = XY - mu[:, None] # (S, S, 2, 1)
    XY_T = XY.swapaxes(-2, -1) # (S, S, 1, 2)
    Z = XY_T @ cov_inv @ XY # (S, S, 1, 1)
    Z = Z[:, :, 0, 0] # (S, S)

    gaussian_mu_trace = go.Scatter(
        x=[mu[0]],
        y=[mu[1]],
        mode="markers",
        marker=dict(
            size=5,
            color=primary_color,
            opacity=0.7,
            symbol="cross",
        ),
        name="mu",
    )

    gaussian_ellipse_1std_trace = go.Scatter(
        x=points_1std[:, 0], y= points_1std[:, 1],
        mode="lines",
        line=dict(
            width=1,
            color=secondary_color,
        ),
        name="1 std",
    )

    gaussian_ellipse_2std_trace = go.Scatter(
        x=points_2std[:, 0], y= points_2std[:, 1],
        mode="lines",
        line=dict(
            width=1,
            color=primary_color,
        ),
        name="2 std",
    )

    gaussian_contour_trace = go.Contour(
        x=x, y=y, z=Z,
        contours=dict(coloring='heatmap'),
        colorscale=colorscale,
        showscale=False,
        opacity=0.5,
        name="x^T COV^-1 x",
    )

    return [gaussian_contour_trace, gaussian_mu_trace,
            gaussian_ellipse_1std_trace, gaussian_ellipse_2std_trace]


def bound_values_nice(x: NDArray):
    return np.median(x[x >= np.median(x)])


def bin_to_plot(
    x: NDArray, y: NDArray,
    num_bins: Optional[int] = 100,
    min_x: Optional[float] = 0, max_x: Optional[float | None] = None,
    return_y_cov: Optional[float] = False
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Bin some (x, y) datapoints into bins in order to plot them.

    Inputs
    - x: `NDArray(N)`
    - y: `NDArray(N)` or `NDArray(N, d)`

    Returns
    - mean_x:    `NDArray(num_bins)`
    - mean_y:    `NDArray(num_bins)` or `NDArray(num_bins, d)`
    - std_y:     `NDArray(num_bins)`, `NDArray(num_bins, d)` or \
                 `NDArray(num_bins, d, d)`
    - bin_sizes: `NDArray(num_bins)`

    Optional Inputs
    - num_bins:     `int` number of desired bins. Default is 100.
    - min_x:        `float` the lower bound for the plotted x values. Default is
                    `0`.
    - max_x:        `float` the upper bound for the plotted x values. Default is
                    `None`, which will compute the upper bound via the
                    `bound_75_percent` function.
    - return_y_cov: `bool` if `True`, for multi-dimensional y data (i.e. d>1),
                    the function returns the standard deviation instead of the
                    variance of the y data. Default is `False`.
    """
    # Assertions
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected input data `x` to be of type `NDArray`, " +
                        f"but found `{type(x)}`.")
    if not isinstance(y, np.ndarray):
        raise TypeError(f"Expected input data `y` to be of type `NDArray`, " +
                        f"but found {type(y)}.")
    if not (y.ndim == 1 or y.ndim == 2):
        raise ValueError(f"Expected input data `y` have one or two" +
                         f"dimensions, but found shape `{y.shape}`.")
    if return_y_cov and y.ndim == 1:
        raise ValueError(f"Cannot set `return_y_cov` to `True` for one-" +
                         f"dimensional y data.")

    if not return_y_cov and y.ndim > 1:
        raise ValueError(f"Need to set `return_y_cov` to `True` when using " +
                         f"multi-dimensional y data.")

    N = len(x)
    if not x.shape == (N,):
        raise ValueError(f"Expected input data `x` to have shape `(N,)`, " +
                        f"but found `{x.shape}`.")
    if not (y.shape == (N,) or y.ndim == 2 and y.shape == (N, y.shape[-1])) :
        raise ValueError(f"Expected input data `y` to have shape `(N,)` or "
                        f"`(N, d)` with N={N}, but found `{y.shape}`.")

    # Initialize output arrays
    mean_x, mean_y, std_y, bin_sizes = [], [], [], []

    # Upper bound of x values to consider
    if max_x is None:
        max_x = bound_values_nice(x)

    # Compute bins and bin the y values
    for b in range(num_bins):

        # Bin range
        l = min_x + (max_x - min_x) / num_bins * b
        r = min_x + (max_x - min_x) / num_bins * (b+1)
        m_x = (l + r) / 2

        # Select datapoints belonging to the bin
        mask = (x >= l) & (x < r)
        bin_x, bin_y = x[mask], y[mask]

        # Compute the mean and the standard deviation (or covariance)
        if len(bin_y) > 0:
            mean_x.append(m_x)
            mean_y.append(bin_y.mean(axis=0))
            std_y.append(
                np.std(bin_y, axis=0) if not return_y_cov else np.cov(bin_y.T)
            )
            bin_sizes.append(len(bin_x))

    # Return mean_x, mean_y, std_y (or cov_y) and bin_sizes
    return np.array(mean_x), \
           np.array(mean_y), np.array(std_y), \
           np.array(bin_sizes)
