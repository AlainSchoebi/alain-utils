# NumPy
import numpy as np

# Python
from typing import Dict, Any, List, List, Optional
from pathlib import Path

# Plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    pass

# Utils
from alutils.decorators import requires_package
from alutils.loggers import get_logger
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
    - shared_x_axis_identifier: `str` the name of the shared x-axis. Note that
                                the name only serves as identification and the
                                actual string is irrelevant.

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
                        logger.error(
                            f"Invalid use of 'colspan'. The entry of the " +
                            f"plot at `plot[{i}][{jj}] should be `None` or " +
                            f"inexistent."
                        )
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

                    logger.error(
                        f"Invalid use of 'rowspan' or 'colspan'. The entry " +
                        f"of the plot at `plot[{i}][{jj}] should be `None` " +
                        f"or inexistent."
                    )
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
    shared_x_axis_identifiers = set()
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
            elif isinstance(trace, (go.Scatter, go.Histogram)):
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
                logger.error(f"Type '{type(trace)}' is not implemented.")
                raise NotImplementedError(f"Type '{type(trace)} is not " +
                                          f"implemented.")

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

            # Shared x-axes
            if 'shared_x_axis_identifier' in entry:
                shared_x_axis_identifiers.add(entry['shared_x_axis_identifier'])
        specs.append(specs_row)

    # Make plot
    fig = make_subplots(
        rows=rows, cols=cols, specs=specs, subplot_titles=titles
    )

    # Build plot
    axes_counter = 0
    shared_x_axis_identifiers_list = list(shared_x_axis_identifiers)
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

            # Shared x-axis
            if 'shared_x_axis_identifier' in entry:
                idx = shared_x_axis_identifiers_list \
                    .index(entry['shared_x_axis_identifier'])
                fig.update_xaxes(matches=f"x{idx+1}", row=i+1, col=j+1)

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
                      hovermode="x unified", **scatter_3d_viewpoints)

    # Save HTML file
    if output_html is not None:
        fig.write_html(str(output_html), include_mathjax='cdn')

    # Open browser
    if open_browser:
        fig.show()

    return None


