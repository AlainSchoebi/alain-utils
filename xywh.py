# Typing
from __future__ import annotations
from typing import Optional, Union, List

# Numpy
import numpy as np
from numpy.typing import NDArray

# Matplotlib
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class XYWH:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.__x = x
        self.__y = y
        self.__w = w
        self.__h = h
        self._check()

    def _check(self):
        # Integer
        if not type(self.x) == int or not type(self.y) == int or \
           not type(self.w) == int or not type(self.h) == int:

            raise NotImplementedError("Only integers values are supported " +
                                      "for creating a XYWH instance.")
        # Positive width and height
        if self.w < 0 or self.h < 0:
            raise ValueError("The width and height of XYWHs must be positive " +
                             "values.")
    @property
    def x(self) -> int:
        return self.__x

    @property
    def y(self) -> int:
        return self.__y

    @property
    def w(self) -> int:
        return self.__w

    @property
    def h(self) -> int:
        return self.__h

    def center(self) -> NDArray:
        return np.sum(self.corner_coordinates()[[0, 2]], axis=0) / 2

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.w
        yield self.h

    def corner_coordinates(self) -> NDArray:
        """
        Compute the four corner coordinates:
          - in a CCW manner when the x-axis points right and the y-axis down
          - in a CW manner when the x-axis points right and the y-axis up

        Returns:
        - corners: (4, 2) array containing the four corners
        """
        return np.array([[self.x, self.y],
                         [self.x, self.y + self.h - 1],
                         [self.x + self.w - 1, self.y + self.h - 1],
                         [self.x + self.w - 1, self.y]], dtype=int)

    def xyxy(self) -> NDArray:
        return np.array([[self.x, self.y],
                         [self.x + self.w - 1, self.y + self.h - 1]], dtype=int)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"XYWH(x={self.x}, y={self.y}, " + \
               f"w={self.w}, h={self.h})"


    # Visualization functions
    if MATPLOTLIB_AVAILABLE :
        def show(self, axes: Optional[Axes] = None) -> Axes:
            """
            Visualize the XYWH in a matloptlib plot.
            """
            return XYWH.visualize(self, axes)

        @staticmethod
        def visualize(xywhs: Union[XYWH, List[XYWH]],
                      axes: Optional[Axes] = None) -> Axes:
            """
            Visualize a list of XYWH in a matloptlib plot.

            Inputs:
            - xywhs: list of XYWHs to plot
            """

            if type(xywhs) == XYWH:
                xywhs = [xywhs]

            # No axes provided
            if axes is None:
                # Create figure
                fig = plt.figure()
                ax: Axes = fig.add_subplot()

                # Title
                ax.set_title(f"XYWH{'s' if len(xywhs) > 1 else ''} " +
                             f"visualization")

                # Axis labels
                ax.set_xlabel('x')
                ax.set_ylabel('y')

            # Axes provided
            else:
                ax = axes

            # Default color cycle
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            colors = [matplotlib.colors.to_rgb(color) for color in colors]


            # Plot poses
            for i, (xywh, color) in enumerate(zip(xywhs, colors)):

                # Rectangle
                rectangle = matplotlib.patches.Rectangle(
                    (xywh.x - 0.5, xywh.y -0.5), xywh.w, xywh.h,
                    edgecolor=np.array(color) * 0.7, facecolor=color,
                   )
                ax.add_patch(rectangle)

                # Labels
                if len(xywhs) > 1 or hasattr(xywh, "label") or hasattr(xywh, "name"):
                    label = i
                    if hasattr(xywh, "name"):
                        label = xywh.name
                    elif hasattr(xywh, "label"):
                        label = xywh.label
                    ax.text(*xywh.center(), label, ha='center', va='center',
                            color=np.array(color) * 0.7, fontsize=12)

            if axes is None:
                # Axis parameters
                ax.axis('equal')
                ax.xaxis.set_major_locator(
                    matplotlib.ticker.MaxNLocator(integer=True)
                )
                ax.yaxis.set_major_locator(
                    matplotlib.ticker.MaxNLocator(integer=True)
                )
                ax.invert_yaxis()

                # Show
                plt.show()

            return ax
