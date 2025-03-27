from __future__ import annotations

# Typing
from typing import Tuple, List, Optional

# Python
import re

# NumPy
import numpy as np
from numpy.typing import NDArray

# Matplotlib
from matplotlib import colors as mcolors

class Color:

    def __init__(
            self,
            # TODO Color(r, g, b) instead of Color((r,g,b))
            arg: str | \
                 Tuple[int, int, int] | \
                 Tuple[float, float, float] | \
                 Tuple[int, int, int, int] | \
                 Tuple[float, float, float, float] | \
                 List[int] | \
                 List[float] | \
                 NDArray = "black",
            opacity: float | int | None = None
        ):

        self.__a = None

        # NDArray
        if isinstance(arg, np.ndarray):
            if arg.ndim != 1 or len(arg) not in (3, 4):
                raise ValueError(f"Invalid NDArray shape: {arg.shape}")

            self.r, self.g, self.b = arg[:3]
            if len(arg) == 4:
                self.a = arg[3]

        # Tuple
        if isinstance(arg, tuple):
            if len(arg) not in (3, 4):
                raise ValueError(f"Invalid color input: {arg}")

            self.r, self.g, self.b = arg[:3]
            if len(arg) == 4:
                self.a = arg[3]

        # String
        elif isinstance(arg, str):

            # Examples: "rgb(255, 12, 120)" or "rgba(255, 255, 231, 135)"
            if arg.startswith("rgb"):

                try:
                    values = tuple(map(int, re.findall(r'\d+', arg)))
                except:
                    raise ValueError(f"Invalid string format: {arg}")

                if not len(values) == 3 or len(values) == 4:
                    raise ValueError(f"Invalid string format: {arg}")

                c = Color(values[:3])
                self.__r, self.__g, self.__b, self.__a = c.r, c.g, c.b, c.a

            elif arg.startswith("#"):
                raise NotImplementedError(
                    "Hexadecimal color strings are not supported."
                )

            else:
                try:
                    rgb = mcolors.to_rgb(arg)
                    self.__r, self.__g, self.__b, self.__a = *rgb[:3], None
                except Exception:
                    raise ValueError(
                        f"Invalid color string: {arg}"
                    )

        # Opacity if provided
        if opacity is not None:
            if self.has_alpha:
                raise ValueError(
                    "Opacity provided but color already has an alpha value."
                )

            self.a = opacity

    @staticmethod
    def random(self) -> Color:
        pass

    # R, G, B, A properties
    @property
    def r(self) -> float:
        """ Red value `float`"""
        return self.__r

    @property
    def r_int(self) -> int:
        """ Red value `int` """
        return int(self.r * 255)

    @property
    def g(self) -> float:
        """ Green value `float` """
        return self.__g

    @property
    def g_int(self) -> int:
        """ Green value `int` """
        return int(self.g * 255)

    @property
    def b(self) -> float:
        """ Blue value `float` """
        return self.__b

    @property
    def b_int(self) -> int:
        """ Blue value `int` """
        return int(self.b * 255)

    @property
    def a(self) -> float | None:
        """ Alpha value `float or `None` """
        return self.__a

    @property
    def has_alpha(self) -> bool:
        """ Check if alpha value is set """
        return self.a is not None

    # R, G, B, A setters
    @r.setter
    def r(self, r: float | int) -> None:
        """ Set red value """
        if isinstance(r, float):
            if not 0 <= r <= 1:
                raise ValueError(f"Invalid red value: {r}.")
            self.__r = r
        elif isinstance(r, int):
            if not 0 <= r <= 255:
                raise ValueError(f"Invalid red value: {r}.")
            self.__r = r / 255
        else:
            raise TypeError("Red value must be a `float` or an `int`.")

    @g.setter
    def g(self, g: float | int) -> None:
        """ Set green value """
        if isinstance(g, float):
            if not 0 <= g <= 1:
                raise ValueError(f"Invalid green value: {g}.")
            self.__g = g
        elif isinstance(g, int):
            if not 0 <= g <= 255:
                raise ValueError(f"Invalid green value: {g}.")
            self.__g = g / 255
        else:
            raise TypeError("Green value must be a `float` or an `int`.")

    @b.setter
    def b(self, b: float | int) -> None:
        """ Set blue value """
        if isinstance(b, float):
            if not 0 <= b <= 1:
                raise ValueError(f"Invalid blue value: {b}.")
            self.__b = b
        elif isinstance(b, int):
            if not 0 <= b <= 255:
                raise ValueError(f"Invalid blue value: {b}.")
            self.__b = b / 255
        else:
            raise TypeError("Blue value must be a `float` or an `int`.")

    @a.setter
    def a(self, a: float | None) -> None:
        """ Set alpha value """
        if a is None:
            self.__a = None
        elif isinstance(a, float):
            if not 0 <= a <= 1:
                raise ValueError(f"Invalid alpha value: {a}.")
            self.__a = a
        else:
            raise TypeError("Alpha value must be a `float` or `None`.")

    # Unpacking
    def __iter__(self: Color) -> Tuple[float, float, float] | \
                                 Tuple[float, float, float, float]:
        """ Unpack the color """
        if self.has_alpha:
            return self.rgba_tuple
        else:
            return self.rgb_tuple

    # RGB and RGBA properties
    @property
    def rgb_tuple(self) -> Tuple[float, float, float]:
        """ RGB tuple `(float, float, float)` """
        return (self.r, self.g, self.b)

    @property
    def rgb_int_tuple(self) -> Tuple[int, int, int]:
        """ RGB tuple `(int, int, int)` """
        return (self.r_int, self.g_int, self.b_int)

    @property
    def rgb_list(self) -> List[float]:
        """ RGB list `[float, float, float]` """
        return list(self.rgb_tuple)

    @property
    def rgb_int_list(self) -> List[int]:
        """ RGB list `[int, int, int]` """
        return list(self.rgb_int_tuple)

    @property
    def rgb_array(self) -> NDArray:
        """ RGB array `NDArray` """
        return np.array(self.rgb_tuple)

    @property
    def rgb_int_array(self) -> NDArray:
        """ RGB array `NDArray` """
        return np.array(self.rgb_int_tuple)

    @property
    def rgba_tuple(self, opacity: float | None = None) \
        -> Tuple[float, float, float, float]:
        """ RGBA tuple `(float, float, float, float)` """
        color = self.with_opacity(opacity)
        return color.rgb_tuple + (color.a,)

    @property
    def rgba_int_tuple(self, opacity: float | None = None) \
        -> Tuple[int, int, int, float]:
        """ RGBA tuple `(int, int, int, float)` """
        color = self.with_opacity(opacity)
        return color.rgb_int_tuple + (color.a,)

    @property
    def rgba_list(self, opacity: float | None = None) \
        -> List[float]:
        """ RGBA list `[float, float, float, float]` """
        color = self.with_opacity(opacity)
        return list(color.rgba_tuple)

    @property
    def rgba_int_list(self, opacity: float | None = None) \
        -> List[int]:
        """ RGBA list `[int, int, int, float]` """
        color = self.with_opacity(opacity)
        return list(color.rgba_int_tuple)

    @property
    def rgba_array(self, opacity: float | None = None) -> NDArray:
        """ RGBA array `NDArray` """
        color = self.with_opacity(opacity)
        return np.array(color.rgba_tuple)

    # Strings
    def rgb_string(self) -> str:
        """ RGB string `rgb(int, int, int)` """
        return f"rgb({self.r_int}, {self.g_int}, {self.b_int})"

    def rgba_string(self, opacity: float | None = None) -> str:
        """ RGBA string `rgba(int, int, int, float)` """
        color = self.with_opacity(opacity)
        return f"rgba({color.r_int}, {color.g_int}, {color.b_int}, {color.a})"

    # Operations
    def __mul__(
            self: Color, scalar: float | int
        ) -> Color:
        """ Multiply color rgb(a) components by a scalar """
        if not isinstance(scalar, (float, int)):
            raise TypeError("Invalid type for multiplication.")
        if scalar < 0:
            raise ValueError("Scalar must be a non-negative number.")

        if self.has_alpha:
            return Color(np.clip(self.rgba_array * scalar, 0, 1))
        else:
            return Color(np.clip(self.rgb_array * scalar, 0, 1))

    def __rmul__(
            self: Color, scalar: float | int
        ) -> Color:
        """ Multiply color rgb(a) components by a scalar """
        return self * scalar

    def __truediv__(
            self: Color, scalar: float | int
        ) -> Color:
        """ Divide color rgb(a) components by a scalar """
        return self * (1/scalar)

    # Interpolation
    @staticmethod
    def interpolate(
            color1: Color, color2: Color, t: float = 0.5
        ) -> Color:
        """ Interpolate between two colors """
        if not 0 <= t <= 1:
            raise ValueError(
                "Interpolation factor `t` must be between 0 and 1."
            )

        if not color1.has_alpha == color2.has_alpha:
            raise ValueError(
                "The two colors must either both have an alpha value or not."
            )

        if color1.has_alpha:
            return Color(
                color1.rgba_array + (color2.rgba_array - color1.rgba_array) * t
            )
        else:
            return Color(
                color1.rgb_array + (color2.rgb_array - color1.rgb_array) * t
            )

    def with_opacity(self: Color, opacity: float | None = None) -> Color:
        """
        Return a new `Color` with the same RGB values and the provided opacity.

        Note: If the color already has an alpha value, it will be replaced by
              the provided opacity.
        Note: If the color does not have an alpha value and no opacity is
              provided, a ValueError is raised.
        """
        if not self.has_alpha and opacity is None:
            raise ValueError(
                "Alpha value is `None` and no `opacity` is provided."
            )

        if opacity is None:
            return self

        if not isinstance(opacity, float) or not 0 <= opacity <= 1:
            raise ValueError(
                "Opacity must be a `float` between 0 and 1."
            )

        return Color(self.rgb_tuple + (opacity,))

    def __str__(self: Color) -> str:
        """ String representation of the color """
        if self.has_alpha:
            return f"Color({self.rgba_string()})"
        else:
            return f"Color({self.rgb_string()})"

    def __repr__(self: Color) -> str:
        """ String representation of the color """
        return self.__str__()
