# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.
import numpy as np
from hyperspy.external.matplotlib.collections import RectangleCollection

from hyperspy.docstrings.markers import (OFFSET_DOCSTRING,
                                         UNITS_DOCSTRING,
                                         ANGLE_DOCSTRING)
from hyperspy.drawing._markers._markers_with_widths_heights import _EqualWidthsHeightsMarkers

class Squares(_EqualWidthsHeightsMarkers):
    """
    A Collection of square markers using
    :py:class`matplotlib.collections.RegularPolyCollection`.
    """

    marker_type = "Squares"

    def __init__(
        self,
        offsets,
        sizes,
        offsets_transform="data",
        units="x",
        angles=(0,),
        **kwargs
    ):
        """
        Initialize the set of square Markers.

        Parameters
        ----------
        %s
        sizes : Array-like
            The half of the squares.
        %s
        %s
        kwargs:
            Additional keyword arguments are passed to
            ``matplotlib.collections.RectangleCollection``.
        """
        super().__init__(
            collection_class=RectangleCollection,
            sizes=sizes,
            offsets=offsets,
            angles=angles,
            units=units,
            offsets_transform=offsets_transform,
            **kwargs
        )

    __init__.__doc__ %= (OFFSET_DOCSTRING,
                         UNITS_DOCSTRING,
                         ANGLE_DOCSTRING)
