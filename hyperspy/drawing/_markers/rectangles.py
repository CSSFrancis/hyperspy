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

from hyperspy.external.matplotlib.collections import RectangleCollection
from hyperspy.drawing._markers._markers_with_widths_heights import _WidthsHeightsMarkers
from hyperspy.docstrings.markers import (OFFSET_DOCSTRING,
                                         UNITS_DOCSTRING,
                                         HEIGHTS_DOCSTRING,
                                         WIDTHS_DOCSTRING,
                                         ANGLE_DOCSTRING)

class Rectangles(_WidthsHeightsMarkers):
    """A Collection of Rectangles Markers"""

    marker_type = "Rectangles"

    def __init__(
        self,
        offsets,
        widths,
        heights,
        angles=(0,),
        units="xy",
        offsets_transform="data",
        **kwargs
    ):
        """Initialize the set of Segments Markers.

        Parameters
        ----------
        %s
        %s
        %s
        %s
        %s
        kwargs:
            Additional keyword arguments are passed to matplotlib.collections.PolyCollection.
        """
        super().__init__(
            collection_class=RectangleCollection,
            offsets=offsets,
            widths=widths,
            heights=heights,
            angles=angles,
            units=units,
            offsets_transform=offsets_transform,
            **kwargs
        )
    __init__.__doc__ %= (OFFSET_DOCSTRING,
                         WIDTHS_DOCSTRING,
                         HEIGHTS_DOCSTRING,
                         UNITS_DOCSTRING,
                         ANGLE_DOCSTRING)


