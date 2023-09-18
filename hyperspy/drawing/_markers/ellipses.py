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

from hyperspy.docstrings.markers import (OFFSET_DOCSTRING,
                                         ANGLE_DOCSTRING,
                                         UNITS_DOCSTRING,
                                         HEIGHTS_DOCSTRING,
                                         WIDTHS_DOCSTRING)

from hyperspy.drawing._markers._markers_with_widths_heights import _WidthsHeightsMarkers
from hyperspy.external.matplotlib.collections import EllipseCollection


class Ellipses(_WidthsHeightsMarkers):
    """A set of Ellipse Markers"""

    marker_type = "Ellipses"

    def __init__(
        self,
        offsets,
        heights,
        widths,
        offsets_transform="data",
        units="xy",
        angles=(0,),
        **kwargs
    ):
        """Initialize the set of Ellipse Markers.

        Parameters
        ----------
        %s
        %s
        %s
        %s
        %s
        kwargs:
            Additional keyword arguments are passed to :py:class:`matplotlib.collections.EllipseCollection`.
        """
        super().__init__(
            collection_class=EllipseCollection,
            offsets=offsets,
            offsets_transform=offsets_transform,
            heights=heights,
            widths=widths,
            angles=angles,
            units=units,
            **kwargs
        )

    __init__.__doc__ %= (OFFSET_DOCSTRING,
                         HEIGHTS_DOCSTRING,
                         WIDTHS_DOCSTRING,
                         ANGLE_DOCSTRING,
                         UNITS_DOCSTRING,)
