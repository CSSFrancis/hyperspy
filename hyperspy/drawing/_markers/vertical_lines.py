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

from hyperspy.drawing.markers import Markers
from matplotlib.collections import LineCollection
import numpy as np


class VerticalLines(Markers):
    """A set of Vertical Line Markers"""

    marker_type = "VerticalLines"

    def __init__(self, offsets, **kwargs):
        """
        Initialize the set of Vertical Line Markers.

        Parameters
        ----------
        offsets: [n]
            Positions of the markers
        kwargs: dict
            Keyword arguments passed to the underlying marker collection. Any argument
            that is array-like and has `dtype=object` is assumed to be an iterating
            argument and is treated as such.
        """
        if "transform" in kwargs and kwargs["transform"] != "xaxis":
            raise ValueError(
                "HorizontalLines markers must have transform='xaxis'."
            )
        if "offsets_transform" in kwargs and kwargs["offsets_transform"] != "display":
            raise ValueError(
                "HorizontalLines markers must have offsets_transform='display'."
            )
        kwargs["transform"] = "xaxis"
        kwargs["offsets_transform"] = "display"
        # Data attributes
        Markers.__init__(
            self,
            offsets=offsets,
            collection_class=LineCollection,
            **kwargs
        )

    def get_data_position(self, get_static_kwargs=True, get_modified=True):
        kwargs = super().get_data_position(get_static_kwargs=get_static_kwargs)
        if get_modified:
            x_pos = kwargs.pop("offsets")
        new_segments = np.array(
            [
                [
                    [x, 0],
                    [x, 1],
                ]
                for x in x_pos
            ]
        )
        kwargs["segments"] = new_segments
        return kwargs
