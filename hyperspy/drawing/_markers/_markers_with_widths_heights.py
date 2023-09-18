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

from copy import deepcopy
from hyperspy.drawing.markers import Markers
from hyperspy.docstrings.markers import (OFFSET_DOCSTRING,
                                         UNITS_DOCSTRING,
                                         HEIGHTS_DOCSTRING,
                                         WIDTHS_DOCSTRING,
                                         ANGLE_DOCSTRING
                                         )


class _EqualWidthsHeightsMarkers(Markers):
    """A set of markers with equal widths and heights such as
    Circles, Points, and Squares."""

    marker_type = "EqualWidthsHeightsMarkers"

    def __init__(
        self,
        offsets,
        collection_class,
        sizes=2,
        angles=(0,),
        units="xy",
        offsets_transform="data",
        **kwargs
    ):
        """
        Initialize the set of Circle Markers.

        Parameters
        ----------
        %s
        %s
        sizes : array-like
            The size of the circles in points.
        facecolors : matplotlib color or list of colors
            Set the facecolor(s) of the markers. It can be a color
            (all patches have same color), or a sequence of colors;
            if it is a sequence the patches will cycle through the sequence.
            If c is 'none', the patch will not be filled.
        kwargs : dict
            Keyword arguments are passed to :py:class:`matplotlib.collections.CircleCollection`.
        """
        if "transform" in kwargs and kwargs["transform"] != "display":
            raise ValueError(
                f"The transform argument is not supported for {self.marker_type} Markers. Instead, "
                "use the offsets_transform argument to specify the transform of the "
                "offsets and use the ``units`` argument to specify transform of the "
                "sizes."
            )
        kwargs["transform"] = "display"  # This is the only option for this marker type.
        super().__init__(
            collection_class=collection_class,
            offsets=offsets,
            sizes=sizes,
            angles=angles,
            units=units,
            offsets_transform=offsets_transform,
            **kwargs
        )

    def get_data_position(self, get_static_kwargs=True, get_modified=True):
        """ Get the data position of the markers and replace the sizes with widths and heights."""
        kwargs = super().get_data_position(get_static_kwargs=get_static_kwargs)
        if get_modified:
            deepcopy_kwargs = deepcopy(kwargs)
            sizes = deepcopy_kwargs.pop("sizes")
            deepcopy_kwargs["widths"] = sizes
            deepcopy_kwargs["heights"] = sizes
            return deepcopy_kwargs
        else:
            return kwargs

    __init__.__doc__ %= (OFFSET_DOCSTRING,UNITS_DOCSTRING)

class _WidthsHeightsMarkers(Markers):
    marker_type = "WidthsHeightsMarkers"

    def __init__(
        self,
        offsets,
        collection_class,
        widths,
        heights,
        units="xy",
        angles=(0,),
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
        if "transform" in kwargs and kwargs["transform"] != "display":
            raise ValueError(
                f"The transform argument is not supported for {self.marker_type} Markers. Instead, "
                "use the offsets_transform argument to specify the transform of the "
                "offsets and use the ``units`` argument to specify transform of the "
                "sizes."
            )
        kwargs["transform"] = "display"
        super().__init__(
            collection_class=collection_class,
            offsets=offsets,
            widths=widths,
            heights=heights,
            units=units,
            angles=angles,
            offsets_transform=offsets_transform,
            **kwargs
        )
    __init__.__doc__ %= (OFFSET_DOCSTRING,
                         WIDTHS_DOCSTRING,
                         HEIGHTS_DOCSTRING,
                         UNITS_DOCSTRING,
                         ANGLE_DOCSTRING)