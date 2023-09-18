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

from hyperspy.docstrings.markers import OFFSET_DOCSTRING
from hyperspy.drawing.markers import Markers
from hyperspy.external.matplotlib.collections import TextCollection


class Texts(Markers):
    """
    A set of text markers
    """

    marker_type = "Texts"

    def __init__(self,
                 offsets,
                 offsets_transform="data",
                 **kwargs
                 ):
        """
        Initialize the set of Circle Markers.

        Parameters
        ----------
        %s
        sizes : array-like
            The size of the text in points.
        facecolors : matplotlib color or list of colors
            Set the facecolor(s) of the markers. It can be a color
            (all patches have same color), or a sequence of colors;
            if it is a sequence the patches will cycle through the sequence.
            If c is 'none', the patch will not be filled.
        kwargs : dict
            Keyword arguments are passed to :py:class:`matplotlib.collections.CircleCollection`.
        """
        if "transform" in kwargs and kwargs["transform"] is not "display":
            raise ValueError(
                "The transform argument is not supported for Texts Markers. Instead, "
                "use the offsets_transform argument to specify the transform of the "
                "offsets the size of the text will be in points and the transform of the "
                "text will be defined by the ``rotation`` ``horizontalalignment`` and "
                "``verticalalignment`` and ``sizes`` arguments."
            )
        kwargs["transform"] = "display"
        super().__init__(
            collection_class=TextCollection,
            offsets=offsets,
            offsets_transform=offsets_transform,
            **kwargs
        )

    __init__.__doc__ %= OFFSET_DOCSTRING
