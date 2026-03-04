# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

    _position_key = "offsets"

    def __init__(self, offsets, offset_transform="data", transform="display",
                 label=None, labels=None, **kwargs):
        """
        Initialize the set of Text Markers.

        Parameters
        ----------
        %s
        label : str or None
            Hover-tooltip shown for every text in this collection.
        labels : list of str or None
            Per-text hover-tooltips.  ``labels[i]`` is shown when hovering
            text *i*.
        kwargs : dict
            Keyword arguments are passed to the text collection.
        """
        super().__init__(
            collection=TextCollection,
            offsets=offsets,
            offset_transform=offset_transform,
            transform=transform,
            label=label,
            labels=labels,
            **kwargs,
        )

    __init__.__doc__ %= OFFSET_DOCSTRING
