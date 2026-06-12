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

"""Backend-agnostic marker collection descriptors.

``HyperMarkerCollection`` subclasses describe the *geometry* of a marker set
without coupling to any rendering backend.  Each subclass bundles three
things that were previously scattered across the ``Markers`` subclasses and
the MPL backend:

1. ``_marker_type`` — dispatched to ``PlottingBackend.create_markers`` so
   backends (anyplotlib, fastplotlib, …) can render natively.
2. ``_position_key`` / ``_position_key_to_set`` — the canonical kwarg keys
   for position data, shared by both the storage layer and all backends.
3. ``mpl_collection()`` — a lazy classmethod (defined once on the base
   class) returning the MPL Collection class used on the MPL fallback path,
   resolved from the single type→class map in
   ``hyperspy.drawing.backends.mpl._collections``.

Pass a *subclass* (not an instance) to ``Markers(collection=...)``, exactly
as you would pass a ``matplotlib.collections.Collection`` subclass today.
The looping / navigation behaviour in ``Markers`` is purely data-side and
does not change.
"""

from __future__ import annotations

# Registry mapping _marker_type string → HyperMarkerCollection subclass.
# Populated automatically via __init_subclass__.
_REGISTRY: dict[str, type[HyperMarkerCollection]] = {}


class HyperMarkerCollection:
    """Base class for backend-agnostic marker collection descriptors.

    Each subclass represents one geometry type.  Pass the *subclass* (not an
    instance) as the ``collection`` argument to
    :class:`~hyperspy.drawing.markers.Markers`.

    Subclasses must define:

    ``_marker_type``
        One of the :class:`~hyperspy.drawing.backends._protocol.MarkerType`
        string constants.  Used for native backend dispatch.
    ``_position_key``
        The kwarg key that holds primary positional data (``"offsets"``,
        ``"segments"``, or ``"verts"``).  Defaults to ``"offsets"``.
    ``_position_key_to_set``
        The key used when *updating* an existing collection.  Defaults to
        ``_position_key``; differs for ``VLinesCollection`` and
        ``HLinesCollection`` which accept positions as ``"offsets"`` but
        construct the full ``"segments"`` internally.

    """

    _marker_type: str = ""
    _position_key: str = "offsets"
    _position_key_to_set: str | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Default _position_key_to_set to _position_key when not overridden.
        if cls._position_key_to_set is None:
            cls._position_key_to_set = cls._position_key
        if cls._marker_type:
            _REGISTRY[cls._marker_type] = cls

    @classmethod
    def mpl_collection(cls):
        """Return the MPL Collection class for the fallback rendering path.

        Resolved from ``_marker_type`` via the single type→class map in
        :mod:`hyperspy.drawing.backends.mpl._collections`, keeping the
        descriptors free of matplotlib imports.

        Returns
        -------
        type
            A subclass of :class:`matplotlib.collections.Collection`.
        """
        from hyperspy.drawing.backends.mpl._collections import get_collection_class

        return get_collection_class(cls._marker_type)

    @staticmethod
    def is_hyper_collection(obj) -> bool:
        """Return True when *obj* is a HyperMarkerCollection subclass (not instance)."""
        return isinstance(obj, type) and issubclass(obj, HyperMarkerCollection)

    @staticmethod
    def from_marker_type(marker_type: str) -> type[HyperMarkerCollection]:
        """Return the HyperMarkerCollection subclass registered for *marker_type*.

        Parameters
        ----------
        marker_type : str
            One of the ``MarkerType.*`` string constants (e.g. ``"points"``).

        Raises
        ------
        ValueError
            When *marker_type* does not match any registered subclass.
        """
        try:
            return _REGISTRY[marker_type]
        except KeyError:
            raise ValueError(
                f"Unknown marker type '{marker_type}'. Valid types: {sorted(_REGISTRY)}"
            )


# ── Concrete collection descriptors ──────────────────────────────────────────


class PointsCollection(HyperMarkerCollection):
    """Descriptor for point markers (filled circle glyphs at offsets)."""

    _marker_type = "points"
    _position_key = "offsets"

class CirclesCollection(HyperMarkerCollection):
    """Descriptor for circle markers with explicit radii (data-space sized)."""

    _marker_type = "circles"
    _position_key = "offsets"

class SquaresCollection(HyperMarkerCollection):
    """Descriptor for square markers with explicit widths."""

    _marker_type = "squares"
    _position_key = "offsets"

class LinesCollection(HyperMarkerCollection):
    """Descriptor for arbitrary line segment markers (segments key)."""

    _marker_type = "lines"
    _position_key = "segments"

class VLinesCollection(HyperMarkerCollection):
    """Descriptor for vertical line markers spanning the full axes height.

    Positions are stored as x-values in ``offsets`` and converted to full
    ``[[x,0],[x,1]]`` segments when passed to the collection.
    """

    _marker_type = "vlines"
    _position_key = "offsets"
    _position_key_to_set = "segments"

class HLinesCollection(HyperMarkerCollection):
    """Descriptor for horizontal line markers spanning the full axes width.

    Positions are stored as y-values in ``offsets`` and converted to full
    ``[[0,y],[1,y]]`` segments when passed to the collection.
    """

    _marker_type = "hlines"
    _position_key = "offsets"
    _position_key_to_set = "segments"

class TextsCollection(HyperMarkerCollection):
    """Descriptor for text annotation markers."""

    _marker_type = "texts"
    _position_key = "offsets"

class RectanglesCollection(HyperMarkerCollection):
    """Descriptor for rectangle markers with explicit widths and heights."""

    _marker_type = "rectangles"
    _position_key = "offsets"

class EllipsesCollection(HyperMarkerCollection):
    """Descriptor for ellipse markers with explicit widths, heights, and angles."""

    _marker_type = "ellipses"
    _position_key = "offsets"

class PolygonsCollection(HyperMarkerCollection):
    """Descriptor for polygon markers defined by explicit vertex lists."""

    _marker_type = "polygons"
    _position_key = "verts"

class ArrowsCollection(HyperMarkerCollection):
    """Descriptor for arrow / quiver markers."""

    _marker_type = "arrows"
    _position_key = "offsets"
