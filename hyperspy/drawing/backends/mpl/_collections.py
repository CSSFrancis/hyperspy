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

"""Marker-type → matplotlib ``Collection`` class mapping and legacy helpers.

This is the single source of truth tying the backend-neutral
:class:`~hyperspy.drawing.backends._protocol.MarkerType` strings to concrete
matplotlib classes.  Both consumers live on the MPL side of the backend
boundary:

* ``MplBackend.create_markers`` — the native marker rendering path.
* ``HyperMarkerCollection.mpl_collection()`` — the fallback path used by
  ``Markers._initialize_collection`` (needed e.g. for ``Arrows``, whose
  ``Quiver`` constructor takes positional arguments).

The legacy helpers at the bottom support files saved before
``HyperMarkerCollection`` existed, where markers were stored as matplotlib
class names or carry matplotlib ``Patch`` instances in their kwargs.
"""

from __future__ import annotations


def get_collection_class(marker_type: str):
    """Return the matplotlib ``Collection`` class for *marker_type*.

    Raises
    ------
    ValueError
        When *marker_type* has no matplotlib equivalent.
    """
    from matplotlib.collections import LineCollection, PolyCollection

    from hyperspy.external.matplotlib.collections import (
        CircleCollection,
        EllipseCollection,
        RectangleCollection,
        SquareCollection,
        TextCollection,
    )
    from hyperspy.external.matplotlib.quiver import Quiver

    classes = {
        "points": CircleCollection,
        "circles": CircleCollection,
        "squares": SquareCollection,
        "lines": LineCollection,
        "hlines": LineCollection,
        "vlines": LineCollection,
        "texts": TextCollection,
        "rectangles": RectangleCollection,
        "ellipses": EllipseCollection,
        "polygons": PolyCollection,
        "arrows": Quiver,
    }
    try:
        return classes[marker_type]
    except KeyError:
        raise ValueError(
            f"Marker type {marker_type!r} has no matplotlib collection class."
        )


# ── Legacy (pre-HyperMarkerCollection) support ────────────────────────────


def resolve_collection_string(name: str):
    """Resolve a matplotlib collection class name string to the class.

    Supports loading files saved before ``HyperMarkerCollection``, where the
    collection was stored by class name (e.g. ``"LineCollection"``).

    Raises
    ------
    ValueError
        When *name* is not the name of a matplotlib collection class.
    """
    import matplotlib.collections as mpl_collections

    try:
        return getattr(mpl_collections, name)
    except AttributeError:
        raise ValueError(
            f"'{name}' is not a known marker type or the name "
            "of a matplotlib collection class."
        )


def validate_collection_class(collection):
    """Validate a user-supplied matplotlib ``Collection`` subclass.

    Only classes importable from ``matplotlib.collections`` or
    ``hyperspy.external`` can be reconstructed when loading from file.

    Raises
    ------
    ValueError
        When *collection* is not a safe ``Collection`` subclass.
    """
    import matplotlib.collections as mpl_collections

    if not issubclass(collection, mpl_collections.Collection):
        raise ValueError(
            f"{collection} is not a subclass of `matplotlib.collection.Collection`."
        )

    if ".".join(collection.__module__.split(".")[:2]) not in [
        "matplotlib.collections",
        "hyperspy.external",
    ]:
        # To be able to load a custom markers, we need to be able to
        # instantiate the class and the safe way to do that is to import
        # from `matplotlib.collections` or `hyperspy.external`.
        raise ValueError(
            "To support loading file saved with custom markers, the "
            "collection must be implemented in matplotlib or hyperspy"
        )


def is_patch(obj) -> bool:
    """Return True if obj is a matplotlib Patch (lazy import)."""
    try:
        from matplotlib.patches import Patch

        return isinstance(obj, Patch)
    except ImportError:
        return False
