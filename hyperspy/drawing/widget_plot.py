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

"""
Widget-based drop-in replacements for the matplotlib ImagePlot and
Signal1DFigure drawing classes.

These classes expose the same interface as :class:`~hyperspy.drawing.image.ImagePlot`
and :class:`~hyperspy.drawing.signal1d.Signal1DFigure` (specifically: ``.figure``,
``.events.closed``, ``.close()``, ``.update()``, ``.ax``, ``.ax_markers``,
``.add_marker()``) so that the existing
:class:`~hyperspy.drawing.mpl_hie.MPL_HyperImage_Explorer` and
:class:`~hyperspy.drawing.mpl_hse.MPL_HyperSignal1D_Explorer` infrastructure can
delegate rendering to :class:`~hyperspy.viewer.viewer2d.Viewer2D` /
:class:`~hyperspy.viewer.viewer1d.Viewer1D` without any changes to the rest of
the codebase.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

try:
    from traits.api import Undefined  # type: ignore[attr-defined]
except ImportError:
    Undefined = None  # type: ignore[assignment,misc]

from hyperspy.events import Event, Events

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Marker-to-Viewer translation bridge
# ---------------------------------------------------------------------------

def _to_hex(c) -> str | None:
    """Convert any matplotlib-understood colour spec to a CSS hex string.

    Returns ``None`` if *c* is falsy or represents 'none'/'transparent'.
    """
    if c is None:
        return None
    if isinstance(c, str) and c.lower() in ("none", "transparent", ""):
        return None
    try:
        import matplotlib.colors as _mc
        rgba = _mc.to_rgba(c)
        r, g, b = (int(round(v * 255)) for v in rgba[:3])
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return str(c) if isinstance(c, str) else None


def _color_sequence(kwargs: dict, n: int, key: str = "colors") -> list[str]:
    """Extract a per-item colour list of length *n* from marker kwargs.

    Checks ``key``, then ``colors``, ``color``, ``facecolor``, ``edgecolor``
    in that order.  Each value is passed through :func:`_to_hex`.
    Falls back to ``'#ff4444'``.
    """
    for k in (key, "colors", "color", "facecolor", "edgecolor"):
        val = kwargs.get(k)
        if val is None:
            continue
        if isinstance(val, str):
            h = _to_hex(val)
            if h:
                return [h] * n
            continue
        try:
            lst = list(val)
        except TypeError:
            continue
        if len(lst) == 0:
            continue
        hexes = [_to_hex(v) or "#ff4444" for v in lst]
        return [hexes[i % len(hexes)] for i in range(n)]
    return ["#ff4444"] * n


def _edge_color(kwargs: dict, n: int) -> list[str]:
    """Return edge colours, preferring ``edgecolor`` / ``edgecolors`` over
    the generic ``color`` / ``colors`` keys."""
    for k in ("edgecolor", "edgecolors", "colors", "color"):
        val = kwargs.get(k)
        if val is None:
            continue
        if isinstance(val, str):
            h = _to_hex(val)
            if h:
                return [h] * n
            continue
        try:
            lst = list(val)
        except TypeError:
            continue
        if not lst:
            continue
        hexes = [_to_hex(v) or "#ff4444" for v in lst]
        return [hexes[i % len(hexes)] for i in range(n)]
    return ["#ff4444"] * n


def _fill_color(kwargs: dict) -> str | None:
    """Return a single fill colour, preferring ``facecolor`` / ``facecolors``
    over the generic ``color`` key.  Returns ``None`` when fill is 'none'."""
    for k in ("facecolor", "facecolors", "color"):
        val = kwargs.get(k)
        if val is None:
            continue
        if isinstance(val, str):
            return _to_hex(val)   # None when 'none'/'transparent'
        try:
            lst = list(val)
            if lst:
                return _to_hex(lst[0])
        except TypeError:
            pass
    return None


def _scalar(kwargs: dict, key: str, default):
    """Return a scalar from marker kwargs, broadcasting if needed."""
    val = kwargs.get(key, default)
    if val is None:
        return default
    if hasattr(val, "__iter__") and not isinstance(val, str):
        lst = list(val)
        return lst[0] if lst else default
    return val


def _hs_marker_to_viewer1d(marker, viewer) -> str | None:
    """Translate one HyperSpy ``Markers`` object → a ``Viewer1D`` marker set.

    Returns the marker-set ID string assigned by the viewer, or ``None`` if
    the marker type is not supported by the widget backend.

    Parameters
    ----------
    marker :
        A :class:`~hyperspy.drawing.markers.Markers` instance.
    viewer :
        A :class:`~hyperspy.viewer.viewer1d.Viewer1D` instance.
    """
    from hyperspy.drawing._markers.vertical_lines import VerticalLines
    from hyperspy.drawing._markers.horizontal_lines import HorizontalLines
    from hyperspy.drawing._markers.points import Points
    from hyperspy.drawing._markers.lines import Lines
    from hyperspy.drawing._markers.texts import Texts

    try:
        kwds = marker.get_current_kwargs()
    except Exception:
        _logger.debug("Could not get kwargs for marker %r; skipping.", marker)
        return None

    lw = float(_scalar(kwds, "linewidth", _scalar(kwds, "lw", 1.5)))

    # ── VerticalLines ────────────────────────────────────────────────────────
    if isinstance(marker, VerticalLines):
        offsets = np.asarray(kwds.get("offsets", []), dtype=float).ravel()
        if offsets.size == 0:
            return None
        n = len(offsets)
        colors = _edge_color(kwds, n)
        return viewer.add_vlines(
            offsets=offsets,
            color=colors[0],
            linewidth=lw,
            label=marker.label,
            labels=marker.get_current_labels(),
        )

    # ── HorizontalLines ──────────────────────────────────────────────────────
    if isinstance(marker, HorizontalLines):
        offsets = np.asarray(kwds.get("offsets", []), dtype=float).ravel()
        if offsets.size == 0:
            return None
        n = len(offsets)
        colors = _edge_color(kwds, n)
        return viewer.add_hlines(
            offsets=offsets,
            color=colors[0],
            linewidth=lw,
            label=marker.label,
            labels=marker.get_current_labels(),
        )

    # ── Points ───────────────────────────────────────────────────────────────
    if isinstance(marker, Points):
        offsets = np.asarray(kwds.get("offsets", []), dtype=float)
        if offsets.ndim == 1:
            offsets = offsets[:, np.newaxis]
        if offsets.size == 0:
            return None
        n = len(offsets)
        sizes_raw = kwds.get("sizes", 5)
        if hasattr(sizes_raw, "__iter__") and not isinstance(sizes_raw, str):
            sizes = np.asarray(list(sizes_raw), dtype=float)
            sizes = np.array([sizes[i % len(sizes)] for i in range(n)])
        else:
            sizes = np.full(n, float(sizes_raw))
        radius = np.sqrt(sizes / np.pi)
        ec = _edge_color(kwds, n)
        fc = _fill_color(kwds)
        return viewer.add_points(
            offsets=offsets,
            sizes=radius,
            color=ec[0],
            linewidth=lw,
            fill_color=fc,
            fill_alpha=0.3,
            label=marker.label,
            labels=marker.get_current_labels(),
        )

    # ── Lines (segments) ─────────────────────────────────────────────────────
    if isinstance(marker, Lines):
        segs = kwds.get("segments", [])
        if len(segs) == 0:
            return None
        segs = np.asarray(segs, dtype=float)
        if segs.ndim == 2 and segs.shape == (2, 2):
            segs = segs[np.newaxis]
        if segs.ndim != 3 or segs.shape[1:] != (2, 2):
            return None
        n = len(segs)
        colors = _edge_color(kwds, n)
        return viewer.add_lines(
            segments=segs,
            color=colors[0],
            linewidth=lw,
            label=marker.label,
            labels=marker.get_current_labels(),
        )

    # ── Texts ────────────────────────────────────────────────────────────────
    if isinstance(marker, Texts):
        offsets = np.asarray(kwds.get("offsets", []), dtype=float)
        if offsets.ndim == 1:
            offsets = offsets[:, np.newaxis]
        if offsets.size == 0:
            return None
        texts_raw = kwds.get("texts", kwds.get("strings", []))
        texts = [str(t) for t in texts_raw]
        n = len(offsets)
        if len(texts) != n:
            return None
        colors = _edge_color(kwds, n)
        fontsize = int(_scalar(kwds, "sizes", _scalar(kwds, "fontsize", 12)))
        fontsize = max(8, min(fontsize, 24))
        return viewer.add_texts(
            offsets=offsets,
            texts=texts,
            color=colors[0],
            fontsize=fontsize,
            label=marker.label,
            labels=marker.get_current_labels(),
        )

    _logger.debug(
        "Marker type %r not supported in widget backend; skipping.",
        type(marker).__name__,
    )
    return None


def _hs_marker_to_viewer2d(marker, viewer) -> str | None:
    """Translate one HyperSpy ``Markers`` object → a ``Viewer2D`` marker set.

    Returns the marker-set ID string, or ``None`` if unsupported.
    """
    from hyperspy.drawing._markers.points import Points
    from hyperspy.drawing._markers.circles import Circles
    from hyperspy.drawing._markers.lines import Lines
    from hyperspy.drawing._markers.texts import Texts
    from hyperspy.drawing._markers.rectangles import Rectangles
    from hyperspy.drawing._markers.squares import Squares
    from hyperspy.drawing._markers.ellipses import Ellipses
    from hyperspy.drawing._markers.arrows import Arrows

    try:
        kwds = marker.get_current_kwargs()
    except Exception:
        _logger.debug("Could not get kwargs for marker %r; skipping.", marker)
        return None

    lw = float(_scalar(kwds, "linewidth", _scalar(kwds, "lw", 1.5)))

    # ── Points / Circles ─────────────────────────────────────────────────────
    if isinstance(marker, (Points, Circles)):
        offsets = np.asarray(kwds.get("offsets", []), dtype=float)
        if offsets.ndim == 1 and offsets.size == 2:
            offsets = offsets[np.newaxis]
        if offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.size == 0:
            return None
        n = len(offsets)
        sizes_raw = kwds.get("sizes", 5)
        if hasattr(sizes_raw, "__iter__") and not isinstance(sizes_raw, str):
            sizes = np.asarray(list(sizes_raw), dtype=float)
            sizes = np.array([sizes[i % len(sizes)] for i in range(n)])
        else:
            sizes = np.full(n, float(sizes_raw))
        radius = np.sqrt(sizes / np.pi)
        ec = _edge_color(kwds, n)
        fc = _fill_color(kwds)
        return viewer.add_circles(
            offsets=offsets, sizes=radius,
            color=ec[0], linewidth=lw,
            fill_color=fc, fill_alpha=0.3,
            label=marker.label, labels=marker.get_current_labels(),
        )

    # ── Lines ────────────────────────────────────────────────────────────────
    if isinstance(marker, Lines):
        segs = kwds.get("segments", [])
        if len(segs) == 0:
            return None
        segs = np.asarray(segs, dtype=float)
        if segs.ndim == 2 and segs.shape == (2, 2):
            segs = segs[np.newaxis]
        if segs.ndim != 3 or segs.shape[1:] != (2, 2):
            return None
        n = len(segs)
        ec = _edge_color(kwds, n)
        return viewer.add_lines(
            segments=segs,
            color=ec[0], linewidth=lw,
            label=marker.label, labels=marker.get_current_labels(),
        )

    # ── Rectangles ───────────────────────────────────────────────────────────
    if isinstance(marker, Rectangles):
        offsets = np.asarray(kwds.get("offsets", []), dtype=float)
        if offsets.ndim == 1 and offsets.size == 2:
            offsets = offsets[np.newaxis]
        if offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.size == 0:
            return None
        n = len(offsets)
        ec = _edge_color(kwds, n)
        fc = _fill_color(kwds)
        return viewer.add_rectangles(
            offsets=offsets,
            widths=kwds.get("widths", np.ones(n)),
            heights=kwds.get("heights", np.ones(n)),
            angles=kwds.get("angles", np.zeros(n)),
            color=ec[0], linewidth=lw,
            fill_color=fc, fill_alpha=0.3,
            label=marker.label, labels=marker.get_current_labels(),
        )

    # ── Squares ──────────────────────────────────────────────────────────────
    if isinstance(marker, Squares):
        offsets = np.asarray(kwds.get("offsets", []), dtype=float)
        if offsets.ndim == 1 and offsets.size == 2:
            offsets = offsets[np.newaxis]
        if offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.size == 0:
            return None
        n = len(offsets)
        ec = _edge_color(kwds, n)
        fc = _fill_color(kwds)
        return viewer.add_squares(
            offsets=offsets,
            widths=kwds.get("widths", np.ones(n)),
            angles=kwds.get("angles", np.zeros(n)),
            color=ec[0], linewidth=lw,
            fill_color=fc, fill_alpha=0.3,
            label=marker.label, labels=marker.get_current_labels(),
        )

    # ── Ellipses ─────────────────────────────────────────────────────────────
    if isinstance(marker, Ellipses):
        offsets = np.asarray(kwds.get("offsets", []), dtype=float)
        if offsets.ndim == 1 and offsets.size == 2:
            offsets = offsets[np.newaxis]
        if offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.size == 0:
            return None
        n = len(offsets)
        ec = _edge_color(kwds, n)
        fc = _fill_color(kwds)
        return viewer.add_ellipses(
            offsets=offsets,
            widths=kwds.get("widths", np.ones(n)),
            heights=kwds.get("heights", np.ones(n)),
            angles=kwds.get("angles", np.zeros(n)),
            color=ec[0], linewidth=lw,
            fill_color=fc, fill_alpha=0.3,
            label=marker.label, labels=marker.get_current_labels(),
        )

    # ── Arrows ───────────────────────────────────────────────────────────────
    if isinstance(marker, Arrows):
        offsets = np.asarray(kwds.get("offsets", []), dtype=float)
        if offsets.ndim == 1 and offsets.size == 2:
            offsets = offsets[np.newaxis]
        if offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.size == 0:
            return None
        n = len(offsets)
        ec = _edge_color(kwds, n)
        return viewer.add_arrows(
            offsets=offsets,
            U=kwds.get("U", np.zeros(n)),
            V=kwds.get("V", np.zeros(n)),
            color=ec[0], linewidth=lw,
            label=marker.label, labels=marker.get_current_labels(),
        )

    # ── Texts ────────────────────────────────────────────────────────────────
    if isinstance(marker, Texts):
        offsets = np.asarray(kwds.get("offsets", []), dtype=float)
        if offsets.ndim == 1 and offsets.size == 2:
            offsets = offsets[np.newaxis]
        if offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.size == 0:
            return None
        texts_raw = kwds.get("texts", kwds.get("strings", []))
        texts = [str(t) for t in texts_raw]
        n = len(offsets)
        if len(texts) != n:
            return None
        ec = _edge_color(kwds, n)
        fontsize = int(_scalar(kwds, "sizes", _scalar(kwds, "fontsize", 12)))
        fontsize = max(8, min(fontsize, 24))
        return viewer.add_texts(
            offsets=offsets, texts=texts,
            color=ec[0], fontsize=fontsize,
            label=marker.label, labels=marker.get_current_labels(),
        )

    _logger.debug(
        "Marker type %r not supported in Viewer2D widget backend; skipping.",
        type(marker).__name__,
    )
    return None


class _HsMarkerBridge:
    """Manages the lifecycle of one HyperSpy ``Markers`` object on a viewer widget.

    The bridge:
    * translates the marker to a viewer marker-set on construction via a
      supplied ``translate_fn``
    * re-pushes the marker-set on every navigation index change (for
      iterating/variable-length markers)
    * removes the viewer marker-set when :meth:`close` is called

    Parameters
    ----------
    marker :
        A :class:`~hyperspy.drawing.markers.Markers` instance.
    viewer :
        A :class:`~hyperspy.viewer.viewer1d.Viewer1D` or
        :class:`~hyperspy.viewer.viewer2d.Viewer2D` instance.
    axes_manager :
        The signal's :class:`~hyperspy.axes.AxesManager` (used to wire up
        the navigation-index-changed event for iterating markers).
    translate_fn :
        Callable ``(marker, viewer) → str | None``.  Defaults to
        :func:`_hs_marker_to_viewer1d`.
    """

    def __init__(self, marker, viewer, axes_manager=None,
                 translate_fn=None) -> None:
        self._marker = marker
        self._viewer = viewer
        self._axes_manager = axes_manager
        self._translate_fn = translate_fn or _hs_marker_to_viewer1d
        self._marker_id: str | None = None

        self._push()

        # For iterating markers, re-push on every navigation step.
        # Also re-push when labels is a dynamic object array even if the
        # marker positions themselves are static.
        _labels_dynamic = (
            isinstance(marker.labels, np.ndarray) and marker.labels.dtype == object
        )
        if (marker._is_iterating or _labels_dynamic) and axes_manager is not None:
            axes_manager.events.indices_changed.connect(self._on_nav_changed, [])

    # ------------------------------------------------------------------
    def _push(self) -> None:
        """Translate + (re-)push the marker to the viewer."""
        mid = self._translate_fn(self._marker, self._viewer)
        if mid is None:
            return
        if self._marker_id is not None:
            try:
                self._viewer.remove_marker(self._marker_id)
            except (KeyError, AttributeError):
                pass
        self._marker_id = mid

    def _on_nav_changed(self) -> None:
        """Called when navigation indices change → refresh iterating markers."""
        if self._viewer is None:
            return
        if self._marker_id is not None:
            try:
                self._viewer.remove_marker(self._marker_id)
            except (KeyError, AttributeError):
                pass
            self._marker_id = None
        self._push()

    def close(self) -> None:
        """Remove the viewer marker-set and disconnect event listeners."""
        if self._axes_manager is not None:
            try:
                self._axes_manager.events.indices_changed.disconnect(
                    self._on_nav_changed
                )
            except Exception:
                pass
        if self._viewer is not None and self._marker_id is not None:
            try:
                self._viewer.remove_marker(self._marker_id)
            except (KeyError, AttributeError):
                pass
        self._viewer = None
        self._marker_id = None


# ---------------------------------------------------------------------------
# Sentinel object used as the "figure" attribute so that is_active works
# ---------------------------------------------------------------------------

class _SentinelFigure:
    """Stands in for a matplotlib Figure so that ``plot.figure is not None``
    evaluates to ``True`` while the viewer is open.

    It also holds ``canvas`` with a no-op ``mpl_connect`` so that the
    explorers that try to bind key-press events don't crash.
    """

    class _Canvas:
        @staticmethod
        def mpl_connect(*args, **kwargs) -> None:
            """No-op — key events are handled inside the JS widget."""
            return None

        @staticmethod
        def draw_idle() -> None:
            pass

        supports_blit = False

    canvas = _Canvas()

    def colorbar(self, *args, **kwargs):
        """No-op — colorbars are not supported in the widget backend."""
        return None


class _SentinelTransform:
    """Minimal stand-in for a matplotlib ``Transform``.

    Returned by :class:`_WidgetAxSentinel` for ``transData``, ``transAxes``,
    ``get_xaxis_transform()`` and ``get_yaxis_transform()`` so that
    :meth:`~hyperspy.drawing.markers.Markers._get_transform` and
    :meth:`~hyperspy.drawing.markers.Markers._initialize_collection` do not
    raise when the widget backend is active.

    All transformation methods are identity operations — the transform is
    never actually applied because :meth:`_WidgetAxSentinel.add_collection`
    is a no-op.
    """

    def transform(self, values):
        return values

    def transform_affine(self, values):
        return values

    def transform_non_affine(self, values):
        return values

    def inverted(self):
        return self

    # matplotlib checks ``transform.is_bbox`` on some code-paths
    is_bbox = False
    is_affine = True

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self


class _WidgetAxSentinel:
    """Minimal stand-in for a matplotlib ``Axes`` object.

    Set as ``marker.ax`` so that :meth:`~hyperspy.drawing.markers.Markers.plot`
    passes its ``self.ax is None`` guard without attempting any real matplotlib
    drawing.  All methods that would draw to a canvas are no-ops.

    All transform attributes and methods (``transData``, ``transAxes``,
    ``get_xaxis_transform``, ``get_yaxis_transform``) return a
    :class:`_SentinelTransform` so that
    :meth:`~hyperspy.drawing.markers.Markers._get_transform` and
    :meth:`~hyperspy.drawing.markers.Markers._initialize_collection` succeed
    without raising ``AttributeError``.
    """

    def __init__(self, plot_obj) -> None:
        self.figure = _SentinelFigure()
        # hspy_fig is accessed by Markers._render_figure()
        self.hspy_fig = plot_obj

        # Transform sentinels — same keys as Markers._get_transform
        _t = _SentinelTransform()
        self.transData = _t
        self.transAxes = _t

    def get_xaxis_transform(self, which="grid"):
        return _SentinelTransform()

    def get_yaxis_transform(self, which="grid"):
        return _SentinelTransform()

    def add_collection(self, *args, **kwargs) -> None:
        """No-op — markers are rendered by the widget backend."""
        pass

    def draw_artist(self, *args, **kwargs) -> None:
        pass


# ---------------------------------------------------------------------------
# Base class shared by WidgetImagePlot and WidgetSignal1DFigure
# ---------------------------------------------------------------------------

class _WidgetPlotBase:
    """Common interface plumbing for widget-backed plot objects.

    Sub-classes must implement:
    * :meth:`update` — refresh the viewer from the current data.

    Sub-classes may override:
    * :attr:`_translate_fn` — callable ``(marker, viewer) → str | None``
      used by :meth:`add_marker` to push HyperSpy ``Markers`` objects to the
      viewer widget.  Defaults to :func:`_hs_marker_to_viewer1d`.
    """

    # Override in subclasses that use a Viewer2D
    _translate_fn = staticmethod(_hs_marker_to_viewer1d)

    def __init__(self, title: str = "") -> None:
        self._title = title
        self.ax = None          # Not meaningful for widget backend, kept for compat
        self.ax_markers: list = []
        self._marker_bridges: list = []   # _HsMarkerBridge instances
        self.axes_manager = None

        # Use a sentinel so .figure is not None while open
        self.figure: _SentinelFigure | None = _SentinelFigure()

        # Events — must have `events.closed`
        self.events = Events()
        self.events.closed = Event(
            """
            Event that triggers when the viewer is closed.

            Parameters
            ----------
            obj : plot instance
                The instance that triggered the event.
            """,
            arguments=["obj"],
        )

        # The underlying anywidget viewer (set by subclasses)
        self.viewer = None
        # When False, _display() is a no-op; Widget_HyperExplorer composes all
        # viewers together and displays them at once via _display_viewers().
        self._auto_display: bool = True

    # ------------------------------------------------------------------
    # Title property (same as BlittedFigure)
    # ------------------------------------------------------------------
    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        self._title = str(value)

    # ------------------------------------------------------------------
    # Marker support
    # ------------------------------------------------------------------
    def add_marker(self, marker) -> None:
        """Translate a HyperSpy ``Markers`` object and add it to the viewer.

        The marker is translated into the native viewer marker API.  For
        *iterating* markers the bridge re-pushes the translated set whenever
        navigation indices change.

        A :class:`_WidgetAxSentinel` is set as ``marker.ax`` so that the
        subsequent ``marker.plot()`` call (issued by
        :meth:`~hyperspy.signal.BaseSignal.add_marker`) does not raise and
        silently absorbs the ``add_collection`` call.

        Parameters
        ----------
        marker :
            A :class:`~hyperspy.drawing.markers.Markers` instance.
        """
        # Give the marker a sentinel axes object so marker.plot() succeeds
        # without attempting any real matplotlib drawing.
        marker.ax = _WidgetAxSentinel(self)
        self.ax_markers.append(marker)
        if self.viewer is None:
            # Viewer not yet built; stored for deferred push after plot()
            return
        bridge = _HsMarkerBridge(marker, self.viewer, self.axes_manager,
                                  translate_fn=self._translate_fn)
        self._marker_bridges.append(bridge)
        marker.events.closed.connect(
            lambda obj, b=bridge: self._on_marker_closed(b), []
        )

    def _on_marker_closed(self, bridge: "_HsMarkerBridge") -> None:
        bridge.close()
        if bridge in self._marker_bridges:
            self._marker_bridges.remove(bridge)

    def _push_deferred_markers(self) -> None:
        """Called after ``self.viewer`` is set to push any markers added
        before the viewer was built (e.g. permanent markers plotted via
        ``_plot_permanent_markers`` before ``show()`` completes)."""
        if self.viewer is None:
            return
        for marker in list(self.ax_markers):
            bridge = _HsMarkerBridge(marker, self.viewer, self.axes_manager,
                                      translate_fn=self._translate_fn)
            self._marker_bridges.append(bridge)
            marker.events.closed.connect(
                lambda obj, b=bridge: self._on_marker_closed(b), []
            )

    def remove_markers(self, render_figure: bool = False) -> None:
        """Remove all markers from the viewer."""
        for bridge in list(self._marker_bridges):
            bridge.close()
        self._marker_bridges.clear()
        self.ax_markers.clear()

    # ------------------------------------------------------------------
    # render_figure — no-op (widget pushes state automatically via traitlets)
    # ------------------------------------------------------------------
    def render_figure(self) -> None:
        """No-op — kept for API compatibility with ``BlittedFigure``."""
        pass

    # ------------------------------------------------------------------
    # Close / lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Tear down all marker bridges, mark the plot as closed, and fire
        the ``events.closed`` event."""
        _logger.debug("Closing widget plot %r.", self)
        for bridge in list(self._marker_bridges):
            bridge.close()
        self._marker_bridges.clear()
        self.figure = None
        self.events.closed.trigger(obj=self)
        for f in list(self.events.closed.connected):
            self.events.closed.disconnect(f)
        self.viewer = None

    # ------------------------------------------------------------------
    # get_mpl_figure — needed by some internals
    # ------------------------------------------------------------------
    def get_mpl_figure(self):
        return None



# ---------------------------------------------------------------------------
# WidgetImagePlot  (drop-in for ImagePlot)
# ---------------------------------------------------------------------------

class WidgetImagePlot(_WidgetPlotBase):
    """Drop-in replacement for :class:`~hyperspy.drawing.image.ImagePlot`
    that renders via :class:`~hyperspy.viewer.viewer2d.Viewer2D`.

    Only the attributes and methods actually used by
    :class:`~hyperspy.drawing.mpl_hie.MPL_HyperImage_Explorer` and
    :meth:`~hyperspy.signal.BaseSignal.plot` are implemented.
    """

    # Use the 2-D translation function for all markers added to this plot
    _translate_fn = staticmethod(_hs_marker_to_viewer2d)

    def __init__(self, title: str = "", **kwargs) -> None:
        super().__init__(title=title)

        # Attributes mirroring ImagePlot so that mpl_hie can ``setattr``
        # them from kwargs
        self.data_function: Callable | None = None
        self.data_function_kwargs: dict = {}
        self.autoscale: str = "v"
        self.norm: str = "auto"
        self.vmin = None
        self.vmax = None
        self.gamma: float = 1.0
        self.linthresh: float = 0.01
        self.linscale: float = 0.1
        self.scalebar: bool = True
        self.scalebar_color: str = "white"
        self.axes_ticks = None
        self.axes_off: bool = False
        self.no_nans: bool = False
        self.colorbar: bool = True
        self.centre_colormap: str = "auto"
        self.min_aspect: float = 0.1
        self.pixel_units = None
        self.quantity_label: str = ""
        self.plot_indices: bool = True
        self.xaxis = None
        self.yaxis = None
        self._current_data: np.ndarray | None = None

    # ------------------------------------------------------------------
    def plot(self, data_function_kwargs: dict | None = None, **kwargs) -> None:
        """Fetch data and build the :class:`~hyperspy.viewer.viewer2d.Viewer2D`."""
        if data_function_kwargs is None:
            data_function_kwargs = {}
        self.data_function_kwargs = data_function_kwargs

        data = self.data_function(
            axes_manager=self.axes_manager, **self.data_function_kwargs
        )
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            _logger.warning(
                "WidgetImagePlot received %d-D data; expected 2-D.", data.ndim
            )
            return

        x_axis, y_axis, units = self._axes_arrays()

        from hyperspy.viewer.viewer2d import Viewer2D

        self.viewer = Viewer2D(data, x_axis=x_axis, y_axis=y_axis, units=units)
        self._current_data = data

        # Push any markers that were added before the viewer was built
        self._push_deferred_markers()

        # Display the widget
        self._display()

    # ------------------------------------------------------------------
    def update(
        self,
        data_changed: bool = True,
        auto_contrast=None,
        vmin=None,
        vmax=None,
        **kwargs,
    ) -> None:
        """Refresh the viewer with the current navigation position."""
        if self.viewer is None or self.data_function is None:
            return

        data = self.data_function(
            axes_manager=self.axes_manager, **self.data_function_kwargs
        )
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            return

        x_axis, y_axis, units = self._axes_arrays()
        self._current_data = data
        self.viewer.update(data, x_axis=x_axis, y_axis=y_axis, units=units)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _axes_arrays(self):
        """Return ``(x_axis, y_axis, units)`` from the HyperSpy axes objects."""
        x_axis = y_axis = None
        units = "px"

        if self.xaxis is not None:
            x_axis = np.asarray(self.xaxis.axis, dtype=float)
            if self.xaxis.units is not Undefined and self.xaxis.units:
                units = str(self.xaxis.units)

        if self.yaxis is not None:
            y_axis = np.asarray(self.yaxis.axis, dtype=float)
            if self.yaxis.units is not Undefined and self.yaxis.units:
                # prefer x units; y units override only if x had none
                if units == "px":
                    units = str(self.yaxis.units)

        return x_axis, y_axis, units

    def _display(self) -> None:
        """Push the viewer to the cell output (no-op when ``_auto_display`` is False)."""
        if self.viewer is None or not self._auto_display:
            return
        try:
            from IPython.display import display
            display(self.viewer)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WidgetNavigatorPlot1D
# ---------------------------------------------------------------------------

class WidgetNavigatorPlot1D(_WidgetPlotBase):
    """Widget-backed 1-D navigator panel.

    Renders via :class:`~hyperspy.viewer.viewer1d.Viewer1D` and adds a
    draggable vertical-line pointer widget.  When the user drags the pointer
    the connected navigation axis index is updated automatically, which
    triggers the signal plot to refresh.

    Parameters
    ----------
    title : str, optional
        Title string (kept for API parity).
    """

    def __init__(self, title: str = "") -> None:
        super().__init__(title=title)
        # Signal1DFigure-compatible attributes
        self.ax_lines: list = []
        self.right_ax_lines: list = []
        self.right_axes_manager = None
        self.xlabel: str = ""
        self.ylabel: str = ""
        self.axis = None  # the navigation DataAxis

        self._pointer_wid: str | None = None
        # Guards against recursive index ↔ widget updates
        self._updating_index: bool = False
        self._updating_widget: bool = False

    # ------------------------------------------------------------------
    # Signal1DFigure-compatible stubs
    # ------------------------------------------------------------------
    def create_axis(self) -> None:
        """No-op — axis lives inside the Viewer1D widget."""
        pass

    def create_right_axis(self, color: str = "black", adjust_layout: bool = True) -> None:
        """No-op."""
        pass

    def add_line(self, line, ax: str = "left", connect_navigation: bool = False) -> None:
        """Register a :class:`~hyperspy.drawing.signal1d.Signal1DLine`."""
        if ax == "left":
            line.ax = self.ax
            if line.axes_manager is None:
                line.axes_manager = self.axes_manager
            self.ax_lines.append(line)
            line.sf_lines = self.ax_lines
        else:
            self.right_ax_lines.append(line)
            line.sf_lines = self.right_ax_lines
        line.axis = self.axis

    # ------------------------------------------------------------------
    def plot(self, data_function_kwargs: dict | None = None, **kwargs) -> None:
        """Fetch navigator data, build Viewer1D, and add the pointer widget."""
        if data_function_kwargs is None:
            data_function_kwargs = {}

        if not self.ax_lines:
            _logger.warning("WidgetNavigatorPlot1D.plot: no data lines registered.")
            return

        line = self.ax_lines[0]
        dkw: dict = getattr(line, "data_function_kwargs", {})
        raw = line.data_function(axes_manager=self.axes_manager, **dkw)
        data = np.asarray(raw, dtype=float).ravel()

        x_axis: np.ndarray | None = None
        units = "px"
        if self.axis is not None:
            x_axis = np.asarray(self.axis.axis, dtype=float)
            if self.axis.units is not Undefined and self.axis.units:
                units = str(self.axis.units)

        from hyperspy.viewer.viewer1d import Viewer1D
        self.viewer = Viewer1D(data, x_axis=x_axis, units=units,
                               y_units=self.ylabel, color="#4fc3f7")

        # ── pointer at the current index ──────────────────────────────
        if self.axis is not None and x_axis is not None:
            cur_x = float(x_axis[self.axis.index])
        elif x_axis is not None:
            cur_x = float(np.median(x_axis))
        else:
            cur_x = 0.0

        self._pointer_wid = self.viewer.add_vline_widget(x=cur_x, color="#ff4444")

        # Drag → update navigation index
        self.viewer.observe(self._on_widget_changed, names=["overlay_widgets_json"])

        # Index change → move the pointer
        if self.axis is not None:
            self.axis.events.index_changed.connect(self._on_index_changed, [])
            self.events.closed.connect(
                lambda: self.axis.events.index_changed.disconnect(
                    self._on_index_changed
                ),
                [],
            )

        self._display()

    def update(self) -> None:
        """No-op — navigator data is static; pointer drags drive signal updates."""
        pass

    # ------------------------------------------------------------------
    def _on_widget_changed(self, change: dict) -> None:
        """Called when the user drags the vline widget → set navigation index."""
        if self._updating_widget or self.viewer is None or self.axis is None:
            return
        try:
            state = self.viewer.get_widget(self._pointer_wid)
        except (KeyError, TypeError):
            return

        import json as _json
        x_val = float(state.get("x", 0.0))
        x_axis = np.array(_json.loads(self.viewer.x_axis_json))
        if len(x_axis) < 1:
            return
        idx = int(np.argmin(np.abs(x_axis - x_val)))
        if idx != self.axis.index:
            self._updating_index = True
            try:
                self.axis.index = idx
            finally:
                self._updating_index = False

    def _on_index_changed(self) -> None:
        """Called when the navigation index changes → move the vline widget."""
        if self._updating_index or self.viewer is None or self.axis is None:
            return
        import json as _json
        x_axis = np.array(_json.loads(self.viewer.x_axis_json))
        if len(x_axis) < 1:
            return
        idx = min(self.axis.index, len(x_axis) - 1)
        new_x = float(x_axis[idx])
        self._updating_widget = True
        try:
            self.viewer.set_vline_x(self._pointer_wid, new_x)
        except (KeyError, AttributeError):
            pass
        finally:
            self._updating_widget = False

    def _display(self) -> None:
        """Display the viewer widget (no-op when ``_auto_display`` is False)."""
        if self.viewer is None or not self._auto_display:
            return
        try:
            from IPython.display import display
            display(self.viewer)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WidgetNavigatorPlot2D  (drop-in navigator for a 2-D navigation image)
# ---------------------------------------------------------------------------

class WidgetNavigatorPlot2D(_WidgetPlotBase):
    """Widget-backed 2-D navigator panel.

    Renders via :class:`~hyperspy.viewer.viewer2d.Viewer2D` and adds a
    draggable crosshair pointer widget.  When the user drags the crosshair the
    connected navigation axes indices are updated, triggering the signal plot
    to refresh.

    Parameters
    ----------
    title : str, optional
        Title string (kept for API parity).
    """

    def __init__(self, title: str = "") -> None:
        super().__init__(title=title)
        self.data_function: Callable | None = None
        self.data_function_kwargs: dict = {}
        self.xaxis = None  # horizontal navigation DataAxis
        self.yaxis = None  # vertical   navigation DataAxis

        self._pointer_wid: str | None = None
        self._updating_index: bool = False
        self._updating_widget: bool = False

    # ------------------------------------------------------------------
    def plot(self, data_function_kwargs: dict | None = None, **kwargs) -> None:
        """Fetch navigator image, build Viewer2D, and add the crosshair."""
        if data_function_kwargs is None:
            data_function_kwargs = {}
        self.data_function_kwargs = data_function_kwargs

        if self.data_function is None:
            _logger.warning("WidgetNavigatorPlot2D.plot: no data function.")
            return

        raw = self.data_function(
            axes_manager=self.axes_manager, **self.data_function_kwargs
        )
        data = np.asarray(raw, dtype=float)
        if data.ndim != 2:
            _logger.warning(
                "WidgetNavigatorPlot2D received %d-D data; expected 2-D.", data.ndim
            )
            return

        x_axis, y_axis, units = self._axes_arrays()

        from hyperspy.viewer.viewer2d import Viewer2D
        self.viewer = Viewer2D(data, x_axis=x_axis, y_axis=y_axis, units=units)

        # ── crosshair at the current index position ───────────────────
        cx = self._axis_val(self.xaxis, x_axis)
        cy = self._axis_val(self.yaxis, y_axis)
        self._pointer_wid = self.viewer.add_crosshair_widget(
            cx=cx, cy=cy, color="#ff4444"
        )

        # Drag → update navigation indices
        self.viewer.observe(self._on_widget_changed, names=["overlay_widgets"])

        # Index change → move the crosshair
        if self.axes_manager is not None:
            for nav_axis in self.axes_manager.navigation_axes[:2]:
                nav_axis.events.index_changed.connect(self._on_index_changed, [])
                self.events.closed.connect(
                    lambda a=nav_axis: a.events.index_changed.disconnect(
                        self._on_index_changed
                    ),
                    [],
                )

        self._display()

    def update(
        self,
        data_changed: bool = True,
        auto_contrast: object = None,
        vmin: object = None,
        vmax: object = None,
        **kwargs: object,
    ) -> None:
        """Refresh the navigator image (e.g. when a higher-dim nav index changes)."""
        if self.viewer is None or self.data_function is None:
            return
        raw = self.data_function(
            axes_manager=self.axes_manager, **self.data_function_kwargs
        )
        data = np.asarray(raw, dtype=float)
        if data.ndim != 2:
            return
        x_axis, y_axis, units = self._axes_arrays()
        self.viewer.update(data, x_axis=x_axis, y_axis=y_axis, units=units)

    # ------------------------------------------------------------------
    def _axes_arrays(self) -> tuple:
        """Return ``(x_axis, y_axis, units)`` from the navigator axis objects."""
        x_axis = y_axis = None
        units = "px"
        if self.xaxis is not None:
            x_axis = np.asarray(self.xaxis.axis, dtype=float)
            if self.xaxis.units is not Undefined and self.xaxis.units:
                units = str(self.xaxis.units)
        if self.yaxis is not None:
            y_axis = np.asarray(self.yaxis.axis, dtype=float)
            if self.yaxis.units is not Undefined and self.yaxis.units:
                if units == "px":
                    units = str(self.yaxis.units)
        return x_axis, y_axis, units

    @staticmethod
    def _axis_val(ax_obj: object, ax_arr: np.ndarray | None) -> float:
        """Return the physical value for the current index of a DataAxis."""
        if ax_obj is not None and ax_arr is not None and len(ax_arr) > 0:
            idx = min(ax_obj.index, len(ax_arr) - 1)
            return float(ax_arr[idx])
        if ax_arr is not None and len(ax_arr) > 0:
            return float(ax_arr[len(ax_arr) // 2])
        return 0.0

    # ------------------------------------------------------------------
    def _on_widget_changed(self, change: dict) -> None:
        """Called when the user drags the crosshair → set navigation indices."""
        if self._updating_widget or self.viewer is None:
            return
        try:
            state = self.viewer.get_widget(self._pointer_wid)
        except (KeyError, TypeError):
            return

        import json as _json
        cx = float(state.get("cx", 0.0))
        cy = float(state.get("cy", 0.0))
        x_arr = np.array(_json.loads(self.viewer.x_axis_json))
        y_arr = np.array(_json.loads(self.viewer.y_axis_json))

        self._updating_index = True
        try:
            if self.xaxis is not None and len(x_arr) > 0:
                ix = int(np.argmin(np.abs(x_arr - cx)))
                if ix != self.xaxis.index:
                    self.xaxis.index = ix
            if self.yaxis is not None and len(y_arr) > 0:
                iy = int(np.argmin(np.abs(y_arr - cy)))
                if iy != self.yaxis.index:
                    self.yaxis.index = iy
        finally:
            self._updating_index = False

    def _on_index_changed(self) -> None:
        """Called when any navigation index changes → move the crosshair."""
        if self._updating_index or self.viewer is None:
            return
        import json as _json
        x_arr = np.array(_json.loads(self.viewer.x_axis_json))
        y_arr = np.array(_json.loads(self.viewer.y_axis_json))
        cx = self._axis_val(self.xaxis, x_arr if len(x_arr) > 0 else None)
        cy = self._axis_val(self.yaxis, y_arr if len(y_arr) > 0 else None)
        self._updating_widget = True
        try:
            widgets = _json.loads(self.viewer.overlay_widgets)
            for w in widgets:
                if w.get("id") == self._pointer_wid:
                    w["cx"] = cx
                    w["cy"] = cy
                    break
            self.viewer.overlay_widgets = _json.dumps(widgets)
        except Exception:
            pass
        finally:
            self._updating_widget = False

    def _display(self) -> None:
        """Display the viewer widget (no-op when ``_auto_display`` is False)."""
        if self.viewer is None or not self._auto_display:
            return
        try:
            from IPython.display import display
            display(self.viewer)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WidgetSignal1DFigure  (drop-in for Signal1DFigure)
# ---------------------------------------------------------------------------

class WidgetSignal1DFigure(_WidgetPlotBase):
    """Drop-in replacement for
    :class:`~hyperspy.drawing.signal1d.Signal1DFigure` that renders via
    :class:`~hyperspy.viewer.viewer1d.Viewer1D`.

    Only the attributes and methods actually used by
    :class:`~hyperspy.drawing.mpl_hse.MPL_HyperSignal1D_Explorer` and
    :meth:`~hyperspy.signal.BaseSignal.plot` are implemented.
    """

    def __init__(self, title: str = "", **kwargs) -> None:
        super().__init__(title=title)

        # attributes expected by MPL_HyperSignal1D_Explorer / Signal1DFigure
        self.ax_lines: list = []
        self.right_ax_lines: list = []
        self.right_axes_manager = None
        self.xlabel: str = ""
        self.ylabel: str = ""
        self.axis = None            # the signal axis object

        # Primary line data function (set via add_line)
        self._data_function: Callable | None = None
        self._data_function_kwargs: dict = {}

    # ------------------------------------------------------------------
    # Signal1DFigure-compatible API used by MPL_HyperSignal1D_Explorer
    # ------------------------------------------------------------------

    def create_axis(self) -> None:
        """No-op — axis is part of the Viewer1D widget."""
        pass

    def create_right_axis(self, color: str = "black", adjust_layout: bool = True) -> None:
        """No-op for widget backend (right-axis overlay not supported)."""
        pass

    def add_line(self, line, ax: str = "left", connect_navigation: bool = False) -> None:
        """Register a :class:`~hyperspy.drawing.signal1d.Signal1DLine`."""
        if ax == "left":
            line.ax = self.ax   # may be None — that's fine
            if line.axes_manager is None:
                line.axes_manager = self.axes_manager
            self.ax_lines.append(line)
            line.sf_lines = self.ax_lines
        else:
            # right axis — record but don't render separately
            self.right_ax_lines.append(line)
            line.sf_lines = self.right_ax_lines
        line.axis = self.axis

    # ------------------------------------------------------------------
    def plot(self, data_function_kwargs: dict | None = None, **kwargs) -> None:
        """Fetch data from the first line and build the Viewer1D."""
        if data_function_kwargs is None:
            data_function_kwargs = {}
        if not self.ax_lines:
            _logger.warning("WidgetSignal1DFigure.plot called with no lines.")
            return

        line = self.ax_lines[0]
        data = line.data_function(
            axes_manager=self.axes_manager,
            **line.data_function_kwargs,
        )
        data = np.asarray(data, dtype=float).ravel()

        x_axis, units, y_units = self._axes_info()

        from hyperspy.viewer.viewer1d import Viewer1D

        color = getattr(line, "color", None) or "#4fc3f7"
        self.viewer = Viewer1D(
            data,
            x_axis=x_axis,
            units=units,
            y_units=y_units,
            color=color,
        )

        # Connect axes_manager navigation to update
        if self.axes_manager is not None:
            self.axes_manager.events.indices_changed.connect(self.update, [])
            self.events.closed.connect(
                lambda: self.axes_manager.events.indices_changed.disconnect(self.update),
                [],
            )

        # Push any markers that were added before the viewer was built
        self._push_deferred_markers()

        self._display()

    # ------------------------------------------------------------------
    def update(self) -> None:
        """Refresh the viewer with the current navigation position."""
        if self.viewer is None or not self.ax_lines:
            return

        line = self.ax_lines[0]
        data = line.data_function(
            axes_manager=self.axes_manager,
            **line.data_function_kwargs,
        )
        data = np.asarray(data, dtype=float).ravel()

        x_axis, units, _ = self._axes_info()
        self.viewer.update(data, x_axis=x_axis)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _axes_info(self):
        """Return ``(x_axis, units, y_units)`` from the signal axis object."""
        x_axis = None
        units = "px"
        y_units = ""

        if self.axis is not None:
            x_axis = np.asarray(self.axis.axis, dtype=float)
            if self.axis.units is not Undefined and self.axis.units:
                units = str(self.axis.units)

        return x_axis, units, y_units

    def _display(self) -> None:
        """Push the viewer to the cell output (no-op when ``_auto_display`` is False)."""
        if self.viewer is None or not self._auto_display:
            return
        try:
            from IPython.display import display
            display(self.viewer)
        except Exception:
            pass

