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

    canvas = _Canvas()


# ---------------------------------------------------------------------------
# Base class shared by WidgetImagePlot and WidgetSignal1DFigure
# ---------------------------------------------------------------------------

class _WidgetPlotBase:
    """Common interface plumbing for widget-backed plot objects.

    Sub-classes must implement:
    * :meth:`_build_viewer` — instantiate and return the viewer widget.
    * :meth:`update` — refresh the viewer from the current data.
    """

    def __init__(self, title: str = "") -> None:
        self._title = title
        self.ax = None          # Not meaningful for widget backend, kept for compat
        self.ax_markers: list = []
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
    # Marker compatibility (BlittedFigure interface)
    # ------------------------------------------------------------------
    def add_marker(self, marker) -> None:  # type: ignore[override]
        """Accept a HyperSpy marker object (silently ignored — markers
        defined in metadata are not yet forwarded to the widget backend)."""
        self.ax_markers.append(marker)

    def remove_markers(self, render_figure: bool = False) -> None:
        """Remove all markers."""
        self.ax_markers.clear()

    # ------------------------------------------------------------------
    # Close / lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Mark the plot as closed and fire the ``events.closed`` event."""
        _logger.debug("Closing widget plot %r.", self)
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

    # ------------------------------------------------------------------
    # render_figure — no-op (widget pushes diffs automatically via traitlets)
    # ------------------------------------------------------------------
    def render_figure(self) -> None:
        pass


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

