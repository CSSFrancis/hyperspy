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
Widget-backed HyperExplorer base.

Overrides :meth:`~hyperspy.drawing.mpl_he.MPL_HyperExplorer.plot_navigator`
so that the navigator panel is rendered via
:class:`~hyperspy.viewer.viewer1d.Viewer1D` /
:class:`~hyperspy.viewer.viewer2d.Viewer2D` with a draggable pointer widget
instead of a matplotlib figure.

The :meth:`plot` method is also overridden to display all widget viewers via
``IPython.display.display`` (laid out side-by-side with ipywidgets HBox) and
to skip the ``ipympl``-specific ``figure.canvas`` display path that the
matplotlib base-class uses.
"""

from __future__ import annotations

import logging
from functools import partial

try:
    from traits.api import Undefined  # type: ignore[attr-defined]
except ImportError:
    Undefined = None  # type: ignore[assignment,misc]

from hyperspy.drawing.mpl_he import MPL_HyperExplorer
from hyperspy.drawing.widget_plot import (
    WidgetNavigatorPlot1D,
    WidgetNavigatorPlot2D,
)

_logger = logging.getLogger(__name__)


class Widget_HyperExplorer(MPL_HyperExplorer):
    """HyperExplorer variant whose navigator panel uses the widget viewers.

    Both the signal *and* navigator panels are rendered as anywidget / Viewer
    objects.  A draggable pointer widget (vline for 1-D nav, crosshair for 2-D
    nav) drives the :class:`~hyperspy.axes.AxesManager` navigation indices,
    which in turn refresh the signal viewer.

    All other behaviour (pointer assignment, ``assign_pointer``, ``close``,
    ``is_active``, etc.) is inherited unchanged from
    :class:`~hyperspy.drawing.mpl_he.MPL_HyperExplorer`.
    """

    # ------------------------------------------------------------------
    def plot_navigator(self, title: str | None = None, **kwargs: object) -> None:
        """Build a widget-backed navigator panel.

        For a 1-D navigation shape a :class:`WidgetNavigatorPlot1D` (Viewer1D
        + vline pointer) is created.  For a 2-D (or higher) navigation shape a
        :class:`WidgetNavigatorPlot2D` (Viewer2D + crosshair) is created.

        When the user drags the pointer the navigation indices on
        ``axes_manager`` change, which fires ``indices_changed``, which in turn
        calls ``signal_plot.update()``.

        Parameters
        ----------
        title : str, optional
            Navigator panel title (informational only; not rendered by the
            widget viewers).
        **kwargs : dict
            Consumed silently — matplotlib-specific kwargs (``fig``,
            ``_on_figure_window_close``, ``cmap``, …) are ignored so that the
            caller does not need to be aware of the backend.
        """
        if self.axes_manager.navigation_dimension == 0:
            return
        if self.navigator_data_function is None:
            return
        if self.navigator_data_function == "slider":
            self._get_navigation_sliders()
            return

        title = title or (self.signal_title + " Navigator" if self.signal_title else "")

        nav_shape = self.navigator_data_function().shape

        if len(nav_shape) == 1:
            # ── 1-D navigator ─────────────────────────────────────────────
            self._plot_navigator_1d(title, **kwargs)
        else:
            # ── 2-D (or higher) navigator ─────────────────────────────────
            self._plot_navigator_2d(title, **kwargs)

    # ------------------------------------------------------------------
    def _plot_navigator_1d(self, title: str, _suppress_display: bool = False,
                           **kwargs: object) -> None:
        """Create a :class:`WidgetNavigatorPlot1D` navigator."""
        from hyperspy.drawing import signal1d

        nav_axis = self.axes_manager.navigation_axes[0]

        nf = WidgetNavigatorPlot1D(title=title)
        nf.axes_manager = self.axes_manager
        nf.axis = nav_axis
        nf._auto_display = not _suppress_display

        # Label
        xlabel = str(nav_axis)
        if nav_axis.units is not Undefined and nav_axis.units:
            xlabel += f" ({nav_axis.units})"
        nf.xlabel = xlabel
        nf.ylabel = r"Σ data"

        # Build a Signal1DLine carrying the navigator data function
        sl = signal1d.Signal1DLine()
        sl.data_function = self.navigator_data_function
        sl.set_line_properties(color="#4fc3f7", type="step")
        nf.add_line(sl)

        nf.plot()
        self.navigator_plot = nf

        # Connect higher-dimensional navigation sliders if needed
        if self.axes_manager.navigation_dimension > 1:
            self._get_navigation_sliders()
            for axis in self.axes_manager.navigation_axes[1:]:
                axis.events.index_changed.connect(nf.update, [])
                self.events.closed.connect(
                    partial(axis.events.index_changed.disconnect, nf.update), []
                )

    # ------------------------------------------------------------------
    def _plot_navigator_2d(self, title: str, _suppress_display: bool = False,
                           **kwargs: object) -> None:
        """Create a :class:`WidgetNavigatorPlot2D` navigator."""
        nf = WidgetNavigatorPlot2D(title=title)
        nf.axes_manager = self.axes_manager
        nf.data_function = self.navigator_data_function
        nf._auto_display = not _suppress_display

        # Assign axes — mirrors the logic in MPL_HyperExplorer.plot_navigator
        if self.axes_manager.navigation_dimension == 1:
            nf.yaxis = self.axes_manager.navigation_axes[0]
            if self.axes_manager.signal_axes:
                nf.xaxis = self.axes_manager.signal_axes[0]
        elif self.axes_manager.navigation_dimension >= 2:
            nf.yaxis = self.axes_manager.navigation_axes[1]
            nf.xaxis = self.axes_manager.navigation_axes[0]
            if self.axes_manager.navigation_dimension > 2:
                self._get_navigation_sliders()
                for axis in self.axes_manager.navigation_axes[2:]:
                    axis.events.index_changed.connect(nf.update, [])
                    self.events.closed.connect(
                        partial(axis.events.index_changed.disconnect, nf.update), []
                    )

        nf.plot()
        self.navigator_plot = nf

    # ------------------------------------------------------------------
    def plot(self, **kwargs: object) -> None:
        """Plot both the signal and navigator panels as widget viewers.

        Overrides the matplotlib base implementation to:

        1. Skip the ``ipympl``-specific ``figure.canvas`` display path.
        2. Call ``plot_navigator`` and ``plot_signal`` directly.
        3. Display the resulting viewers side-by-side in the notebook using
           ``ipywidgets.HBox`` (or stacked with ``VBox``).

        Parameters
        ----------
        **kwargs : dict
            Same keyword arguments as
            :meth:`~hyperspy.drawing.mpl_he.MPL_HyperExplorer.plot` (most
            matplotlib-specific kwargs are silently consumed).
        """
        # Pull out fft/power-spectrum kwargs so signal_data_function_kwargs
        # is populated before plot_signal is called.
        for key in ("power_spectrum", "fft_shift"):
            if key in kwargs:
                self.signal_data_function_kwargs[key] = kwargs.pop(key)

        # plot_style and the sentinel 'fig' are matplotlib-specific — drop them
        kwargs.pop("plot_style", None)
        kwargs.pop("fig", None)   # sentinel set by show() to skip subfigure block

        # Assign the navigation pointer (MPL widget — we only need the
        # connect_navigate() call; we never call set_mpl_ax on it)
        if self.pointer is None:
            pointer_cls = self.assign_pointer()
            if pointer_cls is not None:
                self.pointer = pointer_cls(self.axes_manager)
                self.pointer.is_pointer = True
                self.pointer.color = "red"
                self.pointer.connect_navigate()
                self.events.closed.connect(self.pointer.disconnect, [])

        # Build navigator viewer — _suppress_display prevents individual display
        nav_kwds = kwargs.pop("navigator_kwds", {})
        self.plot_navigator(_suppress_display=True, **nav_kwds)

        # Build signal viewer — same
        self.plot_signal(_suppress_display=True, **kwargs)

        # Compose and display both viewers together
        self._display_viewers()

    # ------------------------------------------------------------------
    def _display_viewers(self) -> None:
        """Display the signal and navigator viewer widgets in the notebook.

        Uses ``ipywidgets.HBox`` to place them side-by-side.  Falls back to
        sequential ``display()`` calls if ipywidgets is unavailable.
        """
        nav_viewer = (
            self.navigator_plot.viewer
            if self.navigator_plot is not None
            else None
        )
        sig_viewer = (
            self.signal_plot.viewer
            if self.signal_plot is not None
            else None
        )

        try:
            from IPython.display import display
        except ImportError:
            return

        if nav_viewer is not None and sig_viewer is not None:
            try:
                from ipywidgets import HBox
                display(HBox([nav_viewer, sig_viewer]))
                return
            except ImportError:
                pass
            # ipywidgets not available — display sequentially
            display(nav_viewer)
            display(sig_viewer)
        elif sig_viewer is not None:
            display(sig_viewer)
        elif nav_viewer is not None:
            display(nav_viewer)

