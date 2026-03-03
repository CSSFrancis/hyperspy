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
Widget-backed HyperImage Explorer.

Drop-in replacement for :class:`~hyperspy.drawing.mpl_hie.MPL_HyperImage_Explorer`
that renders the *signal* panel via :class:`~hyperspy.viewer.viewer2d.Viewer2D`
instead of matplotlib.  The navigator panel (if any) is still handled by the
parent class using standard matplotlib.
"""

from __future__ import annotations

import logging

from hyperspy.drawing.mpl_hie import MPL_HyperImage_Explorer
from hyperspy.drawing.widget_he import Widget_HyperExplorer
from hyperspy.drawing.widget_plot import WidgetImagePlot

_logger = logging.getLogger(__name__)


class Widget_HyperImage_Explorer(Widget_HyperExplorer, MPL_HyperImage_Explorer):
    """Hyper-image explorer that uses :class:`~hyperspy.viewer.viewer2d.Viewer2D`
    for the signal panel and a widget-backed navigator (Viewer1D or Viewer2D
    with a draggable pointer) for the navigation panel.

    Signal-panel rendering is handled by :meth:`plot_signal` (overridden here).
    Navigator rendering and the top-level :meth:`plot` are provided by
    :class:`~hyperspy.drawing.widget_he.Widget_HyperExplorer`.
    """

    def plot_signal(self, **kwargs: object) -> None:
        """Build a :class:`~hyperspy.drawing.widget_plot.WidgetImagePlot` for
        the signal panel and wire it into the explorer infrastructure.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments.  Any key that matches an attribute of
            :class:`~hyperspy.drawing.widget_plot.WidgetImagePlot` is applied
            directly; the remainder are forwarded to
            :meth:`~hyperspy.drawing.widget_plot.WidgetImagePlot.plot`.
            Pass ``_suppress_display=True`` to prevent automatic display
            (used by :class:`~hyperspy.drawing.widget_he.Widget_HyperExplorer`
            to compose all viewers together).
        """
        _suppress_display = kwargs.pop("_suppress_display", False)

        from hyperspy.drawing.mpl_he import MPL_HyperExplorer
        MPL_HyperExplorer.plot_signal(self, **{})  # fft_shift bookkeeping only

        imf = WidgetImagePlot(title=self.signal_title + " Signal")
        imf.axes_manager = self.axes_manager
        imf.data_function = self.signal_data_function
        imf.xaxis, imf.yaxis = self.axes_manager.signal_axes
        imf._auto_display = not _suppress_display

        # Forward any recognised kwargs as attributes
        for key in list(kwargs.keys()):
            if hasattr(imf, key):
                setattr(imf, key, kwargs.pop(key))

        imf.quantity_label = self.quantity_label

        # Remove matplotlib-only kwargs that WidgetImagePlot does not need
        kwargs.pop("_on_figure_window_close", None)
        kwargs.pop("cmap", None)

        kwargs["data_function_kwargs"] = self.signal_data_function_kwargs
        imf.plot(**kwargs)

        self.signal_plot = imf

        # Connect navigation axes indices changes so the widget updates
        if self.axes_manager.navigation_axes:
            self.axes_manager.events.indices_changed.connect(imf.update, [])
            imf.events.closed.connect(
                lambda: self.axes_manager.events.indices_changed.disconnect(imf.update),
                [],
            )
