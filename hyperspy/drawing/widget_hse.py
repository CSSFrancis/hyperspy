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
Widget-backed HyperSignal1D Explorer.

Drop-in replacement for
:class:`~hyperspy.drawing.mpl_hse.MPL_HyperSignal1D_Explorer` that renders
both the signal panel (via :class:`~hyperspy.viewer.viewer1d.Viewer1D`) and
the navigator panel (via :class:`~hyperspy.viewer.viewer1d.Viewer1D` or
:class:`~hyperspy.viewer.viewer2d.Viewer2D` with a draggable pointer widget).
"""

from __future__ import annotations

import logging

from hyperspy.drawing.mpl_hse import MPL_HyperSignal1D_Explorer
from hyperspy.drawing.widget_he import Widget_HyperExplorer
from hyperspy.drawing.widget_plot import WidgetSignal1DFigure

_logger = logging.getLogger(__name__)


class Widget_HyperSignal1D_Explorer(Widget_HyperExplorer, MPL_HyperSignal1D_Explorer):
    """Hyper-spectrum explorer using widget viewers for both panels.

    * **Signal panel** — :class:`~hyperspy.viewer.viewer1d.Viewer1D`.
    * **Navigator panel** — :class:`~hyperspy.viewer.viewer1d.Viewer1D` (1-D
      nav) or :class:`~hyperspy.viewer.viewer2d.Viewer2D` (2-D nav), each
      with a draggable pointer that drives the navigation indices.

    Navigator rendering and the top-level :meth:`plot` are provided by
    :class:`~hyperspy.drawing.widget_he.Widget_HyperExplorer`.
    """

    def plot_signal(self, **kwargs: object) -> None:
        """Build a :class:`~hyperspy.drawing.widget_plot.WidgetSignal1DFigure`
        for the signal panel and wire it into the explorer infrastructure.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments forwarded where applicable.  Matplotlib-specific
            keys (``fig``, ``_on_figure_window_close``, ``data_function_kwargs``)
            are silently consumed.  Pass ``_suppress_display=True`` to prevent
            automatic display (used by
            :class:`~hyperspy.drawing.widget_he.Widget_HyperExplorer` to
            compose all viewers together).
        """
        import numpy as np
        from hyperspy.drawing import signal1d
        from hyperspy.drawing.mpl_he import MPL_HyperExplorer

        # fft/power-spectrum bookkeeping only — no matplotlib axis setup
        MPL_HyperExplorer.plot_signal(self, **{})

        # Pull out non-widget kwargs before constructing the figure
        _suppress_display = bool(kwargs.pop("_suppress_display", False))
        for _k in ("fig", "_on_figure_window_close", "data_function_kwargs"):
            kwargs.pop(_k, None)

        self.axis = self.axes_manager.signal_axes[0]

        sf = WidgetSignal1DFigure(title=self.signal_title + " Signal")
        sf.axis = self.axis
        sf.axes_manager = self.axes_manager
        sf._auto_display = not _suppress_display

        try:
            from traits.api import Undefined  # type: ignore[attr-defined]
        except ImportError:
            Undefined = None  # type: ignore[assignment,misc]

        self.xlabel = str(self.axes_manager.signal_axes[0])
        if self.axes_manager.signal_axes[0].units is not Undefined:
            self.xlabel += f" ({self.axes_manager.signal_axes[0].units})"
        self.ylabel = self.quantity_label if self.quantity_label else "Intensity"
        sf.xlabel = self.xlabel
        sf.ylabel = self.ylabel

        self.signal_plot = sf

        # Primary line
        is_complex = np.iscomplexobj(self.signal_data_function())
        sl = signal1d.Signal1DLine()
        sl.data_function = self.signal_data_function
        sl.data_function_kwargs = self.signal_data_function_kwargs
        sl.plot_indices = True
        color = self.pointer.color if self.pointer is not None else "red"
        sl.set_line_properties(color=color, type="step")
        sf.add_line(sl)

        # Imaginary part for complex data
        if is_complex:
            sl_imag = signal1d.Signal1DLine()
            sl_imag.data_function = self.signal_data_function
            sl_imag.data_function_kwargs = self.signal_data_function_kwargs
            sl_imag._plot_imag = True
            sl_imag.set_line_properties(color="blue", type="step")
            sf.add_line(sl_imag)

        sf.plot(**kwargs)
