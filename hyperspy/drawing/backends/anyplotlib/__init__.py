"""anyplotlib plotting backend for hyperspy."""

from __future__ import annotations

import numpy as np

from hyperspy.drawing.backends._protocol import (
    BackendBase,
    BackendCapabilityError,
    unsupported,
)

_NOT_YET = (
    "anyplotlib does not yet support '{}'. "
    "See the feature table in hyperspy/drawing/AGENTS.md."
)


def _unwrap_cycling(value):
    """Return *value* unchanged, or unwrap a 1-element cycling sequence to a scalar.

    HyperSpy stores singleton style values as ``(v,)`` tuples so that MPL
    collections cycle through them.  anyplotlib expects either a bare scalar
    or an array whose length matches the number of markers; a 1-element list
    fails ``_broadcast`` when n > 1, so we flatten it here.
    """
    if hasattr(value, "__len__") and not isinstance(value, str) and len(value) == 1:
        v = value[0]
        return float(v) if not isinstance(v, str) else v
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _chain(fn1, fn2):
    """Return a callable that calls fn1() then fn2()."""

    def _combined():
        fn1()
        fn2()

    return _combined


class _AplFigureProxy:
    """Proxy for a single axes panel within a shared anyplotlib Figure.

    When signal and navigator share one anyplotlib Figure (combined layout),
    each plot gets its own proxy. Both proxies share the same underlying
    anyplotlib Figure widget so that ``display()`` is only called once.
    """

    def __init__(self, real_fig, ax):
        self._real_fig = real_fig  # shared anyplotlib Figure widget
        self._hspy_ax = ax
        self._hspy_on_close = None


class AnyplotlibBackend(BackendBase):
    """Maps hyperspy drawing primitives to anyplotlib API.

    Optional features without an anyplotlib equivalent inherit the
    ``BackendCapabilityError`` defaults from :class:`BackendBase`; the
    ``@unsupported``-marked methods below are core-protocol gaps with
    anyplotlib-specific messages.  Query ``backend.supports(feature)``
    to discover them programmatically.
    """

    # ── Figure lifecycle ──────────────────────────────────────────────────

    def create_figure(self, title=None, on_close=None, **kwargs):
        import anyplotlib as apl

        # Pop the MPL-style window-close callback so it doesn't reach apl.subplots.
        # We store it on the figure object and invoke it from close_figure(),
        # giving anyplotlib figures the same semantics as MPL close callbacks.
        window_close = kwargs.pop("_on_figure_window_close", None)

        # If a pre-created panel proxy is passed, adopt it (combined layout).
        fig_kwarg = kwargs.pop("fig", None)
        if isinstance(fig_kwarg, _AplFigureProxy):
            if on_close is not None:
                fig_kwarg._hspy_on_close = on_close
            if window_close is not None:
                # Store additionally so close_figure fires it too.
                existing = getattr(fig_kwarg, "_hspy_window_close", None)
                fig_kwarg._hspy_window_close = (
                    window_close if existing is None else _chain(existing, window_close)
                )
            return fig_kwarg

        figsize = kwargs.pop("figsize", (640, 480))
        figsize = tuple(float(v) for v in figsize)  # normalize (handles ndarray)
        if max(figsize) < 50:
            # matplotlib uses inches; convert to pixels at 96 dpi
            figsize = (int(figsize[0] * 96), int(figsize[1] * 96))
        else:
            figsize = (int(figsize[0]), int(figsize[1]))
        fig, ax = apl.subplots(1, 1, figsize=figsize)
        fig._hspy_ax = ax
        ax.figure = fig  # hyperspy widgets use ax.figure to reach the figure
        if on_close is not None:
            fig._hspy_on_close = on_close
        if window_close is not None:
            fig._hspy_window_close = window_close
        return fig

    def close_figure(self, fig):
        if fig is None:
            return
        if isinstance(fig, _AplFigureProxy):
            # Proxy: fire close callbacks but don't close the shared real figure.
            on_close = fig._hspy_on_close
            window_close = getattr(fig, "_hspy_window_close", None)
            fig._hspy_on_close = None
            fig._hspy_window_close = None
            if on_close is not None:
                on_close()
            if window_close is not None:
                window_close()
            return
        # Clear before calling to prevent re-entrant close loop:
        # BlittedFigure stores on_close=self.close which calls close_figure again.
        on_close = getattr(fig, "_hspy_on_close", None)
        window_close = getattr(fig, "_hspy_window_close", None)
        fig._hspy_on_close = None
        fig._hspy_window_close = None
        try:
            fig.close()
        except Exception:
            pass
        if on_close is not None:
            on_close()
        if window_close is not None:
            window_close()

    def create_combined_figure_panels(self, figsize=None):
        """Create a 2-panel anyplotlib Figure; return (nav_proxy, signal_proxy).

        Both proxies wrap the same underlying Figure widget so ``draw_idle``
        shows the figure only once both panels have rendered (no half-drawn
        flicker).  Pass the proxies as ``fig=`` in ``navigator_kwds`` and the
        main ``kwargs`` respectively.
        """
        import anyplotlib as apl

        if figsize is None:
            figsize = (1280, 640)
        else:
            figsize = tuple(float(v) for v in figsize)
            if max(figsize) < 50:
                figsize = (int(figsize[0] * 96 * 2), int(figsize[1] * 96))
            else:
                figsize = (int(figsize[0] * 2), int(figsize[1]))

        fig, axes = apl.subplots(1, 2, figsize=figsize)
        nav_ax, signal_ax = axes[0], axes[1]

        nav_proxy = _AplFigureProxy(fig, nav_ax)
        signal_proxy = _AplFigureProxy(fig, signal_ax)

        # hyperspy widgets reach the figure via ax.figure; use the proxy so
        # draw_idle(ax.figure) goes through the panel-countdown logic.
        nav_ax.figure = nav_proxy
        signal_ax.figure = signal_proxy

        # Count down to zero as each panel calls draw_idle for the first time.
        fig._hspy_panels_remaining = 2

        return nav_proxy, signal_proxy

    def ensure_displayed(self, fig):
        """Force-display fig, bypassing the panel countdown.

        Called from signal.py after plot() completes so that a figure is always
        shown even when the navigator was skipped (slider / None).
        """
        if fig is None:
            return
        real_fig = fig._real_fig if isinstance(fig, _AplFigureProxy) else fig
        if getattr(real_fig, "_hspy_displayed", False):
            return
        real_fig._hspy_panels_remaining = 0  # clear any pending countdown
        try:
            from IPython.display import display

            display(real_fig)
            real_fig._hspy_displayed = True
        except ImportError:
            pass

    def draw_idle(self, fig):
        if fig is None:
            return
        real_fig = fig._real_fig if isinstance(fig, _AplFigureProxy) else fig
        if getattr(real_fig, "_hspy_displayed", False):
            return

        # For combined layouts: each proxy decrements _hspy_panels_remaining
        # exactly once (on its first draw_idle call).  Display only when all
        # panels have rendered so the figure never appears half-populated.
        if isinstance(fig, _AplFigureProxy) and not getattr(fig, "_hspy_drawn", False):
            fig._hspy_drawn = True
            remaining = getattr(real_fig, "_hspy_panels_remaining", 1)
            real_fig._hspy_panels_remaining = remaining - 1

        if getattr(real_fig, "_hspy_panels_remaining", 0) > 0:
            return  # still waiting for other panels

        try:
            from IPython.display import display

            display(real_fig)
            real_fig._hspy_displayed = True
        except ImportError:
            pass

    # Blit methods: BlitMixin no-op defaults (anyplotlib repaints natively).

    def disconnect_event(self, fig_or_ax, cid):
        if cid is None:
            return
        plot = self._get_plot(fig_or_ax)
        if plot is not None and hasattr(plot, "callbacks"):
            plot.callbacks.disconnect(cid)

    # ── Axes setup ───────────────────────────────────────────────────────

    def create_axes(self, fig, **kwargs):
        return fig._hspy_ax

    def set_xlabel(self, ax, label):
        if ax._plot is not None:
            ax._plot.set_xlabel(label)
        else:
            ax._hspy_pending_xlabel = label

    def set_ylabel(self, ax, label):
        if ax._plot is not None:
            ax._plot.set_ylabel(label)
        else:
            ax._hspy_pending_ylabel = label

    def set_title(self, ax, title):
        if ax._plot is not None:
            ax._plot.set_title(title)
        else:
            ax._hspy_pending_title = title

    def set_xlim(self, ax, xmin, xmax):
        if ax._plot is not None:
            ax._plot.set_xlim(xmin, xmax)

    def set_ylim(self, ax, ymin, ymax):
        if ax._plot is not None:
            ax._plot.set_ylim(ymin, ymax)

    def get_xlim(self, ax):
        if ax._plot is not None:
            plot = ax._plot
            # anyplotlib Plot2D exposes get_xbound but not get_xlim
            for attr in ("get_xlim", "get_xbound"):
                fn = getattr(plot, attr, None)
                if fn is not None:
                    return fn()
        return (0.0, 1.0)

    def get_ylim(self, ax):
        if ax._plot is not None:
            return ax._plot.get_ylim()
        return (0.0, 1.0)

    def get_xbound(self, ax):
        if ax._plot is not None:
            return ax._plot.get_xbound()
        return (0.0, 1.0)

    def set_axis_off(self, ax):
        if ax._plot is not None:
            ax._plot.set_axis_off()

    def set_aspect(self, ax, ratio):
        if ax._plot is not None and hasattr(ax._plot, "set_aspect"):
            ax._plot.set_aspect(ratio)

    @unsupported
    def add_right_axis(self, ax, color="black"):
        raise BackendCapabilityError(_NOT_YET.format("add_right_axis (twinx)"))

    @unsupported
    def remove_right_axis(self, ax, right_ax):
        raise BackendCapabilityError(_NOT_YET.format("remove_right_axis"))

    # ── 1-D line plotting ─────────────────────────────────────────────────

    def plot_line(self, ax, x, y, **props):
        """Draw a line; return the Plot1D handle."""
        color = props.get("color", "#4fc3f7")
        linestyle = props.get("linestyle", "solid")
        linewidth = props.get("linewidth", 1.5)
        alpha = props.get("alpha", 1.0)
        axes_arg = [np.asarray(x)] if x is not None else None
        plot = ax.plot(
            np.asarray(y),
            axes=axes_arg,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )
        self._apply_pending_labels(ax, plot)
        return plot

    def _apply_pending_labels(self, ax, plot):
        """Apply any labels buffered before a plot was attached to ax."""
        for attr, method in (
            ("_hspy_pending_xlabel", "set_xlabel"),
            ("_hspy_pending_ylabel", "set_ylabel"),
            ("_hspy_pending_title", "set_title"),
        ):
            val = getattr(ax, attr, None)
            if val is not None:
                getattr(plot, method)(val)
                try:
                    delattr(ax, attr)
                except AttributeError:
                    pass

    def update_line(self, handle, x, y):
        handle.set_data(np.asarray(y), x_axis=np.asarray(x))

    def remove_line(self, ax, handle):
        pass  # primary Plot1D cannot be individually removed from its panel

    def set_line_props(self, handle, **props):
        # Map the subset of MPL-style props that anyplotlib Plot1D supports.
        _prop_map = {
            "color": "color",
            "linewidth": "linewidth",
            "linestyle": "linestyle",
            "alpha": "alpha",
        }
        for mpl_key, apl_key in _prop_map.items():
            if mpl_key in props and hasattr(handle, apl_key):
                try:
                    setattr(handle, apl_key, props[mpl_key])
                except (AttributeError, TypeError):
                    pass

    def get_line_props(self, handle):
        return {
            "color": getattr(handle, "color", "#4fc3f7"),
            "linewidth": float(getattr(handle, "linewidth", 1.5)),
            "xdata": handle.x,
        }

    # ── Text annotations ─────────────────────────────────────────────────

    def add_text(self, ax, x, y, s, transform="axes", **kwargs):
        plot = self._primary_plot(ax)
        if plot is not None and hasattr(plot, "add_text"):
            # anyplotlib Plot2D/Plot1D exposes add_text when available.
            color = kwargs.get("color", "white")
            try:
                handle = plot.add_text(x, y, s, color=color)
                return handle
            except (TypeError, AttributeError):
                pass
        # Fall back to a lightweight sentinel that remembers the text content
        # and colour so that update_text / set_text_props work correctly even
        # without native text support.
        return _AplTextHandle(s, kwargs.get("color", "white"))

    def update_text(self, handle, s):
        if handle is None:
            return
        if isinstance(handle, _AplTextHandle):
            handle.s = s
        elif hasattr(handle, "set_text"):
            handle.set_text(s)

    def remove_text(self, ax, handle):
        if handle is None:
            return
        if isinstance(handle, _AplTextHandle):
            return  # sentinel — nothing to remove from the canvas
        try:
            handle.remove()
        except Exception:
            pass

    def set_text_props(self, handle, **props):
        if handle is None:
            return
        color = props.get("color")
        if color is None:
            return
        if isinstance(handle, _AplTextHandle):
            handle.color = color
        elif hasattr(handle, "set_color"):
            try:
                handle.set_color(color)
            except Exception:
                pass

    # ── Generic artist ────────────────────────────────────────────────────

    def artist_set_animated(self, handle, animated):
        pass  # anyplotlib repaints natively; no per-artist animated flag

    # ── 2-D image plotting ────────────────────────────────────────────────

    def plot_image(
        self,
        ax,
        data,
        extent=None,
        vmin=None,
        vmax=None,
        norm=None,
        cmap="gray",
        **kwargs,
    ):
        x_axis = y_axis = None
        if extent is not None:
            x0, x1, y1, y0 = extent
            x_axis = np.linspace(x0, x1, data.shape[1])
            y_axis = np.linspace(y0, y1, data.shape[0])
        plot = ax.imshow(
            np.asarray(data),
            axes=[x_axis, y_axis] if x_axis is not None else None,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        self._apply_pending_labels(ax, plot)
        # Store explicitly so overlay plot_line calls don't shadow the image.
        ax._hspy_image_plot = plot
        return plot

    def plot_mesh(self, ax, x, y, data, **kwargs):
        return ax.pcolormesh(
            np.asarray(data),
            x_edges=np.asarray(x),
            y_edges=np.asarray(y),
        )

    def image_set_data(self, handle, data):
        handle.set_data(np.asarray(data))

    def image_set_extent(self, handle, extent):
        x0, x1, y1, y0 = extent
        h = handle._state["image_height"]
        w = handle._state["image_width"]
        x_axis = np.linspace(x0, x1, w)
        y_axis = np.linspace(y0, y1, h)
        handle.set_extent(x_axis, y_axis)

    def image_set_clim(self, handle, vmin, vmax):
        handle.set_clim(vmin, vmax)

    def image_set_norm(self, handle, norm):
        from hyperspy.drawing.norm import HyperNorm

        if norm is None or not isinstance(norm, HyperNorm):
            return
        vmin = getattr(norm, "vmin", None)
        vmax = getattr(norm, "vmax", None)
        if vmin is not None or vmax is not None:
            handle.set_clim(vmin, vmax)

    def get_image_handle(self, ax):
        # Prefer the explicit image handle over ax._plot, which may be
        # overwritten by overlay line plots (e.g. scalebar).
        return getattr(ax, "_hspy_image_plot", None) or (
            ax._plot if ax._plot is not None else None
        )

    # ── Colorbar ─────────────────────────────────────────────────────────

    def add_colorbar(self, fig, im_handle, ax):
        im_handle.set_colorbar_visible(True)
        return _AplColorbar(im_handle)

    def colorbar_set_label(self, cb, label):
        cb._im.set_colorbar_label(label)

    def colorbar_remove(self, cb):
        cb._im.set_colorbar_visible(False)

    def colorbar_redraw(self, cb, fig):
        pass

    # ── Events ───────────────────────────────────────────────────────────

    @staticmethod
    def _wrap(fn):
        """Return a plain function wrapping fn.

        anyplotlib's add_event_handler sets fn._event_types, which fails on
        bound methods (they have no __dict__). Wrapping guarantees a plain
        function object that allows arbitrary attribute assignment.
        """

        def _handler(*args, **kwargs):
            return fn(*args, **kwargs)

        return _handler

    def connect_key_press(self, fig_or_ax, fn):
        plot = self._get_plot(fig_or_ax)
        if plot is not None:
            return plot.add_event_handler(self._wrap(fn), "key_down")
        return None

    def connect_mouse_move(self, fig_or_ax, fn):
        plot = self._get_plot(fig_or_ax)
        if plot is not None:
            return plot.add_event_handler(self._wrap(fn), "pointer_move")
        return None

    def connect_mouse_press(self, fig_or_ax, fn):
        plot = self._get_plot(fig_or_ax)
        if plot is not None:
            return plot.add_event_handler(self._wrap(fn), "pointer_down")
        return None

    def connect_mouse_release(self, fig_or_ax, fn):
        plot = self._get_plot(fig_or_ax)
        if plot is not None:
            return plot.add_event_handler(self._wrap(fn), "pointer_up")
        return None

    def connect_pick(self, fig_or_ax, fn):
        return self.connect_mouse_press(fig_or_ax, fn)

    def _get_plot(self, fig_or_ax):
        """Return the Plot1D/Plot2D attached to the given figure or axes."""
        if hasattr(fig_or_ax, "_plot"):
            # It's an Axes object
            return fig_or_ax._plot
        if hasattr(fig_or_ax, "_hspy_ax"):
            # It's a Figure or _AplFigureProxy — use the stored hyperspy axes
            return fig_or_ax._hspy_ax._plot
        return None

    # ── Navigation pointer widgets ────────────────────────────────────────

    def _primary_plot(self, ax):
        """Return the primary (image or line) plot for widget creation.

        Falls back from the explicit image handle to the current ax._plot so
        that scalebar overlay lines (which overwrite ax._plot) do not shadow
        the real primary plot.
        """
        return getattr(ax, "_hspy_image_plot", None) or getattr(ax, "_plot", None)

    def create_line_pointer(self, ax, axis, pos, color="red"):
        plot = self._primary_plot(ax)
        if plot is None:
            raise RuntimeError("ax has no plot; call plot_line or plot_image first")
        if axis == "x":
            return plot.add_vline_widget(x=float(pos), color=color)
        if not hasattr(plot, "add_widget"):
            raise BackendCapabilityError(_NOT_YET.format("create_line_pointer(y)"))
        return plot.add_widget("crosshair", cx=0.0, cy=float(pos), color=color)

    def update_line_pointer(self, handle, pos):
        if hasattr(handle, "x"):
            handle.x = float(pos)
        else:
            handle.cy = float(pos)

    def connect_widget_drag(self, handle, on_drag):
        try:
            from anyplotlib.widgets._widgets1d import VLineWidget
            from anyplotlib.widgets._widgets2d import CrosshairWidget
        except ImportError:
            return

        if isinstance(handle, VLineWidget):

            def _cb(event):
                on_drag(handle.x)

        elif isinstance(handle, CrosshairWidget):

            def _cb(event):
                on_drag(handle.cx, handle.cy)

        else:
            return
        handle.add_event_handler(self._wrap(_cb), "pointer_move")

    def create_rect_pointer(self, ax, x, y, w, h, color="red"):
        plot = self._primary_plot(ax)
        if plot is None or not hasattr(plot, "add_widget"):
            raise BackendCapabilityError(_NOT_YET.format("create_rect_pointer"))
        # x, y are the lower-left corner; convert to center for the crosshair
        cx = float(x) + float(w) / 2.0
        cy = float(y) + float(h) / 2.0
        return plot.add_widget("crosshair", cx=cx, cy=cy, color=color)

    def update_rect_pointer(self, handle, x, y, w, h):
        # x, y are lower-left; convert to center so cx/cy equal the nav axis value
        handle.cx = float(x) + float(w) / 2.0
        handle.cy = float(y) + float(h) / 2.0

    def remove_pointer(self, ax, handle):
        try:
            handle.remove()
        except Exception:
            pass

    def set_pointer_style(self, handle, *, color=None, alpha=None, animated=None):
        if color is not None:
            handle.color = color
        if alpha is not None:
            raise BackendCapabilityError(_NOT_YET.format("set_pointer_style(alpha)"))

    # add_artist, simulate_pick: PointerMixin no-op defaults.
    # create_rect_patch, get_data_transform_inverse, transform_point,
    # create_span_selector, create_polygon_selector, get_ax_transform,
    # convert_coords, create_line2d_patch, create_circle_patch:
    # PointerMixin @unsupported defaults (raise BackendCapabilityError).

    # add_collection / collection_update / collection_remove: BackendBase
    # @unsupported defaults — anyplotlib renders markers natively instead.

    # ── Native marker collections ─────────────────────────────────────────

    # Transforms supported by anyplotlib markers.
    _MARKER_SPACE_MAP = {
        "data": "data",
        "axes": "axes",
        "display": "display",
        "xaxis": "data",  # MPL xaxis = x data, y axes → treat positions as data
        "yaxis": "data",  # MPL yaxis = y data, x axes → treat positions as data
        "relative": "data",
    }

    def create_markers(self, ax, marker_type, **kwargs):
        """Add a native anyplotlib marker group to *ax*.

        Parameters
        ----------
        marker_type : str
            One of the ``MarkerType.*`` string constants.
        **kwargs : dict
            HyperSpy/MPL-style marker kwargs plus ``offset_space`` and
            ``transform_space`` (popped before translation).

        Returns
        -------
        anyplotlib.markers.MarkerGroup
            The live handle; pass to ``update_markers`` / ``remove_markers``.
        """
        plot = self._primary_plot(ax)
        if plot is None:
            raise RuntimeError("ax has no plot; call plot_line or plot_image first")

        offset_space = kwargs.pop("offset_space", "data")
        kwargs.pop("transform_space", None)  # handled via offset_space

        translated = self._translate_marker_kwargs(marker_type, offset_space, kwargs)

        try:
            return plot.markers.add(marker_type, **translated)
        except ValueError as exc:
            raise BackendCapabilityError(
                f"anyplotlib does not support marker type '{marker_type}' "
                f"on this plot type: {exc}"
            ) from exc

    def update_markers(self, handle, **kwargs):
        """Update a ``MarkerGroup`` returned by ``create_markers``."""
        if not kwargs:
            return
        marker_type = handle._type
        # Derive the stored coordinate space so vlines/hlines un-segment correctly.
        offset_space = handle._data.get("transform", "data")
        translated = self._translate_marker_kwargs(marker_type, offset_space, kwargs)
        if translated:
            handle.set(**translated)

    def remove_markers(self, ax, handle):
        try:
            handle.remove()
        except Exception:
            pass

    @staticmethod
    def _translate_marker_kwargs(marker_type, offset_space, kwargs):
        """Translate HyperSpy/MPL-style marker kwargs to anyplotlib wire kwargs.

        Handles:
        - Coordinate space strings → anyplotlib ``transform``
        - ``circles.sizes`` → ``radius``
        - ``vlines/hlines`` full segments → 1-D offset lists
        - ``polygons.verts`` → ``vertices_list``
        - ``colors`` (MPL plural) → ``edgecolors``
        - ``linewidth`` (singular) → ``linewidths``
        - Strips MPL-only kwargs (``units``, ``patches``, ``drawstyle``, …)
        """
        out = {}

        _SPACE_MAP = {
            "data": "data",
            "axes": "axes",
            "display": "display",
            "xaxis": "data",
            "yaxis": "data",
            "relative": "data",
        }
        out["transform"] = _SPACE_MAP.get(offset_space, "data")

        # Work on a shallow copy so we can pop without mutating the caller's dict.
        work = dict(kwargs)

        # ── colour / linewidth renaming ─────────────────────────────────────
        if "colors" in work:
            val = work.pop("colors")
            # Flatten a one-element cycling list to a scalar colour string.
            if isinstance(val, (list, tuple)) and len(val) == 1:
                val = val[0]
            work["edgecolors"] = val

        if "linewidth" in work and "linewidths" not in work:
            work["linewidths"] = work.pop("linewidth")

        # ── type-specific positional key translations ───────────────────────
        if marker_type == "circles":
            # HyperSpy passes MPL-style ``sizes`` (display-unit area);
            # anyplotlib circles uses ``radius`` (data-unit radius).
            if "sizes" in work:
                out["radius"] = _unwrap_cycling(work.pop("sizes"))

        elif marker_type == "points":
            # HyperSpy wraps scalar sizes as a 1-element tuple for cycling.
            # anyplotlib expects either a scalar or a per-marker array.
            if "sizes" in work:
                work["sizes"] = _unwrap_cycling(work["sizes"])

        elif marker_type in ("vlines", "hlines"):
            # VerticalLines/HorizontalLines expand positions into full
            # [[x,0],[x,1]] / [[0,y],[1,y]] segments for the MPL path.
            # anyplotlib vlines/hlines want [[x], ...] / [[y], ...].
            if "segments" in work:
                segs = np.asarray(work.pop("segments"), dtype=float)
                if marker_type == "vlines":
                    out["offsets"] = [[float(v)] for v in segs[:, 0, 0]]
                else:
                    out["offsets"] = [[float(v)] for v in segs[:, 0, 1]]
            # Positions are always in data space for span-line types.
            out["transform"] = "data"

        elif marker_type == "polygons":
            # MPL PolyCollection uses ``verts``; anyplotlib uses ``vertices_list``.
            if "verts" in work:
                verts = work.pop("verts")
                out["vertices_list"] = [
                    np.asarray(v, dtype=float).tolist() for v in verts
                ]

        # ── copy remaining compatible kwargs ────────────────────────────────
        _STRIP = {"offset_transform", "units", "patches", "drawstyle"}
        for k, v in work.items():
            if k in _STRIP:
                continue
            # Eagerly convert numpy arrays so downstream JSON serialisation works.
            if hasattr(v, "tolist"):
                out[k] = v.tolist()
            else:
                out[k] = v

        return out

    # ── Misc primitives ────────────────────────────────────────────────────

    def plot_step(self, ax, x, y, **props):
        # anyplotlib has no dedicated step plot API; fall back to plot_line.
        # 'drawstyle' / 'steps-mid' props are silently dropped.
        props.pop("drawstyle", None)
        return self.plot_line(ax, x, y, **props)

    def set_autoscale(self, ax, enable):
        pass  # anyplotlib manages zoom internally

    def set_ticklabels(self, ax, axis, labels):
        pass  # cosmetic; anyplotlib tick control not yet exposed

    def get_figure_from_ax(self, ax):
        # For MPL-fallback axes (used until native anyplotlib figures exist),
        # delegate to the standard attribute.
        fig = getattr(ax, "figure", None)
        if fig is not None:
            return fig
        raise BackendCapabilityError(_NOT_YET.format("get_figure_from_ax"))

    # connect_close_event: BackendBase default (returns None) — anyplotlib
    # close handling is done via the on_close= kwarg at figure creation time.
    # create_signal1d_figure / create_image_figure / create_scalebar /
    # remove_scalebar: BackendBase defaults (generic figure managers).

    def get_explorer(self, signal_dim):
        if signal_dim == 1:
            from hyperspy.drawing.backends.anyplotlib._explorers import (
                Apl_HyperSignal1D_Explorer,
            )

            return Apl_HyperSignal1D_Explorer
        elif signal_dim == 2:
            from hyperspy.drawing.backends.anyplotlib._explorers import (
                Apl_HyperImage_Explorer,
            )

            return Apl_HyperImage_Explorer
        # 0-D (and the signal_dim validation) use the generic default.
        return super().get_explorer(signal_dim)

    def get_image_cmap_name(self, handle):
        return getattr(handle, "cmap", None) or "gray"


class _AplColorbar:
    """Sentinel returned by add_colorbar for the anyplotlib backend."""

    def __init__(self, im_handle):
        self._im = im_handle


class _AplTextHandle:
    """Lightweight sentinel for text annotations on backends without native text.

    Stores the text content and colour so that ``update_text`` /
    ``set_text_props`` remain functional even when the underlying canvas
    cannot render text.  ``remove_text`` is a no-op because there is nothing
    on the canvas to remove.
    """

    __slots__ = ("s", "color")

    def __init__(self, s: str, color: str = "white") -> None:
        self.s = s
        self.color = color

    def __repr__(self) -> str:
        return f"_AplTextHandle({self.s!r}, color={self.color!r})"
