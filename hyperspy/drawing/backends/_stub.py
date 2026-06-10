"""Stub template for new HyperSpy plotting backends.

Copy this file, rename ``StubBackend`` to ``MyBackend``, and implement the
methods marked REQUIRED.  Complete signatures and semantics for every method
are documented in :mod:`hyperspy.drawing.backends._protocol`.

Registration::

    from hyperspy.drawing.backends import register_backend
    register_backend(MyBackend())

Then users can switch with::

    import hyperspy.api as hs
    hs.preferences.Plot.backend = "mybackend"

Signature conventions
---------------------
* ``fig``      — whatever ``create_figure`` returned
* ``ax``       — whatever ``create_axes`` returned
* ``handle``   — whatever the corresponding ``plot_*`` / ``create_*`` returned
* ``cid``      — whatever the ``connect_*`` call returned (for disconnection)

HyperSpy never imports your library directly; all calls go through this class.
"""

from __future__ import annotations

from hyperspy.drawing.backends._protocol import (
    BlitMixin,
    PointerMixin,
)


class StubBackend(BlitMixin, PointerMixin):
    """Minimal backend skeleton.

    Methods in the REQUIRED sections below must be implemented.
    See ``_protocol.py`` for the full signature and semantics of each method.

    OPTIONAL sections show what to override to enable additional features;
    the default implementations (from BlitMixin / PointerMixin) are safe
    no-ops or ``BackendCapabilityError`` raisers.
    """

    # =========================================================================
    # Figure lifecycle                                               [REQUIRED]
    # =========================================================================

    def create_figure(self, title=None, on_close=None, **kwargs):
        # Call on_close() when the user closes the window.
        raise NotImplementedError

    def close_figure(self, fig):
        raise NotImplementedError

    def draw_idle(self, fig):
        raise NotImplementedError

    def disconnect_event(self, fig_or_ax, cid):
        raise NotImplementedError

    # =========================================================================
    # Axes                                                           [REQUIRED]
    # =========================================================================

    def create_axes(self, fig, **kwargs):
        raise NotImplementedError

    def set_xlabel(self, ax, label):
        raise NotImplementedError

    def set_ylabel(self, ax, label):
        raise NotImplementedError

    def set_title(self, ax, title):
        raise NotImplementedError

    def set_xlim(self, ax, xmin, xmax):
        raise NotImplementedError

    def set_ylim(self, ax, ymin, ymax):
        raise NotImplementedError

    def get_xlim(self, ax):
        raise NotImplementedError

    def get_ylim(self, ax):
        raise NotImplementedError

    def get_xbound(self, ax):
        raise NotImplementedError

    def set_axis_off(self, ax):
        raise NotImplementedError

    def set_aspect(self, ax, ratio):
        raise NotImplementedError

    def add_right_axis(self, ax, color="black"):
        raise NotImplementedError

    def remove_right_axis(self, ax, right_ax):
        raise NotImplementedError

    # =========================================================================
    # 1-D line plotting                                              [REQUIRED]
    # =========================================================================

    def plot_line(self, ax, x, y, **props):
        raise NotImplementedError

    def update_line(self, handle, x, y):
        raise NotImplementedError

    def remove_line(self, ax, handle):
        raise NotImplementedError

    def set_line_props(self, handle, **props):
        raise NotImplementedError

    def line_get_xdata(self, handle):
        raise NotImplementedError

    def line_get_color(self, handle):
        raise NotImplementedError

    def line_get_linewidth(self, handle):
        raise NotImplementedError

    # =========================================================================
    # Text annotations                                               [REQUIRED]
    # =========================================================================

    def add_text(self, ax, x, y, s, transform="axes", **kwargs):
        raise NotImplementedError

    def update_text(self, handle, s):
        raise NotImplementedError

    def remove_text(self, ax, handle):
        raise NotImplementedError

    def text_set_color(self, handle, color):
        raise NotImplementedError

    def text_get_color(self, handle):
        raise NotImplementedError

    # =========================================================================
    # Artist property                                                [REQUIRED]
    # =========================================================================

    def artist_set_animated(self, handle, animated):
        raise NotImplementedError

    # =========================================================================
    # 2-D image plotting                                             [REQUIRED]
    # =========================================================================

    def plot_image(self, ax, data, extent=None, vmin=None, vmax=None, norm=None, cmap="gray", **kwargs):
        raise NotImplementedError

    def plot_mesh(self, ax, x, y, data, **kwargs):
        raise NotImplementedError

    def image_set_data(self, handle, data):
        raise NotImplementedError

    def image_set_extent(self, handle, extent):
        raise NotImplementedError

    def image_set_clim(self, handle, vmin, vmax):
        raise NotImplementedError

    def image_set_norm(self, handle, norm):
        raise NotImplementedError

    def get_image_handle(self, ax):
        raise NotImplementedError

    # =========================================================================
    # Colorbar                                                       [REQUIRED]
    # =========================================================================

    def add_colorbar(self, fig, im_handle, ax):
        raise NotImplementedError

    def colorbar_set_label(self, cb, label):
        raise NotImplementedError

    def colorbar_remove(self, cb):
        raise NotImplementedError

    def colorbar_redraw(self, cb, fig):
        raise NotImplementedError

    # =========================================================================
    # Events                                                         [REQUIRED]
    # =========================================================================

    def connect_key_press(self, fig_or_ax, fn):
        raise NotImplementedError

    def connect_mouse_move(self, fig_or_ax, fn):
        raise NotImplementedError

    def connect_mouse_press(self, fig_or_ax, fn):
        raise NotImplementedError

    def connect_mouse_release(self, fig_or_ax, fn):
        raise NotImplementedError

    def connect_pick(self, fig_or_ax, fn):
        raise NotImplementedError

    # =========================================================================
    # Marker collections                                             [REQUIRED]
    # =========================================================================

    def add_collection(self, ax, collection):
        raise NotImplementedError

    def collection_update(self, handle, **kwargs):
        raise NotImplementedError

    def collection_remove(self, ax, handle):
        raise NotImplementedError

    # =========================================================================
    # Layout                                                         [REQUIRED]
    # =========================================================================

    def tight_layout(self, fig):
        pass  # no-op is acceptable

    def get_figure_from_ax(self, ax):
        raise NotImplementedError

    # =========================================================================
    # Explorer                                                       [REQUIRED]
    # =========================================================================

    def get_explorer(self, signal_dim: int):
        """Return the HyperExplorer subclass for signal_dim (0, 1, or 2).

        The built-in explorers work with any backend that passes protocol
        conformance.  Subclass them only if your backend needs custom layout::

            from hyperspy.drawing.he import HyperExplorer
            from hyperspy.drawing.hse import HyperSignal1D_Explorer
            from hyperspy.drawing.hie import HyperImage_Explorer

            _MAP = {0: HyperExplorer, 1: HyperSignal1D_Explorer,
                    2: HyperImage_Explorer}
            return _MAP.get(signal_dim, HyperExplorer)
        """
        raise NotImplementedError

    # =========================================================================
    # Figure manager factories                                       [REQUIRED]
    # =========================================================================

    def create_signal1d_figure(self, title="", on_close=None, **kwargs):
        from hyperspy.drawing.signal1d import Signal1DFigure
        sf = Signal1DFigure(title=title, **kwargs)
        if on_close is not None:
            sf.events.closed.connect(lambda obj: on_close(), [])
        return sf

    def create_image_figure(self, title="", **kwargs):
        from hyperspy.drawing.image import ImagePlot
        return ImagePlot(title=title)

    # =========================================================================
    # BlitMixin overrides                                             [OPTIONAL]
    # =========================================================================
    # Inherit BlitMixin no-ops (supports_blit → False, etc.).
    # Override if your backend supports blitting:
    #
    #   def supports_blit(self, fig) -> bool:
    #       return fig.canvas.supports_blit
    #
    #   def copy_background(self, fig):
    #       return fig.canvas.copy_from_bbox(fig.bbox)
    #
    #   def restore_background(self, fig, background) -> None:
    #       fig.canvas.restore_region(background)
    #
    #   def blit(self, fig) -> None:
    #       fig.canvas.blit(fig.bbox)
    #
    #   def connect_draw_event(self, fig, fn):
    #       return fig.canvas.mpl_connect("draw_event", fn)
    #
    #   def draw_animated_artists(self, fig) -> None:
    #       for ax in fig.axes:
    #           for a in ax.get_children():
    #               if a.get_animated():
    #                   ax.draw_artist(a)

    # =========================================================================
    # PointerMixin overrides                                          [OPTIONAL]
    # =========================================================================
    # Default: raise BackendCapabilityError.  Override to enable interactive
    # navigation widgets.
    #
    #   def create_line_pointer(self, ax, axis, pos, color="red"):
    #       # axis='x' → vertical line; axis='y' → horizontal line
    #       ...
    #
    #   def update_line_pointer(self, handle, pos) -> None: ...
    #
    #   def create_rect_pointer(self, ax, x, y, w, h, color="red"): ...
    #
    #   def update_rect_pointer(self, handle, x, y, w, h) -> None: ...
    #
    #   def remove_pointer(self, ax, handle) -> None: ...
    #
    #   def set_pointer_style(self, handle, *, color=None, alpha=None, animated=None): ...
    #
    #   # For native marker rendering:
    #   def create_markers(self, ax, marker_type, **kwargs): ...
    #   def update_markers(self, handle, **kwargs): ...
    #   def remove_markers(self, ax, handle): ...
    #
    #   # For coordinate conversion (DATA / AXES / DISPLAY spaces):
    #   def convert_coords(self, ax, points, from_space, to_space): ...
    #
    #   # For interactive selection tools:
    #   def create_span_selector(self, ax, **kwargs): ...
    #   def create_polygon_selector(self, ax, **kwargs): ...

    # =========================================================================
    # Combined layout                                                 [OPTIONAL]
    # =========================================================================

    def create_combined_figure_panels(self, figsize=None):
        """Return (nav_fig, signal_fig) for a combined window; None → two windows."""
        return None

    def ensure_displayed(self, fig) -> None:
        """Force final render after plot() completes (for deferred backends)."""
        pass

    def connect_close_event(self, fig, fn):
        """Connect fn() to figure close; return a cid."""
        raise NotImplementedError

    # =========================================================================
    # Scalebar                                                         [OPTIONAL]
    # =========================================================================

    def create_scalebar(self, ax, **kwargs):
        raise NotImplementedError

    def remove_scalebar(self, ax, handle):
        raise NotImplementedError

    # =========================================================================
    # Image colormap                                                   [OPTIONAL]
    # =========================================================================

    def get_image_cmap_name(self, handle):
        raise NotImplementedError
