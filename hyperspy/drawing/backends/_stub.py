"""Stub template for new HyperSpy plotting backends.

Copy this file, rename ``StubBackend`` to ``MyBackend``, and implement the
methods marked REQUIRED.  Complete signatures and semantics for every method
are documented in :mod:`hyperspy.drawing.backends._protocol`.

``BackendBase`` already provides:

* no-op blit defaults (``supports_blit`` → False, …),
* ``BackendCapabilityError`` defaults for optional pointer / marker /
  selector features,
* working figure-manager, scale-bar and explorer factories that route all
  drawing back through your backend, and
* capability discovery (``backend.supports(feature)``).

So a *minimal* backend implements only the core drawing primitives below;
optional features are opt-in overrides.

Registration::

    from hyperspy.drawing.backends import register_backend
    register_backend(MyBackend())

or, for a distributable package, in ``pyproject.toml``::

    [project.entry-points."hyperspy.backends"]
    mybackend = "mypackage.plotting:MyBackend"

Then users can switch with::

    import hyperspy.api as hs
    hs.preferences.Plot.backend = "mybackend"

Verify conformance in your test suite::

    from hyperspy.drawing.backends.testing import BackendConformanceSuite

    class TestMyBackendConformance(BackendConformanceSuite):
        backend_factory = MyBackend

Signature conventions
---------------------
* ``fig``      — whatever ``create_figure`` returned
* ``ax``       — whatever ``create_axes`` returned
* ``handle``   — whatever the corresponding ``plot_*`` / ``create_*`` returned
* ``cid``      — whatever the ``connect_*`` call returned (for disconnection)

HyperSpy never imports your library directly; all calls go through this class.
"""

from __future__ import annotations

from hyperspy.drawing.backends._protocol import BackendBase


class StubBackend(BackendBase):
    """Minimal backend skeleton.

    Methods in the REQUIRED sections below must be implemented.
    See ``_protocol.py`` for the full signature and semantics of each method.

    Everything else is inherited from ``BackendBase`` as a safe no-op or a
    ``BackendCapabilityError`` raiser; the OPTIONAL section at the bottom
    shows what to override to enable additional features.
    """

    # =========================================================================
    # Figure lifecycle                                               [REQUIRED]
    # =========================================================================

    def create_figure(self, title=None, on_close=None, **kwargs):
        # Call on_close() when the user closes the window.  Also store and
        # fire the `_on_figure_window_close` kwarg (the explorer-level close
        # callback) from close_figure — see AnyplotlibBackend.create_figure.
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
    # Your axes objects must accept arbitrary attribute assignment
    # (HyperSpy sets ax.hspy_fig, ax.figure, ...).

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

    def set_autoscale(self, ax, enable):
        pass  # no-op is acceptable

    def set_ticklabels(self, ax, axis, labels):
        pass  # cosmetic; no-op is acceptable

    def add_right_axis(self, ax, color="black"):
        raise NotImplementedError  # or @unsupported + BackendCapabilityError

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

    def get_line_props(self, handle):
        # Must include at least "color" and "linewidth".
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

    def set_text_props(self, handle, **props):
        raise NotImplementedError

    # =========================================================================
    # Artist property                                                [REQUIRED]
    # =========================================================================

    def artist_set_animated(self, handle, animated):
        pass  # only meaningful for blitting backends; no-op is acceptable

    # =========================================================================
    # 2-D image plotting                                             [REQUIRED]
    # =========================================================================

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
        # norm is a hyperspy.drawing.norm.HyperNorm descriptor.
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
        pass  # no-op is acceptable

    # =========================================================================
    # Events                                                         [REQUIRED]
    # =========================================================================
    # Return None when an event type is not available; HyperSpy skips
    # disconnection for None cids.

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
    # Optional features                                              [OPTIONAL]
    # =========================================================================
    # Everything below is inherited from BackendBase.  Override to enable:
    #
    # Interactive navigation pointers (recommended — without these,
    # multi-dimensional navigation has no draggable cursor):
    #
    #   def create_line_pointer(self, ax, axis, pos, color="red"): ...
    #   def update_line_pointer(self, handle, pos): ...
    #   def create_rect_pointer(self, ax, x, y, w, h, color="red"): ...
    #   def update_rect_pointer(self, handle, x, y, w, h): ...
    #   def remove_pointer(self, ax, handle): ...
    #   def set_pointer_style(self, handle, *, color=None, alpha=None,
    #                         animated=None): ...
    #   def connect_widget_drag(self, handle, on_drag): ...
    #
    # Native marker rendering (otherwise markers raise
    # BackendCapabilityError unless you implement add_collection):
    #
    #   def create_markers(self, ax, marker_type, **kwargs): ...
    #   def update_markers(self, handle, **kwargs): ...
    #   def remove_markers(self, ax, handle): ...
    #
    # Coordinate conversion (CoordSpace strings: data/axes/display/…):
    #
    #   def convert_coords(self, ax, points, from_space, to_space): ...
    #   def get_ax_transform(self, ax, kind): ...
    #
    # Interactive selection tools (ROIs):
    #
    #   def create_span_selector(self, ax, **kwargs): ...
    #   def create_polygon_selector(self, ax, **kwargs): ...
    #
    # Blitting (see MplBackend for the reference implementation):
    #
    #   def supports_blit(self, fig): ...
    #   def copy_background(self, fig): ...
    #   def restore_background(self, fig, background): ...
    #   def blit(self, fig): ...
    #   def connect_draw_event(self, fig, fn): ...
    #   def draw_animated_artists(self, fig): ...
    #
    # Layout / lifecycle:
    #
    #   def create_combined_figure_panels(self, figsize=None): ...
    #   def ensure_displayed(self, fig): ...
    #   def connect_close_event(self, fig, fn): ...
    #   def tight_layout(self, fig): ...
    #   def get_figure_from_ax(self, ax): ...
    #
    # Custom explorers (the BackendBase default returns the generic
    # HyperExplorer classes, which work with any conformant backend):
    #
    #   def get_explorer(self, signal_dim): ...
