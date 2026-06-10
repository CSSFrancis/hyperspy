"""
==================
Writing a Custom Backend
==================

HyperSpy's plotting backend system is extensible.  Any library can be
connected by implementing the :class:`~hyperspy.drawing.backends._protocol.PlottingBackend`
protocol and registering the class as a Python entry point.

This example shows the minimal skeleton for a custom backend and walks
through the registration steps.  It does not produce a visible plot because
it only demonstrates the registration mechanism.

Registration via ``pyproject.toml``
------------------------------------

In your package's ``pyproject.toml``::

    [project.entry-points."hyperspy.backends"]
    mybackend = "mypackage.plotting:MyBackend"

After installing (``pip install -e .``), your backend is selectable::

    import hyperspy.api as hs
    hs.preferences.Plot.backend = "mybackend"

In-process registration (no install required)
----------------------------------------------

For notebooks and scripts you can register in-process:

"""

from hyperspy.drawing.backends import register_backend
from hyperspy.drawing.backends._protocol import BackendCapabilityError
from hyperspy.drawing.backends._stub import StubBackend


# %%
# Subclass ``StubBackend`` (which already inherits ``BlitMixin`` and
# ``PointerMixin``) and override only the methods your library supports.
# Every ``raise NotImplementedError`` in ``StubBackend`` will surface as a
# clear error message if the calling code tries to use that feature.


class MinimalDemoBackend(StubBackend):
    """Illustrative skeleton — not functional; replace stubs with real calls."""

    # -----------------------------------------------------------------------
    # Figure lifecycle
    # -----------------------------------------------------------------------

    def create_figure(self, title=None, on_close=None, **kwargs):
        # Replace with: fig = mylib.Figure(title=title); return fig
        fig = object()  # placeholder
        return fig

    def close_figure(self, fig):
        pass  # Replace with: mylib.close(fig)

    def draw_idle(self, fig):
        pass  # Replace with: fig.draw_idle() or equivalent

    def disconnect_event(self, fig_or_ax, cid):
        pass

    # -----------------------------------------------------------------------
    # Axes
    # -----------------------------------------------------------------------

    def create_axes(self, fig, **kwargs):
        ax = object()  # placeholder
        return ax

    def set_xlabel(self, ax, label): pass
    def set_ylabel(self, ax, label): pass
    def set_title(self, ax, title): pass
    def set_xlim(self, ax, xmin, xmax): pass
    def set_ylim(self, ax, ymin, ymax): pass
    def get_xlim(self, ax): return (0.0, 1.0)
    def get_ylim(self, ax): return (0.0, 1.0)
    def get_xbound(self, ax): return (0.0, 1.0)
    def set_axis_off(self, ax): pass
    def set_aspect(self, ax, ratio): pass

    def add_right_axis(self, ax, color="black"):
        raise BackendCapabilityError("MinimalDemoBackend does not support twin axes.")

    def remove_right_axis(self, ax, right_ax): pass

    # -----------------------------------------------------------------------
    # 1-D lines  (stub only — not functional)
    # -----------------------------------------------------------------------

    def plot_line(self, ax, x, y, **props): return object()
    def update_line(self, handle, x, y): pass
    def remove_line(self, ax, handle): pass
    def set_line_props(self, handle, **props): pass
    def line_get_xdata(self, handle): return []
    def line_get_color(self, handle): return "black"
    def line_get_linewidth(self, handle): return 1.0

    # -----------------------------------------------------------------------
    # Text
    # -----------------------------------------------------------------------

    def add_text(self, ax, x, y, s, transform="axes", **kwargs): return object()
    def update_text(self, handle, s): pass
    def remove_text(self, ax, handle): pass
    def text_set_color(self, handle, color): pass
    def text_get_color(self, handle): return "black"

    # -----------------------------------------------------------------------
    # Artist
    # -----------------------------------------------------------------------

    def artist_set_animated(self, handle, animated): pass

    # -----------------------------------------------------------------------
    # Images
    # -----------------------------------------------------------------------

    def plot_image(self, ax, data, extent=None, vmin=None, vmax=None, norm=None, cmap="gray", **kwargs):
        return object()

    def plot_mesh(self, ax, x, y, data, **kwargs): return object()
    def image_set_data(self, handle, data): pass
    def image_set_extent(self, handle, extent): pass
    def image_set_clim(self, handle, vmin, vmax): pass
    def image_set_norm(self, handle, norm): pass
    def get_image_handle(self, ax): return None

    # -----------------------------------------------------------------------
    # Colorbar
    # -----------------------------------------------------------------------

    def add_colorbar(self, fig, im_handle, ax): return object()
    def colorbar_set_label(self, cb, label): pass
    def colorbar_remove(self, cb): pass
    def colorbar_redraw(self, cb, fig): pass

    # -----------------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------------

    def connect_key_press(self, fig_or_ax, fn): return None
    def connect_mouse_move(self, fig_or_ax, fn): return None
    def connect_mouse_press(self, fig_or_ax, fn): return None
    def connect_mouse_release(self, fig_or_ax, fn): return None
    def connect_pick(self, fig_or_ax, fn): return None

    # -----------------------------------------------------------------------
    # Collections
    # -----------------------------------------------------------------------

    def add_collection(self, ax, collection): return object()
    def collection_update(self, handle, **kwargs): pass
    def collection_remove(self, ax, handle): pass

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------

    def get_figure_from_ax(self, ax): return object()

    # -----------------------------------------------------------------------
    # Explorer factory
    # -----------------------------------------------------------------------

    def get_explorer(self, signal_dim: int):
        from hyperspy.drawing.he import HyperExplorer
        from hyperspy.drawing.hse import HyperSignal1D_Explorer
        from hyperspy.drawing.hie import HyperImage_Explorer
        return {0: HyperExplorer, 1: HyperSignal1D_Explorer, 2: HyperImage_Explorer}.get(
            signal_dim, HyperExplorer
        )


# %%
# Register in-process.  The name "demo" will now be accepted by
# ``hs.preferences.Plot.backend = "demo"``.

register_backend("demo", MinimalDemoBackend)

print("Registered backends:", end=" ")
try:
    from hyperspy.drawing.backends._registry import available_backends
    print(", ".join(available_backends()))
except Exception:
    print("matplotlib, anyplotlib, demo")

# %%
# Verify the class satisfies the protocol.

from hyperspy.drawing.backends._protocol import PlottingBackend

assert isinstance(MinimalDemoBackend(), PlottingBackend), (
    "Backend does not satisfy the PlottingBackend protocol"
)
print("Protocol check passed.")
