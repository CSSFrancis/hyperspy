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

"""Public conformance suite for HyperSpy plotting backends.

Third-party backend packages run this against their backend in their own
test suite — analogous to ``numpy.testing``::

    # test_conformance.py in the hyperspy-mybackend package
    from hyperspy.drawing.backends.testing import BackendConformanceSuite

    from hyperspy_mybackend import MyBackend


    class TestMyBackendConformance(BackendConformanceSuite):
        backend_factory = MyBackend

For a quick check outside pytest::

    from hyperspy.drawing.backends.testing import check_backend

    check_backend(MyBackend())  # raises AssertionError listing problems

HyperSpy's own backends are checked with this same suite in
``hyperspy/tests/drawing/test_backend_protocol.py``.
"""

from __future__ import annotations

#: Every method of the :class:`~hyperspy.drawing.backends._protocol.PlottingBackend`
#: protocol.  A conformant backend must *define* all of them; optional
#: features may be inherited as ``BackendCapabilityError``-raising defaults
#: from :class:`~hyperspy.drawing.backends._protocol.BackendBase`.
PROTOCOL_METHODS = [
    # Capability discovery
    "supports",
    # Figure lifecycle
    "create_figure",
    "close_figure",
    "draw_idle",
    "disconnect_event",
    # Blit (BlitMixin defaults: no blit support)
    "supports_blit",
    "copy_background",
    "restore_background",
    "blit",
    "connect_draw_event",
    "draw_animated_artists",
    "render_figure_from_ax",
    "invalidate_blit_background",
    "supports_blit_from_ax",
    # Axes setup
    "create_axes",
    "set_xlabel",
    "set_ylabel",
    "set_title",
    "set_xlim",
    "set_ylim",
    "get_xlim",
    "get_ylim",
    "get_xbound",
    "set_axis_off",
    "set_aspect",
    "add_right_axis",
    "remove_right_axis",
    "set_autoscale",
    "set_ticklabels",
    # 1-D lines
    "plot_line",
    "update_line",
    "remove_line",
    "set_line_props",
    "get_line_props",
    "plot_step",
    # Text annotations
    "add_text",
    "update_text",
    "remove_text",
    "set_text_props",
    # Generic artist
    "artist_set_animated",
    # 2-D images
    "plot_image",
    "plot_mesh",
    "image_set_data",
    "image_set_extent",
    "image_set_clim",
    "image_set_norm",
    "get_image_handle",
    "get_image_cmap_name",
    # Colorbar
    "add_colorbar",
    "colorbar_set_label",
    "colorbar_remove",
    "colorbar_redraw",
    # Event connections
    "connect_key_press",
    "connect_mouse_move",
    "connect_mouse_press",
    "connect_mouse_release",
    "connect_pick",
    "connect_close_event",
    # Pointer widgets (PointerMixin defaults: BackendCapabilityError)
    "create_line_pointer",
    "update_line_pointer",
    "create_rect_pointer",
    "update_rect_pointer",
    "remove_pointer",
    "set_pointer_style",
    "add_artist",
    "create_rect_patch",
    "simulate_pick",
    "connect_widget_drag",
    "create_span_selector",
    "create_polygon_selector",
    "get_ax_transform",
    "get_data_transform_inverse",
    "transform_point",
    "convert_coords",
    "create_line2d_patch",
    "create_circle_patch",
    # Native markers
    "create_markers",
    "update_markers",
    "remove_markers",
    # Legacy MPL collections
    "add_collection",
    "collection_update",
    "collection_remove",
    # Layout / lifecycle
    "tight_layout",
    "get_figure_from_ax",
    "create_combined_figure_panels",
    "ensure_displayed",
    # Figure manager / explorer factories
    "get_explorer",
    "create_signal1d_figure",
    "create_image_figure",
    # Scale bar
    "create_scalebar",
    "remove_scalebar",
]


def check_backend(backend) -> None:
    """Check *backend* against the :class:`PlottingBackend` protocol.

    Raises
    ------
    AssertionError
        Listing every missing or non-callable protocol method.
    """
    from hyperspy.drawing.backends._protocol import PlottingBackend

    problems = []
    for name in PROTOCOL_METHODS:
        method = getattr(backend, name, None)
        if method is None:
            problems.append(f"missing method: {name}")
        elif not callable(method):
            problems.append(f"not callable: {name}")
    assert not problems, (
        f"{type(backend).__name__} does not satisfy the PlottingBackend "
        "protocol:\n  " + "\n  ".join(problems)
    )
    assert isinstance(backend, PlottingBackend), (
        f"{type(backend).__name__} fails isinstance(..., PlottingBackend)"
    )


class BackendConformanceSuite:
    """Inheritable pytest test class checking protocol conformance.

    Subclass it in a test module and set :attr:`backend_factory` to your
    backend class (or any zero-argument callable returning an instance).
    Requires ``pytest``.
    """

    #: Zero-argument callable returning a backend instance.
    backend_factory = None

    def _backend(self):
        assert self.backend_factory is not None, (
            f"{type(self).__name__} must set `backend_factory` to the "
            "backend class under test."
        )
        return self.backend_factory()

    def test_satisfies_protocol(self):
        check_backend(self._backend())

    def test_supports_reports_booleans(self):
        backend = self._backend()
        for name in PROTOCOL_METHODS:
            result = backend.supports(name)
            assert isinstance(result, bool), f"supports({name!r}) returned {result!r}"

    def test_supports_unknown_feature_is_false(self):
        assert self._backend().supports("__not_a_protocol_method__") is False

    def test_get_explorer_returns_explorer_subclasses(self):
        from hyperspy.drawing.he import HyperExplorer

        backend = self._backend()
        for dim in (0, 1, 2):
            cls = backend.get_explorer(dim)
            assert issubclass(cls, HyperExplorer), (
                f"get_explorer({dim}) returned {cls!r}, not a HyperExplorer subclass"
            )

    def test_unsupported_features_raise_capability_error(self):
        """Every feature reported unsupported must raise BackendCapabilityError.

        Checked on a representative subset whose signatures allow a generic
        call with dummy arguments.
        """
        import pytest

        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        backend = self._backend()
        dummy_calls = {
            "create_span_selector": lambda b: b.create_span_selector(None),
            "create_polygon_selector": lambda b: b.create_polygon_selector(None),
            "convert_coords": lambda b: b.convert_coords(
                None, [[0, 0]], "data", "display"
            ),
            "create_rect_patch": lambda b: b.create_rect_patch((0, 0), 1, 1),
            "create_line2d_patch": lambda b: b.create_line2d_patch([0], [0]),
            "create_circle_patch": lambda b: b.create_circle_patch((0, 0), 1),
            "add_collection": lambda b: b.add_collection(None, None),
        }
        for feature, call in dummy_calls.items():
            if not backend.supports(feature):
                with pytest.raises(BackendCapabilityError):
                    call(backend)
