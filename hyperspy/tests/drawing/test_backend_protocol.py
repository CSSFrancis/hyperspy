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

"""Protocol tests for the built-in backends.

The conformance checks live in the public
:mod:`hyperspy.drawing.backends.testing` module so external backend packages
can run the identical suite; this file applies it to the built-in backends
and adds matplotlib-specific assertions.
"""

import inspect

import pytest

from hyperspy.drawing.backends.mpl import MplBackend
from hyperspy.drawing.backends.testing import (
    PROTOCOL_METHODS,
    BackendConformanceSuite,
)


def test_protocol_declares_all_required_methods():
    from hyperspy.drawing.backends._protocol import PlottingBackend

    members = {name for name, _ in inspect.getmembers(PlottingBackend)}
    for name in PROTOCOL_METHODS:
        assert name in members, f"PlottingBackend missing: {name}"


def test_protocol_has_no_undocumented_methods():
    """Every public protocol method must be listed in PROTOCOL_METHODS."""
    from hyperspy.drawing.backends._protocol import PlottingBackend

    public = {
        name
        for name, member in inspect.getmembers(PlottingBackend)
        if not name.startswith("_") and callable(member)
    }
    undocumented = public - set(PROTOCOL_METHODS)
    assert not undocumented, (
        f"Protocol methods missing from testing.PROTOCOL_METHODS: {undocumented}"
    )


class TestMplBackendConformance(BackendConformanceSuite):
    backend_factory = MplBackend


class TestAnyplotlibBackendConformance(BackendConformanceSuite):
    @staticmethod
    def backend_factory():
        pytest.importorskip("anyplotlib")
        from hyperspy.drawing.backends.anyplotlib import AnyplotlibBackend

        return AnyplotlibBackend()


class TestStubBackendConformance(BackendConformanceSuite):
    @staticmethod
    def backend_factory():
        from hyperspy.drawing.backends._stub import StubBackend

        return StubBackend()


def test_backend_capability_error_is_notimplementederror():
    from hyperspy.drawing.backends._protocol import BackendCapabilityError

    assert issubclass(BackendCapabilityError, NotImplementedError)


def test_default_backend_get_explorer_all_dims():
    import hyperspy.drawing  # noqa: F401 — ensure backend is registered
    from hyperspy.drawing.backends import get_backend
    from hyperspy.drawing.he import HyperExplorer

    b = get_backend()
    for dim in (0, 1, 2):
        cls = b.get_explorer(dim)
        assert issubclass(cls, HyperExplorer), f"dim={dim} returned non-HyperExplorer"


def test_mpl_backend_create_combined_returns_none_by_default():
    from hyperspy.defaults_parser import preferences

    original = preferences.Plot.use_subfigure
    try:
        preferences.Plot.use_subfigure = False
        result = MplBackend().create_combined_figure_panels()
        assert result is None
    finally:
        preferences.Plot.use_subfigure = original


def test_mpl_backend_get_explorer_correct_classes():
    from hyperspy.drawing.backends.mpl.mpl_he import MPL_HyperExplorer
    from hyperspy.drawing.backends.mpl.mpl_hie import MPL_HyperImage_Explorer
    from hyperspy.drawing.backends.mpl.mpl_hse import MPL_HyperSignal1D_Explorer

    b = MplBackend()
    assert b.get_explorer(0) is MPL_HyperExplorer
    assert b.get_explorer(1) is MPL_HyperSignal1D_Explorer
    assert b.get_explorer(2) is MPL_HyperImage_Explorer


class TestSupports:
    """Capability discovery on the built-in backends."""

    def test_mpl_supports_interactive_features(self):
        b = MplBackend()
        for feature in (
            "create_span_selector",
            "create_polygon_selector",
            "convert_coords",
            "create_markers",
            "add_collection",
            "get_ax_transform",
        ):
            assert b.supports(feature), feature

    def test_anyplotlib_reports_gaps(self):
        pytest.importorskip("anyplotlib")
        from hyperspy.drawing.backends.anyplotlib import AnyplotlibBackend

        b = AnyplotlibBackend()
        # Native markers and pointers are functional…
        assert b.supports("create_markers")
        assert b.supports("create_line_pointer")
        # …while MPL-collection and selector features are not.
        for feature in (
            "add_collection",
            "create_span_selector",
            "create_polygon_selector",
            "convert_coords",
            "add_right_axis",
        ):
            assert not b.supports(feature), feature

    def test_stub_inherits_unsupported_defaults(self):
        from hyperspy.drawing.backends._stub import StubBackend

        b = StubBackend()
        assert not b.supports("create_span_selector")
        assert not b.supports("create_markers")
        assert b.supports("create_figure")
