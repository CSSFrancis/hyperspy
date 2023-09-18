# Copyright 2007-2023 The HyperSpy developers
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
import pytest
from copy import deepcopy


import numpy as np
from matplotlib.collections import (
    LineCollection,
    PolyCollection,
)
from matplotlib.transforms import (
    IdentityTransform,
    Affine2D,
    CompositeGenericTransform,
    BlendedGenericTransform,
    BboxTransformTo,
)
import matplotlib.pyplot as plt
import dask.array as da

from hyperspy.drawing.markers import markers2collection
from hyperspy._signals.signal2d import Signal2D, BaseSignal, Signal1D
from hyperspy.axes import UniformDataAxis
from hyperspy.misc.test_utils import update_close_figure
from matplotlib.collections import (
    PolyCollection,
    CircleCollection,
    LineCollection,
    RegularPolyCollection,
)
from hyperspy.external.matplotlib.collections import (
    TextCollection,
    RectangleCollection,
    EllipseCollection,
)
from hyperspy.external.matplotlib.quiver import Quiver
from hyperspy.utils.markers import (
    Arrows,
    Circles,
    Ellipses,
    HorizontalLines,
    Markers,
    Points,
    VerticalLines,
    Rectangles,
    Squares,
    Texts,
    Lines,
)

BASELINE_DIR = "marker_collection"
DEFAULT_TOL = 2.0
STYLE_PYTEST_MPL = "default"
plt.style.use(STYLE_PYTEST_MPL)


class TestCollections:
    @pytest.fixture
    def data(self):
        d = np.empty((3,), dtype=object)
        for i in np.ndindex(d.shape):
            d[i] = np.stack([np.arange(3), np.ones(3) * i], axis=1)
        return d

    @pytest.fixture
    def lazy_data(self):
        d = np.empty((3,), dtype=object)
        for i in np.ndindex(d.shape):
            d[i] = np.stack([np.arange(3), np.ones(3) * i], axis=1)
        d = da.from_array(d, chunks=(1, 1, 1))

        return d

    @pytest.fixture
    def signal(self, data):
        sig = BaseSignal(data, ragged=True)
        sig.metadata.set_item(
            "Peaks.signal_axes",
            (
                UniformDataAxis(scale=0.5, offset=-1),
                UniformDataAxis(scale=2.0, offset=-2),
            ),
        )
        return sig

    @pytest.fixture
    def collections(self):
        collections = [
            Circles,
            Ellipses,
            Rectangles,
            Squares,
            Points,
        ]
        num_col = len(collections)
        offsets = [
            np.stack([np.ones(num_col) * i, np.arange(num_col)], axis=1)
            for i in range(len(collections))
        ]
        kwargs = [
            {"sizes": (1,)},
            {"widths": (0.2,), "heights": (0.7,), "angles": (60,), "units": "xy"},
            {"widths": (0.2,), "heights": (0.7,), "angles": (60,), "units": "xy"},
            {"sizes": (0.9,)},
            {"sizes": (1,)},
        ]
        for k, o in zip(kwargs, offsets):
            k["offsets"] = o
        collections = [c(**k) for k, c in zip(kwargs, collections)]
        return collections

    @pytest.mark.mpl_image_compare(
        baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL
    )
    def test_multi_collections_signal(self, collections):
        num_col = len(collections)
        s = Signal2D(np.zeros((2, num_col, num_col)))
        s.axes_manager.signal_axes[0].offset = 0
        s.axes_manager.signal_axes[1].offset = 0
        s.plot(interpolation=None)
        [s.add_marker(c) for c in collections]
        return s._plot.signal_plot.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL
    )
    def test_multi_collections_navigator(self, collections):
        num_col = len(collections)
        s = Signal2D(np.zeros((num_col, num_col, 1, 1)))
        s.axes_manager.signal_axes[0].offset = 0
        s.axes_manager.signal_axes[1].offset = 0
        s.plot(interpolation=None)
        [s.add_marker(c, plot_on_signal=False) for c in collections]
        return s._plot.navigator_plot.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL
    )
    @pytest.mark.parametrize("iter_data", ("lazy_data", "data"))
    def test_iterating_marker(self, request, iter_data):
        data = request.getfixturevalue(iter_data)
        s = Signal2D(np.ones((3, 5, 6)))
        markers = Points(offsets=data, sizes=(0.2,))
        s.add_marker(markers)
        s.axes_manager.navigation_axes[0].index = 2
        return s._plot.signal_plot.figure

    def test_parameters_2_scatter(self, data):
        m = Points(
            offsets=np.array([[100, 70], [70, 100]]),
            color="g",
            sizes=(3,),
        )
        s = Signal2D(np.zeros((100, 100)))
        s.add_marker(m)

    def test_parameters_singletons(self, signal, data):
        m = Points(offsets=np.array([[100, 70], [70, 100]]), color="b", sizes=3)
        s = Signal2D(np.zeros((2, 100, 100)))
        s.add_marker(m)

    def test_parameters_singletons_iterating(self):
        data = np.empty(2, dtype=object)
        data[0] = np.array([[100, 70], [70, 100]])
        data[1] = np.array([[100, 70], [70, 100]])
        sizes = np.empty(2, dtype=object)
        sizes[0] = 3
        sizes[1] = 4
        m = Points(offsets=np.array([[100, 70], [70, 100]]), color="b", sizes=sizes)
        s = Signal2D(np.zeros((2, 100, 100)))
        s.add_marker(m)

    @pytest.mark.parametrize(
        "signal_axes",
        (
            "metadata",
            (
                UniformDataAxis(scale=0.5, offset=-1),
                UniformDataAxis(scale=2.0, offset=-2),
            ),
            None,
        ),
    )
    def test_from_signal(self, signal, data, signal_axes):
        col = Points.from_signal(signal, sizes=(0.3,), signal_axes=signal_axes)

        s = Signal2D(np.ones((3, 5, 6)))
        s.add_marker(col)
        s.axes_manager.navigation_axes[0].index = 1
        if isinstance(signal_axes, (tuple, str)):
            ans = np.zeros_like(data[1])
            ans[:, 1] = data[1][:, 0] * 2 - 2
            ans[:, 0] = data[1][:, 1] * 0.5 - 1
            np.testing.assert_array_equal(col.get_data_position()["offsets"], ans)
        else:
            np.testing.assert_array_equal(col.get_data_position()["offsets"], data[1])

    def test_from_signal_fail(self, signal):
        with pytest.raises(ValueError):
            _ = Points.from_signal(signal, sizes=(0.3,), signal_axes="test")

    def test_find_peaks(self):
        from skimage.draw import disk
        from skimage.morphology import disk as disk2

        rr, cc = disk(
            center=(10, 8),
            radius=4,
        )
        img = np.zeros((2, 20, 20))
        img[:, rr, cc] = 1
        s = Signal2D(img)
        s.axes_manager.signal_axes[0].scale = 1.5
        s.axes_manager.signal_axes[1].scale = 2
        s.axes_manager.signal_axes[0].offset = -1
        s.axes_manager.signal_axes[1].offset = -1
        pks = s.find_peaks(
            interactive=False,
            method="template_matching",
            template=disk2(4),
        )
        col = Points.from_signal(
            pks, sizes=(0.3,), signal_axes=s.axes_manager.signal_axes
        )
        s.add_marker(col)
        np.testing.assert_array_equal(col.get_data_position()["offsets"], [[11, 19]])

    def test_find_peaks0d(self):
        from skimage.draw import disk
        from skimage.morphology import disk as disk2

        rr, cc = disk(
            center=(10, 8),
            radius=4,
        )
        img = np.zeros((1, 20, 20))
        img[:, rr, cc] = 1
        s = Signal2D(img)
        s.axes_manager.signal_axes[0].scale = 1.5
        s.axes_manager.signal_axes[1].scale = 2
        s.axes_manager.signal_axes[0].offset = -1
        s.axes_manager.signal_axes[1].offset = -1
        pks = s.find_peaks(
            interactive=False,
            method="template_matching",
            template=disk2(4),
        )
        col = Points.from_signal(
            pks, sizes=(0.3,), signal_axes=s.axes_manager.signal_axes
        )
        s.add_marker(col)
        np.testing.assert_array_equal(col.get_data_position()["offsets"], [[11, 19]])

    def test_deepcopy_markers(self, collections):
        num_col = len(collections)
        s = Signal2D(np.zeros((2, num_col, num_col)))
        s.axes_manager.signal_axes[0].offset = 0
        s.axes_manager.signal_axes[1].offset = 0
        s.plot(interpolation=None)
        [s.add_marker(c, permanent=True) for c in collections]
        new_s = deepcopy(s)
        assert len(new_s.metadata["Markers"]) == num_col

    def test_get_current_signal(self, collections):
        num_col = len(collections)
        s = Signal2D(np.zeros((2, num_col, num_col)))
        s.axes_manager.signal_axes[0].offset = 0
        s.axes_manager.signal_axes[1].offset = 0
        s.plot(interpolation=None)
        [s.add_marker(c, permanent=True) for c in collections]
        cs = s.get_current_signal()
        assert len(cs.metadata["Markers"]) == num_col

    def test_plot_and_render(self):
        markers = Points(offsets=[[1, 1], [4, 4]], sizes=(1,))
        s = Signal1D(np.arange(100).reshape((10, 10)))
        s.add_marker(markers)
        markers.plot(render_figure=True)


class TestInitMarkerCollection:
    @pytest.fixture
    def signal(self):
        signal = Signal2D(np.zeros((3, 10, 10)))
        return signal

    @pytest.fixture
    def static_line_collection(self, signal):
        segments = np.ones((10, 2, 2))
        markers = Markers(LineCollection, segments=segments)
        markers.axes_manager = signal.axes_manager
        return markers

    @pytest.fixture
    def iterating_line_collection(self, signal):
        data = np.empty((3,), dtype=object)
        for i in np.ndindex(data.shape):
            data[i] = np.ones((10, 2, 2)) * i
        markers = Markers(LineCollection, segments=data)
        markers.axes_manager = signal.axes_manager
        return markers

    def test_multiple_collections(
        self, static_line_collection, iterating_line_collection, signal
    ):
        signal.add_marker(static_line_collection, permanent=True)
        signal.add_marker(iterating_line_collection, permanent=True)
        assert len(signal.metadata.Markers) == 2

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_init(self, collection, request):
        col = request.getfixturevalue(collection)
        assert isinstance(col, Markers)

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_get_data(self, collection, request):
        col = request.getfixturevalue(collection)
        kwds = col.get_data_position()
        assert isinstance(kwds, dict)
        assert kwds["segments"].shape == (10, 2, 2)

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_to_dictionary(self, collection, request):
        col = request.getfixturevalue(collection)
        dict = col._to_dictionary()
        assert dict["collection_class"] == "LineCollection"
        assert dict["plot_on_signal"] is True

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_update(self, collection, request, signal):
        col = request.getfixturevalue(collection)
        signal.plot()
        signal.add_marker(col)
        signal.axes_manager.navigation_axes[0].index = 2
        if collection == "iterating_line_collection":
            col.get_data_position()["segments"]
            np.testing.assert_array_equal(
                col.get_data_position()["segments"], np.ones((10, 2, 2)) * 2
            )
        else:
            np.testing.assert_array_equal(
                col.get_data_position()["segments"], np.ones((10, 2, 2))
            )

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_fail_plot(self, collection, request):
        col = request.getfixturevalue(collection)
        with pytest.raises(AttributeError):
            col.plot()

    def test_deepcopy(self, iterating_line_collection):
        it_2 = deepcopy(iterating_line_collection)
        assert it_2 is not iterating_line_collection

    def test_wrong_navigation_size(self):
        s = Signal2D(np.zeros((2, 3, 3)))
        offsets = np.empty((3, 2), dtype=object)
        for i in np.ndindex(offsets.shape):
            offsets[i] = np.ones((3, 2))
        m = Points(offsets=offsets, sizes=(1,))
        with pytest.raises(ValueError):
            s.add_marker(m)

    def test_add_markers_to_multiple_signals(self):
        s = Signal2D(np.zeros((2, 3, 3)))
        s2 = Signal2D(np.zeros((2, 3, 3)))
        m = Points(offsets=[[1, 1], [2, 2]], sizes=(1,))
        s.add_marker(m, permanent=True)
        with pytest.raises(ValueError):
            s2.add_marker(m, permanent=True)

    def test_add_markers_to_same_signal(self):
        s = Signal2D(np.zeros((2, 3, 3)))
        m = Points(offsets=[[1, 1], [2, 2]], sizes=3)
        s.add_marker(m, permanent=True)
        with pytest.raises(ValueError):
            s.add_marker(m, permanent=True)

    def test_add_markers_to_navigator_without_nav(self):
        s = Signal2D(np.zeros((3, 3)))
        m = Points(offsets=[[1, 1], [2, 2]], sizes=(1,))
        with pytest.raises(ValueError):
            s.add_marker(m, plot_on_signal=False)

    def test_marker_collection_lazy_nonragged(self):
        m = Points(offsets=da.array([[1, 1], [2, 2]]), sizes=(1,))
        assert not isinstance(m.kwargs["offsets"], da.Array)

    def test_append_kwarg(self):
        offsets = np.empty(2, dtype=object)
        for i in range(2):
            offsets[i] = np.array([[1, 1], [2, 2]])
        m = Points(offsets=offsets, sizes=(1,))
        m.append_kwarg(keys="offsets", value=[[0, 1]])
        assert len(m.kwargs["offsets"][0]) == 3

    def test_delete_index(self):
        offsets = np.empty(2, dtype=object)
        for i in range(2):
            offsets[i] = np.array([[1, 1], [2, 2]])
        m = Points(offsets=offsets, sizes=(1,))
        m.delete_index(keys="offsets", index=1)
        assert len(m.kwargs["offsets"][0]) == 1

    def test_rep(self):
        m = Markers(
            offsets=[[1, 1], [2, 2]], verts=3, sizes=3, collection_class=PolyCollection
        )
        assert "Markers" in m.__repr__()

    def test_update_static(self):
        m = Points(offsets=([[1, 1], [2, 2]]), sizes=(1,))
        s = Signal1D(np.ones((10, 10)))
        s.plot()
        s.add_marker(m)
        s.axes_manager.navigation_axes[0].index = 2

    @pytest.mark.parametrize(
        "subclass",
        (
            (Arrows, Quiver, {"offsets": [[1, 1]], "U": [1], "V": [1]}),
            (Circles, EllipseCollection, {"offsets": [[1, 1]], "sizes": [1]}),
            (
                Ellipses,
                EllipseCollection,
                {"offsets": [1, 2], "widths": [1], "heights": [1]},
            ),
            (HorizontalLines, LineCollection, {"offsets": [1, 2]}),
            (Points, EllipseCollection, {"offsets": [[1, 1]], "sizes": [1]}),
            (VerticalLines, LineCollection, {"offsets": [1, 2]}),
            (
                Rectangles,
                RectangleCollection,
                {"offsets": [[1, 1]], "widths": [1], "heights": [1]},
            ),
            (Squares, RectangleCollection, {"offsets": [[1, 1]], "sizes": [1]}),
            (Texts, TextCollection, {"offsets": [[1, 1]], "texts": ["a"]}),
            (Lines, LineCollection, {"segments": [[0, 0], [1, 1]]}),
        ),
    )
    def test_initialize_subclasses(self, subclass):
        m = subclass[0](**subclass[2])
        assert m.collection_class is subclass[1]

    @pytest.mark.parametrize(
        "subclass",
        (
            (Arrows, Quiver, {"offsets": [[1, 1]], "U": [1], "V": [1]}),
            (Circles, CircleCollection, {"offsets": [[1, 1]], "sizes": [1]}),
            (
                Ellipses,
                EllipseCollection,
                {"offsets": [1, 2], "widths": [1], "heights": [1]},
            ),
            (HorizontalLines, LineCollection, {"offsets": [1, 2]}),
            (Points, CircleCollection, {"offsets": [[1, 1]], "sizes": [1]}),
            (VerticalLines, LineCollection, {"offsets": [1, 2]}),
            (
                Rectangles,
                RectangleCollection,
                {"offsets": [[1, 1]], "widths": [1], "heights": [1]},
            ),
            (Squares, RegularPolyCollection, {"offsets": [[1, 1]], "sizes": [1]}),
            (Texts, TextCollection, {"offsets": [[1, 1]], "texts": ["a"]}),
            (Lines, LineCollection, {"segments": [[0, 0], [1, 1]]}),
        ),
    )
    def test_deepcopy(self, subclass):
        m = subclass[0](**subclass[2])
        m2 = deepcopy(m)
        assert m2 is not m
        for key in subclass[2]:
            assert np.all(m2.kwargs[key] == m.kwargs[key])
        assert m2.transform == m.transform
        assert m2.offsets_transform == m.offsets_transform
        assert "transform" not in m.kwargs
        assert "transform" not in m2.kwargs


class TestMarkers2Collection:
    @pytest.fixture
    def iter_data(self):
        data = {
            "x1": np.arange(10),
            "y1": np.arange(10),
            "x2": np.arange(10),
            "y2": np.arange(10),
            "size": np.arange(10),
            "text": np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
        }
        return data

    @pytest.fixture
    def static_data(self):
        data = {
            "x1": 1,
            "y1": 2,
            "x2": 3,
            "y2": 4,
            "text": "a",
            "size": 5,
        }
        return data

    @pytest.fixture
    def static_and_iter_data(self):
        data = {
            "x1": 1,
            "y1": np.arange(10),
            "x2": np.arange(10),
            "y2": 4,
            "text": np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
            "size": np.arange(10),
        }
        return data

    @pytest.fixture
    def signal(self):
        return Signal1D(np.ones((10, 20)))

    @pytest.mark.parametrize(
        "data", ("iter_data", "static_data", "static_and_iter_data")
    )
    @pytest.mark.parametrize(
        "marker_type",
        (
            "Point",
            "HorizontalLineSegment",
            "LineSegment",
            "Ellipse",
            "HorizontalLine",
            "VerticalLine",
            "Arrow",
            "Rectangle",
            "VerticalLineSegment",
            "Text",
        ),
    )
    def test_marker2collection(self, request, marker_type, data, signal):
        d = request.getfixturevalue(data)
        test_dict = {}
        test_dict["data"] = d
        test_dict["marker_type"] = marker_type
        test_dict["marker_properties"] = {"color": "black"}
        test_dict["plot_on_signal"] = True
        markers = markers2collection(test_dict)

        signal.add_marker(
            markers,
        )
        signal.plot()

    @pytest.mark.parametrize(
        "data", ("iter_data", "static_data", "static_and_iter_data")
    )
    @pytest.mark.parametrize("marker_type", "Text")
    def test_marker2collectionfail(self, request, marker_type, data):
        d = request.getfixturevalue(data)
        test_dict = {}
        test_dict["data"] = d
        test_dict["marker_type"] = marker_type
        test_dict["marker_properties"] = {"color": "black"}
        test_dict["plot_on_signal"] = True
        with pytest.raises(ValueError):
            markers2collection(test_dict)

    def test_marker2collection_empty(
        self,
    ):
        assert markers2collection({}) == {}


def _test_marker_collection_close():
    signal = Signal2D(np.ones((10, 10)))
    segments = np.ones((10, 2, 2))
    markers = Markers(LineCollection, segments=segments)
    signal.add_marker(markers)
    return signal


@update_close_figure()
def test_marker_collection_close():
    return _test_marker_collection_close()


class TestMarkersTransform:
    @pytest.mark.parametrize(
        "offsets_transform",
        (
            "data",
            "display",
            "xaxis",
            "yaxis",
            "axes",
            "relative",
        ),
    )
    def test_set_offset_transform(self, offsets_transform):
        markers = Points(
            offsets=[[1, 1], [4, 4]],
            sizes=(10,),
            color=("black",),
            offsets_transform=offsets_transform,
        )
        assert markers.offsets_transform == offsets_transform
        signal = Signal1D((np.arange(100) + 1).reshape(10, 10))

        signal.plot()
        signal.add_marker(markers)
        mapping = {
            "data": CompositeGenericTransform,
            "display": IdentityTransform,
            "xaxis": BlendedGenericTransform,
            "yaxis": BlendedGenericTransform,
            "xaxis_scale": Affine2D,
            "yaxis_scale": Affine2D,
            "axes": BboxTransformTo,
            "relative": CompositeGenericTransform,
        }

        assert isinstance(markers.offsets_transform, mapping[offsets_transform])

    def test_set_plotted_transform(
        self,
    ):
        markers = Points(
            offsets=[[1, 1], [4, 4]],
            sizes=(10,),
            color=("black",),
            transform="display",
            offsets_transform="display",
        )
        assert markers.transform == "display"
        assert markers.offsets_transform == "display"
        signal = Signal1D((np.arange(100) + 1).reshape(10, 10))
        signal.plot()
        signal.add_marker(markers)
        assert isinstance(markers.transform, IdentityTransform)
        assert isinstance(markers.offsets_transform, IdentityTransform)
        markers.transform = "data"
        markers.offsets_transform = "data"
        assert isinstance(markers.transform, CompositeGenericTransform)
        assert isinstance(markers.offsets_transform, CompositeGenericTransform)
        assert markers.collection.get_transform() == markers.transform

    def test_unknown_tranform(self):
        with pytest.raises(ValueError):
            markers = Points(
                offsets=[[1, 1], [4, 4]],
                sizes=(10,),
                color=("black",),
                transform="test",
                offsets_transform="test",
            )


class TestRelativeMarkerCollection:
    def test_relative_marker_collection(self):
        signal = Signal1D((np.arange(100) + 1).reshape(10, 10))
        segments = np.zeros((10, 2, 2))
        segments[:, 1, 1] = 1  # set y values end
        segments[:, 0, 0] = np.arange(10).reshape(10)  # set x values
        segments[:, 1, 0] = np.arange(10).reshape(10)  # set x values

        markers = Lines(segments=segments, transform="relative")
        texts = Texts(offsets=segments[:, 1], texts="a", offsets_transform="relative")
        signal.plot()
        signal.add_marker(markers)
        signal.add_marker(texts)
        signal.axes_manager.navigation_axes[0].index = 1
        segs = markers.collection.get_segments()
        offs = texts.collection.get_offsets()
        assert segs[0][0][0] == 0
        assert segs[0][1][1] == 11
        assert offs[0][1] == 11

    def test_relative_marker_collection_with_shifts(self):
        signal = Signal1D((np.arange(100) + 1).reshape(10, 10))
        segments = np.zeros((10, 2, 2))
        segments[:, 1, 1] = 1  # set y values end
        segments[:, 0, 0] = np.arange(10).reshape(10)  # set x values
        segments[:, 1, 0] = np.arange(10).reshape(10)  # set x values

        markers = Lines(segments=segments,shift=1/9, transform="relative")
        texts = Texts(offsets=segments[:, 1], shift=1/9, texts="a", offsets_transform="relative")
        signal.plot()
        signal.add_marker(markers)
        signal.add_marker(texts)
        signal.axes_manager.navigation_axes[0].index = 1
        segs = markers.collection.get_segments()
        offs = texts.collection.get_offsets()
        assert segs[0][0][0] == 0
        assert segs[0][1][1] == 12
        assert offs[0][1] == 12

class TestLineCollections:
    @pytest.fixture
    def offsets(self):
        d = np.empty((3,), dtype=object)
        for i in np.ndindex(d.shape):
            d[i] = np.arange(i[0] + 1)
        return d

    def test_vertical_line_collection(self, offsets):
        vert = VerticalLines(offsets=offsets)
        s = Signal2D(np.zeros((3, 3, 3)))
        s.axes_manager.signal_axes[0].offset = 0
        s.axes_manager.signal_axes[1].offset = 0
        s.plot(interpolation=None)
        s.add_marker(vert)
        kwargs = vert.get_data_position()
        # Offsets --> segments for vertical lines
        np.testing.assert_array_equal(kwargs["segments"], [[[0.0, 0], [0.0, 1]]])

    def test_horizontal_line_collection(self, offsets):
        hor = HorizontalLines(offsets=offsets)
        s = Signal2D(np.zeros((3, 3, 3)))
        s.axes_manager.signal_axes[0].offset = 0
        s.axes_manager.signal_axes[1].offset = 0
        s.plot(interpolation=None)
        s.add_marker(hor)
        kwargs = hor.get_data_position()
        np.testing.assert_array_equal(kwargs["segments"], [[[0, 0], [1, 0]]])

    def test_horizontal_vertical_line_error(self, offsets):
        with pytest.raises(ValueError):
            hor = HorizontalLines(offsets=offsets, transform="data")
        with pytest.raises(ValueError):
            vert = VerticalLines(offsets=offsets, transform="data")

def test_marker_collection_close_render():
    signal = Signal2D(np.ones((2, 10, 10)))
    markers = Points(offsets=[[1, 1], [4, 4]], sizes=(10,), color=("black",))
    signal.plot()
    signal.add_marker(markers, render_figure=True)
    markers.close(render_figure=True)


class TestMarkers:
    @pytest.fixture
    def offsets(self):
        d = np.empty((3,), dtype=object)
        for i in np.ndindex(d.shape):
            d[i] = np.stack([np.arange(3), np.ones(3) * i], axis=1)
        return d

    @pytest.fixture
    def extra_kwargs(self):
        widths = np.empty((3,), dtype=object)
        for i in np.ndindex(widths.shape):
            widths[i] = np.ones(3)

        heights = np.empty((3,), dtype=object)
        for i in np.ndindex(heights.shape):
            heights[i] = np.ones(3)
        angles = np.empty((3,), dtype=object)
        for i in np.ndindex(angles.shape):
            angles[i] = np.ones(3)

        kwds = {
            Points: {"sizes": (1,)},
            Circles: {"sizes": (1,)},
            Arrows: {"U": 1, "V": 1},
            Ellipses: {"widths": widths, "heights": heights, "angles": angles},
        }
        return kwds

    @pytest.fixture
    def signal(self):
        return Signal2D(np.ones((3, 10, 10)))

    @pytest.mark.parametrize("MarkerClass", [Points, Circles, Ellipses, Arrows])
    def test_offsest_markers(self, extra_kwargs, MarkerClass, offsets, signal):
        markers = MarkerClass(offsets=offsets, **extra_kwargs[MarkerClass])
        signal.plot()
        signal.add_marker(markers)
        signal.axes_manager.navigation_axes[0].index = 1

    def test_arrows(self, signal):
        arrows = Arrows(offsets=[[1, 1], [4, 4]], U=1, V=1, C=(2, 2))
        signal.plot()
        signal.add_marker(arrows)
        signal.axes_manager.navigation_axes[0].index = 1

    @pytest.mark.parametrize("MarkerClass", [Points, Circles, Ellipses, Arrows])
    def test_markers_length_offsets(self, extra_kwargs, MarkerClass, offsets, signal):
        markers = MarkerClass(offsets=offsets, **extra_kwargs[MarkerClass])
        assert len(markers) == 3

        signal.plot()
        signal.add_marker(markers)

        assert len(markers) == 3
