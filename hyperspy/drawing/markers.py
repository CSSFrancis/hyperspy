# -*- coding: utf-8 -*-
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

import numpy as np
import dask.array as da
import logging
from copy import deepcopy

import matplotlib.collections as mpl_collections
from matplotlib.transforms import Affine2D, IdentityTransform
from matplotlib.patches import Patch

from hyperspy.events import Event, Events
from hyperspy.misc.array_tools import _get_navigation_dimension_chunk_slice
from hyperspy.misc.utils import isiterable
import hyperspy.external.matplotlib.collections as hs_mpl_collections


_logger = logging.getLogger(__name__)


def convert_positions(peaks, signal_axes):
    new_data = np.empty(peaks.shape[:-1] + (len(signal_axes),))
    for i, ax in enumerate(signal_axes[::-1]):
        # indexes need to be reversed
        new_data[..., (-i - 1)] = ax.scale * peaks[..., i] + ax.offset
    return new_data


class Markers:
    """
    A set of markers for faster plotting.

    This is a generic class which is subclassed by other marker classes. Mostly
    by defining the collection_class.

    In most cases each marker is defined by some keyword argument and
    a (n,2) array of offsets which define the position for each marker
    in the plot.

    For example if we wanted to define a set of ellipses with constant
    height, width, and size we can do the following.

    >>>from matplotlib.collections import EllipseCollection
    >>>import hyperspy.api as hs
    >>>import hyperspy.api as hs
    >>>import numpy as np
    >>>m = hs.plot.marker.Markers(collection_class=EllipseCollection, widths=(2,),
    >>>                 heights=(1,), angles=(1,), units="xy", offsets=np.random.rand(10,2)*10)
    >>>s = hs.signals.Signal2D(np.ones((10,10,10,10)))
    >>>s.plot()
    >>>s.add_marker(m)

    To define a non-static marker any kwarg that can be set with the `matplotlib.collections.set`
    method can be passed as an array with `dtype=object` and the same size as the navigation axes
    for a signal.
    """

    marker_type = "Markers"

    def __init__(
        self,
        collection_class,
        offsets_transform="data",
        transform="display",
        shift=None,
        **kwargs,
    ):
        """
        Initialize a Marker Collection.

        Parameters
        ----------
        collection_class: matplotlib.collections or str
            A Matplotlib collection to be initialized.
        offsets : [n, 2]
            Positions of the markers

        offsets_transform: str
            Define the transformation used for the `offsets`. This only operates on the offset point so it won't
            scale the size of the ``Path``.  It can be one of the following:
            - ``"data"``: the offsets are defined in data coordinates and the ``ax.transData`` transformation is used.
            - ``"relative"``: The offsets are defined in data coordinates in x and coordinates in y relative to the
              data plotted. Only for 1D figure.
            - ``"axes"``: the offsets are defined in axes coordinates and the ``ax.transAxes`` transformation is used.
              (0, 0) is bottom left of the axes, and (1, 1) is top right of the axes.
            - ``"xaxis"``: The offsets are defined in data coordinates in x and axes coordinates in y direction; use
              :py:meth:`matplotlib.axes.Axes.get_xaxis_transform` transformation.
            - ``"yaxis"``: The offsets are defined in data coordinates in y and axes coordinates in x direction; use
              :py:meth:`matplotlib.axes.Axes.get_xaxis_transform` transformation..
            - ``"display"``: the offsets are not transformed, i.e. are defined in the display coordinate system.
              (0, 0) is the bottom left of the window, and (width, height) is top right of the output in "display units"
              :py:class:`matplotlib.transforms.IndentityTransform`.

            transform: str
            Define the transformation to be applied to each marker. This operates on the ``Path`` rather than
            the position so can be used to scale as well as translate some marker.
            It can be any of the transforms in offset_transform or one of the following:
            - ``"xaxis_scale"``: The size of the marker is scaled by scale factor of the x axis.
            - ``"yaxis_scale"``: The size of the marker is scaled by scale factor of the y axis.
            For most cases having transform="data"/"axis" and offsets_transform="data" will result
            in unexpected behavior as the position of the marker will be transformed twice.

        **kwargs :
            Keyword arguments passed to the underlying marker collection. Any argument
            that is array-like and has `dtype=object` is assumed to be an iterating
            argument and is treated as such.

        Examples
        --------

        Adding a marker using an EllipseCollection

        >>>from matplotlib.collections import EllipseCollection
        >>>import hyperspy.api as hs
        >>>import numpy as np
        >>>m = hs.plot.markers.Markers(collection_class=EllipseCollection, widths=(2,),
        ...                 heights=(1,), angles=(1,), units="xy", offsets=np.random.rand(10,2)*10)
        >>>s = hs.signals.Signal2D(np.ones((10,10,10,10)))
        >>>s.plot()
        >>>s.add_marker(m)

        Adding a marker using a PatchCollection (Making a circle)

        >>>from matplotlib.collections import PatchCollection
        >>>from matplotlib.patches import Circle
        >>>import hyperspy.api as hs
        >>>import numpy as np
        >>>m = hs.plot.markers.Markers(collection_class=PatchCollection,
        ...                                     patches=[Circle((0,0), 1)], offsets=np.random.rand(10,2)*10)
        >>>s = hs.signals.Signal2D(np.ones((10,10,10,10)))
        >>>s.plot()
        >>>s.add_marker(m)

        Adding a series of lines using a MarkerCollection

        >>>from matplotlib.collections import LineCollection
        >>>import hyperspy.api as hs
        >>>import numpy as np
        >>>m = hs.plot.markers.Markers(collection_class=LineCollection,
        ...                                     segments=np.random.rand(10,2,2)*10,)
        >>>s = hs.signals.Signal2D(np.ones((10,10,10,10)))
        >>>s.plot()
        >>>s.add_marker(m)

        Adding a marker using PolyCollection (Making a square)

        >>>from matplotlib.collections import PolyCollection
        >>>import hyperspy.api as hs
        >>>import numpy as np
        >>>m = hs.plot.markers.Markers(collection_class=PolyCollection,
        ...                                     offsets=np.random.rand(10,2)*10,
        ...                                     verts=np.array([[[0,0],[0,1],[1,1],[1,0]]]),color="red")
        >>>s = hs.signals.Signal2D(np.ones((10,10,10,10)))
        >>>s.plot()
        >>>s.add_marker(m)

        """
        if isinstance(collection_class, str):
            try:
                # Remove when external changes are merged
                if collection_class in [
                    "TextCollection",
                    "RegularPolyCollection",
                    "EllipseCollection",
                    "RectangleCollection",
                ]:
                    collection_class = getattr(hs_mpl_collections, collection_class)
                elif collection_class is "Quiver":
                    from matplotlib.quiver import Quiver as collection_class
                else:
                    collection_class = getattr(mpl_collections, collection_class)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "The argument `collection_class` must be a string or"
                    " a matplotlib.collections class. matplotlib.collections."
                    + collection_class
                    + " is not a valid matplotlib.collections class."
                )

        if "matplotlib.collections" not in str(
            collection_class
        ) and "matplotlib.quiver" not in str(collection_class):
            raise ValueError(
                "The argument `collection_class` must be a class in "
                "`matplotlib.collection."
            )

        if offsets_transform is "data" and transform in ["axes","data"]:
            raise ValueError(
                "Weird behavior will occur when argument `offsets_transform` cannot be 'data' when "
                "transform is 'axes' or 'data'."
            )
        # Data attributes
        self.kwargs = kwargs  # all keyword arguments.
        self.axes_manager = None
        self.ax = None
        self.auto_update = True
        self._offsets_transform = None
        self._transform = None

        # Handling dask arrays
        self.dask_kwargs = {}
        for key, value in self.kwargs.items():
            if isinstance(value, da.Array) and value.dtype == object:
                self.dask_kwargs[key] = self.kwargs[key]
            elif isinstance(value, da.Array):  # and value.dtype != object:
                self.kwargs[key] = value.compute()
            # Patches or verts shouldn't be cast to array
            elif (
                isinstance(value, list)
                and len(value) > 0
                and not isinstance(value[0], Patch)
                and not key == "verts"
            ):
                self.kwargs[key] = np.array(value)
            elif isinstance(value, list) and len(value) == 0:
                self.kwargs[key] = np.array(value)

            if key in ["sizes", "color"] and (
                not hasattr(value, "__len__") or isinstance(value, str)
            ):
                self.kwargs[key] = (value,)
            if key in ["s"] and isinstance(value, str):
                self.kwargs[key] = (value,)

        self._cache_dask_chunk_kwargs = {}
        self._cache_dask_chunk_kwargs_slice = {}

        self.name = self.marker_type
        # Properties
        self.collection = None
        self.collection_class = collection_class
        self.signal = None
        self.temp_signal = None
        self._plot_on_signal = True
        self.shift = shift
        self.plot_marker = True
        self.offsets_transform = offsets_transform
        self.transform = transform

        # Events
        self.events = Events()
        self.events.closed = Event(
            """
            Event triggered when a marker is closed.

            Arguments
            ---------
            marker : Marker
                The marker that was closed.
            """,
            arguments=["obj"],
        )
        self._closing = False

    @property
    def _is_iterating(self):
        if self._transform == "relative" or self._offsets_transform == "relative":
            return False
        elif self._plot_on_signal:
            return np.any([is_iterating(value) for key, value in self.kwargs.items()])
        else:  # currently iterating navigation markers arenot supported
            return False

    @property
    def offsets_transform(self):
        return self._get_transform(attr="_offsets_transform")

    @offsets_transform.setter
    def offsets_transform(self, value):
        self._set_transform(value, attr="_offsets_transform")
        # setting live plot
        if self.collection is not None and self.ax is not None:
            self.collection.set_offset_transform(self.offsets_transform)
            self.update()

    def _get_transform(self, attr="_transform"):
        if self.ax is not None:  # return the transform
            transforms = {
                "data": self.ax.transData,
                "axes": self.ax.transAxes,
                "display": IdentityTransform(),
                "yaxis": self.ax.get_yaxis_transform(),
                "xaxis": self.ax.get_xaxis_transform(),
                "relative": self.ax.transData,
            }
            if attr == "_transform":
                transforms["xaxis_scale"] = Affine2D().scale(
                    self.ax.bbox.width / self.ax.viewLim.width
                )
                transforms["yaxis_scale"] = Affine2D().scale(
                    self.ax.bbox.height / self.ax.viewLim.height
                )
            return transforms[getattr(self, attr)]
        else:  # return the string value
            return getattr(self, attr)

    def _set_transform(self, value, attr="_transform"):
        if value not in [
            "data",
            "axes",
            "xaxis",
            "yaxis",
            "display",
            "relative",
            "xaxis_scale",
            "yaxis_scale",
        ]:
            raise ValueError(
                "The offset transform must be one of 'data', 'axes',"
                " 'xaxis', 'yaxis', 'display', 'relative'"
            )
        setattr(self, attr, value)

    @property
    def transform(self):
        return self._get_transform(attr="_transform")

    @transform.setter
    def transform(self, value):
        self._set_transform(value, attr="_transform")
        # setting live plot
        if self.collection is not None and self.ax is not None:
            self.collection.set_transform(self.transform)
            self.update()

    def __len__(self):
        """Return the number of markers in the collection."""

        # LineSegments doesn't have "offsets" but "segments"
        key = "offsets" if "offsets" in self.kwargs.keys() else "segments"
        return self.kwargs[key].shape[0]

    def delete_index(self, keys, index):
        """
        Delete the index from the kwargs.

        Parameters
        ----------
        keys: list
            List of keys to delete from.
        index: slice, int or array of ints
            Indicate indices of sub-arrays to remove along the specified axis.
        """
        if isinstance(keys, str):
            keys = [
                keys,
            ]
        for key in keys:
            if self.kwargs[key].dtype == object:
                for i in np.ndindex(self.kwargs[key].shape):
                    self.kwargs[key][i] = np.delete(self.kwargs[key][i], index, axis=0)
            else:
                self.kwargs[key] = np.delete(self.kwargs[key], index, axis=0)

    def append_kwarg(self, keys, value):
        """
        Add the index from the kwargs.

        Parameters
        ----------
        keys: list
            List of keys to append.
        value:
            The value to append to the kwarg.
        """
        if isinstance(keys, str):
            keys = [
                keys,
            ]
        # skip if value is empty
        if len(value) == 0:
            return
        for key in keys:
            if self.kwargs[key].dtype == object:
                for i in np.ndindex(self.kwargs[key].shape):
                    self.kwargs[key][i] = np.append(self.kwargs[key][i], value, axis=0)
            else:
                self.kwargs[key] = np.append(self.kwargs[key], value, axis=0)

    def _get_chunk_slice(self, key, index_slice):
        """
        Get the slice for a chunk of data.

        Parameters
        ----------
        key: str
            The key to get the slice for.
        index_slice: slice, int or array of ints
            Indicate indices of sub-arrays to remove along the specified axis.
        """
        if self.kwargs[key].dtype == object:
            return index_slice
        else:
            return index_slice

    def _get_cache_dask_kwargs_chunk(self, indices):
        """
        Get the kwargs at some index.  If the index is cached return the cached value
        otherwise compute the kwargs and cache them.
        """
        chunks = {key: value.chunks for key, value in self.dask_kwargs.items()}
        chunk_slices = {
            key: _get_navigation_dimension_chunk_slice(indices, chunk)
            for key, chunk in chunks.items()
        }
        to_compute = {}
        for key, value in self.dask_kwargs.items():
            index_slice = chunk_slices[key]
            current_slice = self._cache_dask_chunk_kwargs_slice.get(key, None)
            if current_slice is None or current_slice != index_slice:
                to_compute[key] = value[index_slice]
                self._cache_dask_chunk_kwargs_slice[key] = index_slice

        if len(to_compute) > 0:
            # values = da.compute([value for value in to_compute.values()])
            # self._cache_dask_chunk_kwargs.update({key: value for
            #                                      key, value in zip(to_compute.keys(), values)})
            for key in to_compute:  # this should be one da.compute() function...
                self._cache_dask_chunk_kwargs[key] = to_compute[key].compute()

        out_kwargs = {}
        for key, value in self._cache_dask_chunk_kwargs.items():
            temp_indices = list(indices)
            for i, temp_slice in enumerate(chunk_slices[key]):
                # add offset to the indices
                temp_indices[i] -= temp_slice.start
            temp_indices = tuple(temp_indices)
            out_kwargs[key] = self._cache_dask_chunk_kwargs[key][temp_indices]
        return out_kwargs

    def __repr__(self):
        return f"<{self.marker_type}| Iterating == {self._is_iterating}>"

    @classmethod
    def from_signal(
        cls,
        signal,
        key="offsets",
        signal_axes="metadata",
        **kwargs,
    ):
        """
        Initialize a marker collection from a hyperspy Signal.

        Parameters
        ----------
        signal: BaseSignal
            A value passed to the Collection as {key:signal.data}
        key: str
            The key used to create a key value pair to
            create the Collection. Passed as {key: signal.data}.
        collection_class: None or matplotlib.collections
            The collection which is initialized
        signal_axes: str, tuple of UniformAxes or None
            If "metadata" look for signal_axes saved in metadata under .metadata.Peaks.signal_axes
            and convert from pixel positions to real units before creating the collection. If a tuple
            of signal axes those Axes will be used otherwise no transformation will
            happen.
        """
        if signal_axes is None or (
            signal_axes == "metadata"
            and not signal.metadata.has_item("Peaks.signal_axes")
        ):
            new_signal = signal
        elif signal_axes == "metadata" and signal.metadata.has_item(
            "Peaks.signal_axes"
        ):
            new_signal = signal.map(
                convert_positions,
                inplace=False,
                ragged=True,
                output_dtype=object,
                signal_axes=signal.metadata.Peaks.signal_axes,
            )
        elif isinstance(signal_axes, (tuple, list)):
            new_signal = signal.map(
                convert_positions,
                inplace=False,
                ragged=True,
                output_dtype=object,
                signal_axes=signal_axes,
            )
        else:
            raise ValueError(
                "The keyword argument `signal_axes` must be one of 'metadata' a"
                "tuple of `DataAxes` or None"
            )
        kwargs[key] = new_signal.data
        return cls(**kwargs)

    def _get_data_shape(self):
        for key, item in self.kwargs.items():
            if is_iterating(item):
                return item.shape
        return ()

    def __deepcopy__(self, memo):
        new_marker = markers2collection(self._to_dictionary())
        return new_marker

    def _to_dictionary(self):
        marker_dict = {
            "marker_type": self.marker_type,
            "collection_class": self.collection_class.__name__,
            "plot_on_signal": self._plot_on_signal,
            "offsets_transform": self._offsets_transform,
            "transform": self._transform,
            "kwargs": self.kwargs,
        }
        return marker_dict

    def get_data_position(
        self,
        get_static_kwargs=True,
        get_modified=True
    ):
        """
        Return the current keyword arguments for updating the collection.

        Parameters
        ----------
        get_static_kwargs: bool, optional, default True
            If True, will return the all of the kwargs including those that are not iterating.
            If False, will only return the kwargs that are iterating.
        get_modified: bool, optional, default True
            Get the modified kwargs.  This is useful for operations that change some of the kwargs
            before plotting.  For example, if you want to scale the marker position by the current
            data value using the "relative" transform.
        """
        current_keys = {}
        if self._is_iterating:
            indices = self.axes_manager.indices[::-1]
            for key, value in self.kwargs.items():
                if is_iterating(value):
                    if key not in self.dask_kwargs:
                        val = value[indices]
                        # some keys values need to iterate
                        if key in ["sizes", "color"] and not hasattr(val, "__len__"):
                            val = (val,)
                        current_keys[key] = val
                elif get_static_kwargs:
                    current_keys[key] = value
                else:  # key already set in init
                    pass
            if len(self.dask_kwargs) > 0:
                current_keys.update(self._get_cache_dask_kwargs_chunk(indices))
        else:
            current_keys = self.kwargs
        if get_modified:
            # Handling relative markers
            if (
                self._offsets_transform == "relative" or self._transform == "relative"
            ):  # scale based on current data
                if "offsets" in current_keys:
                    current_keys = self._scale_kwarg(current_keys, "offsets")
                if "segments" in current_keys:
                    current_keys = self._scale_kwarg(current_keys, "segments")
        return current_keys

    def _scale_kwarg(self, kwds, key):
        """
        Scale the kwarg by the current data.  This is useful for scaling the
        marker position by the current index or data value.

        When self.reference is "data" the kwarg is scaled by the current data value of the
        "offset" or "segments" key

        When self.reference is "data_index" the kwarg is scaled by the current data value of the
        "offset" or "segments" key and the given value of the index.  This is useful when you want
        to scale things by some value in the data that is not the same value.
        """
        new_kwds = deepcopy(kwds)
        current_data = self.temp_signal(as_numpy=True)
        x_positions = new_kwds[key][..., 0]
        ax = self.axes_manager.signal_axes[0]
        indexes = np.round((x_positions - ax.offset) / ax.scale).astype(int)
        y_positions = new_kwds[key][..., 1]
        new_y_positions = current_data[indexes] * y_positions

        if self.shift is not None:
            yrange = np.max(current_data) - np.min(current_data)
            new_y_positions = new_y_positions + self.shift * yrange
        new_kwds[key][..., 1] = new_y_positions
        return new_kwds

    def update(self):
        if (
            self._transform == "relative"
            or self._offsets_transform == "relative"
            or self._is_iterating
        ):
            kwds = self.get_data_position(get_static_kwargs=False)
            self.collection.set(**kwds)

    def _initialize_collection(self):
        self.collection = self.collection_class(
            **self.get_data_position(),
            offset_transform=self.offsets_transform,
        )
        self.collection.set_transform(self.transform)

    def plot(self, render_figure=True):
        """
        Plot a marker which has been added to a signal.

        Parameters
        ----------
        render_figure : bool, optional, default True
            If True, will render the figure after adding the marker.
            If False, the marker will be added to the plot, but will the figure
            will not be rendered. This is useful when plotting many markers,
            since rendering the figure after adding each marker will slow
            things down.
        """
        if self.ax is None:
            raise AttributeError(
                "To use this method the marker needs to be first add to a "
                + "figure using `s._plot.signal_plot.add_marker(m)` or "
                + "`s._plot.navigator_plot.add_marker(m)`"
            )
        self._initialize_collection()
        self.collection.set_animated(self.ax.figure.canvas.supports_blit)
        self.ax.add_collection(self.collection)
        if render_figure:
            self._render_figure()

    def _render_figure(self):
        self.ax.hspy_fig.render_figure()

    def close(self, render_figure=True):
        """Remove and disconnect the marker.

        Parameters
        ----------
        render_figure : bool, optional, default True
            If True, the figure is rendered after removing the marker.
            If False, the figure is not rendered after removing the marker.
            This is useful when many markers are removed from a figure,
            since rendering the figure after removing each marker will slow
            things down.
        """
        if self._closing:
            return
        self._closing = True
        self.collection.remove()
        self.events.closed.trigger(obj=self)
        self.temp_signal = None
        for f in self.events.closed.connected:
            self.events.closed.disconnect(f)
        if render_figure:
            self._render_figure()


def is_iterating(arg):
    return isinstance(arg, (np.ndarray, da.Array)) and arg.dtype == object


def dict2vector(data, keys=None, return_size=True, dtype=float):
    """Take some dictionary of values and create offsets based on the input keys.
    For instances like creating a horizontal or vertical line then some key is duplicated.

    Multiple keys can be passed as well. For example to define a rectangle:

    >>> dict2offsets(data,keys= [['x1','y1'], ['x2','y1'], ['x2', 'y2'],['x1', 'y2']])

    In this example the keys will be unpacked to create a rectangle.
    """
    if keys is None:
        keys = [["x1", "x2"]]
    keys = np.array(keys)
    # check to see if the array should be ragged
    unique_keys = np.unique(keys)
    is_key_iter = [
        isiterable(data[key]) and not isinstance(data[key], str) for key in unique_keys
    ]
    if not any(is_key_iter):  # no iterable keys
        if dtype is str:
            dtype = object
        vector = np.empty(keys.shape, dtype=dtype)
        for i in np.ndindex(
            keys.shape
        ):  # iterate through keys and create resulting vector
            vector[i] = data[keys[i]]
        if dtype is object:
            vector = np.asarray(vector, dtype=str)
    else:
        iter_key = unique_keys[is_key_iter][0]
        nav_shape = data[iter_key].shape
        if not all(is_key_iter):  # only some values are iterating
            non_iterkeys = unique_keys[np.logical_not(is_key_iter)]
            for k in non_iterkeys:
                data[k] = np.full(shape=nav_shape, fill_value=data[k])
        vector = np.empty(nav_shape, dtype=object)  # Create ragged array
        for i in np.ndindex(nav_shape):
            if dtype is str:
                vect = []
                for j in np.ndindex(keys.shape):
                    vect.append(data[keys[j]][i])
                vect = np.array(vect)
            else:
                vect = np.empty(keys.shape, dtype=dtype)
                for j in np.ndindex(keys.shape):
                    vect[j] = data[keys[j]][i]
            vector[i] = vect
    if return_size:
        if not isiterable(data["size"]):
            size = data["size"]
        else:
            size = np.empty(data["size"].shape, dtype=object)
            for i in np.ndindex(data["size"].shape):
                size[i] = data["size"][i]
        return vector, size
    else:
        return vector


def markers2collection(marker_dict):
    """This function maps a marker dict to a MarkerCollection class.

    This provides continuity from markers saved with hyperspy 1.x.x and hyperspy 2.x.x.

    Warning: This function is not complete and will transfer all of the marker properties.
    """
    from hyperspy.utils.markers import (
        Arrows,
        Ellipses,
        Circles,
        HorizontalLines,
        Lines,
        Points,
        Polygons,
        Rectangles,
        Squares,
        Texts,
        VerticalLines,
    )

    marker_mapping = {
        "Arrows": Arrows,
        "Ellipses": Ellipses,
        "Circles": Circles,
        "HorizontalLines": HorizontalLines,
        "Lines": Lines,
        "Points": Points,
        "Polygons": Polygons,
        "Rectangles": Rectangles,
        "Squares": Squares,
        "Texts": Texts,
        "VerticalLines": VerticalLines,
    }

    from matplotlib.collections import PolyCollection

    if len(marker_dict) == 0:
        return {}
    marker_type = marker_dict.pop("marker_type")
    plot_on_signal = marker_dict.pop("plot_on_signal")
    offsets_transform = marker_dict.pop("offsets_transform", None)
    transform = marker_dict.pop("transform", None)

    if marker_type == "Point":
        offsets, size = dict2vector(marker_dict["data"], keys=None, return_size=True)
        marker = Points(offsets=offsets, sizes=size, **marker_dict["marker_properties"])
        marker
    elif marker_type == "HorizontalLine":
        offsets = dict2vector(marker_dict["data"], keys=["y1"], return_size=False)
        marker = HorizontalLines(offsets=offsets, **marker_dict["marker_properties"])

    elif marker_type == "HorizontalLineSegment":
        segments = dict2vector(
            marker_dict["data"], keys=[[["x1", "y1"], ["x2", "y1"]]], return_size=False
        )

        marker = Lines(segments=segments, **marker_dict["marker_properties"])
    elif marker_type == "LineSegment":
        segments = dict2vector(
            marker_dict["data"], keys=[[["x1", "y1"], ["x2", "y2"]]], return_size=False
        )
        marker = Lines(segments=segments, **marker_dict["marker_properties"])
    elif marker_type == "Arrow":
        offsets = dict2vector(
            marker_dict["data"],
            keys=[
                ["x1", "y1"],
            ],
            return_size=False,
        )

        dx = dict2vector(
            marker_dict["data"],
            keys=[
                ["x2"],
            ],
            return_size=False,
        )
        dy = dict2vector(
            marker_dict["data"],
            keys=[
                ["y2"],
            ],
            return_size=False,
        )

        marker = Arrows(offsets, dx, dy, **marker_dict["marker_properties"])

    elif marker_type == "Rectangle":
        verts = dict2vector(
            marker_dict["data"],
            keys=[
                [["x1", "y1"], ["x2", "y1"], ["x2", "y2"], ["x1", "y2"]],
            ],
            return_size=False,
        )
        marker = Markers(
            collection_class=PolyCollection,
            verts=verts,
            **marker_dict["marker_properties"],
        )
    elif marker_type == "Ellipse":
        offsets = dict2vector(
            marker_dict["data"],
            keys=[
                ["x1", "y1"],
            ],
            return_size=False,
        )

        width = dict2vector(marker_dict["data"], keys=["x2"], return_size=False)
        height = dict2vector(marker_dict["data"], keys=["y2"], return_size=False)
        marker = Ellipses(
            offsets=offsets,
            widths=width,
            heights=height,
            **marker_dict["marker_properties"],
        )
    elif marker_type == "Text":
        offsets = dict2vector(
            marker_dict["data"], keys=[["x1", "y1"]], return_size=False
        )
        texts = dict2vector(
            marker_dict["data"], keys=["text"], return_size=False, dtype=str
        )
        marker = Texts(offsets=offsets, texts=texts, **marker_dict["marker_properties"])
    elif marker_type == "VerticalLine":
        x = dict2vector(marker_dict["data"], keys=["x1"], return_size=False)

        marker = VerticalLines(offsets=x, **marker_dict["marker_properties"])
    elif marker_type == "VerticalLineSegment":
        segments = dict2vector(
            marker_dict["data"], keys=[[["x1", "y1"], ["x1", "y2"]]], return_size=False
        )

        marker = Lines(
            segments=segments,
            **marker_dict["marker_properties"],
        )
    elif marker_type in marker_mapping:
        marker = marker_mapping[marker_type](
            transform=transform,
            offsets_transform=offsets_transform,
            **marker_dict["kwargs"],
        )

    elif marker_type == "Markers":
        marker = Markers(
            collection_class=marker_dict["collection_class"],
            transform=transform,
            offsets_transform=offsets_transform,
            **marker_dict["kwargs"],
        )
    else:
        raise ValueError(
            f"The marker_type: {marker_type} is not a hyperspy.marker class "
            f"and cannot be converted to a MarkerCollection"
        )
    marker.plot_on_signal = plot_on_signal
    return marker
