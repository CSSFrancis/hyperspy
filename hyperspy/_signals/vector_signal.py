# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np

from hyperspy.signal import BaseSignal
from hyperspy._signals.lazy import LazySignal
from hyperspy.drawing._markers.point import Point
from hyperspy.roi import _get_mpl_ax
from hyperspy.axes import VectorDataAxis


class BaseVectorSignal(BaseSignal):
    """A generic class for a ragged signal representing a set of vectors.
    """

    def __init__(self, *args, **kwargs):
        if "ragged" not in kwargs:
            kwargs["ragged"] = True
        super().__init__(*args, **kwargs)
        self.vector=True

    def nav_to_vector(self, inplace=True, axis=None):
        """Converts the navigation positions to vectors. Useful if you want to
        preform additional vector operations rather than a ragged array. If lazy
        this will compute the vectors.

        Parameters
        ----------
        axes: None, tuple
            The navigation axes to be converted to vectors
        """

        if inplace:
            vectors = self
        else:
            vectors = self.deepcopy()
        if axis is None:
            axis = vectors.axes_manager.navigation_axes
        else:
            axis = vectors.axes_manager[axis]
        if not isinstance(axis, tuple):
            axis = (axis,)

        v_ind = np.isin(vectors.axes_manager.navigation_axes, axis)
        n_ind = np.invert(v_ind)
        vector_shape = tuple(np.array(self.data.shape)[v_ind])
        new_shape = tuple(np.array(self.data.shape)[n_ind])
        if len(new_shape) == 0:
            new_shape = 1
        new = np.empty(shape=new_shape, dtype=object)
        for ind in np.ndindex(new_shape):
            vector_indexes = np.array(list(np.ndindex(tuple(vector_shape))))
            for i, pos in zip(ind, np.argwhere(n_ind)):
                vector_indexes = np.insert(vector_indexes, pos, i, axis=1)
            new[ind] = [
                np.append(ind[v_ind], v)
                for ind in vector_indexes
                for v in self.data[ind]
            ]
        vectors.data = new
        for a in axis:
            if not isinstance(a, VectorDataAxis):
                a.convert_to_vector_axis()
            a.navigate = False
        return vectors

    def get_real_vectors(
        self, axis=None, real_units=True, flatten=False
    ):
        """Returns the vectors in real units. Using the scale and offset as determined by the axes manager.
        If navigation axes are included the navigation axes will be transformed into vectors.

        Parameters
        ----------
        axis: None, int
            If None then the vectors for all axes will be returned. Otherwise, only the real units for
            the specified axes will be returned.
        flatten: bool
            If the vectors should be returned as one list or as a set of objects.
        pixel_units: bool
            Returns the vectors in pixel units rather than using the axes manager.
        """
        if axis is None:
            nav_axis = self.axes_manager.navigation_axes
            sig_axis = self.axes_manager.signal_axes
        else:
            if not np.iterable(axis):
                axis = tuple(axis)
            axis = self.axes_manager[axis]
            nav_axis = [a for a in axis if a.navigate]
            sig_axis = [a for a in axis if not a.navigate]
        nav_indexes = tuple([a.index_in_array for a in nav_axis])
        sig_indexes = tuple([a.index_in_vector_array for a in sig_axis])
        navigate = len(nav_indexes) > 0
        if not real_units:
            sig_scales = np.ones(len(sig_indexes))
            sig_offsets = np.zeros(len(sig_indexes))
        else:
            sig_scales = [self.axes_manager.signal_axes[i].scale for i in sig_indexes]
            sig_offsets = [self.axes_manager.signal_axes[i].offset for i in sig_indexes]
        if not navigate:  # Only vector axes
            real_vector = np.empty(self.data.shape, dtype=object)
            for ind in np.ndindex(self.data.shape):
                vectors = self.data[ind][:, sig_indexes]
                real = vectors * sig_scales + sig_offsets
                real_vector[ind] = real
        elif len(sig_scales) == 0:  # Only navigation axes
            nav_positions = self._get_navigation_positions()
            # check to see if we need to find the navigation axis index...
            nav_positions = nav_positions[..., nav_indexes]
            real_vector = np.empty(self.data.shape, dtype=object)
            for ind in np.ndindex(self.data.shape):
                vectors = self.data[ind]
                real_vector[ind] = np.array([nav_positions[ind] for v in vectors])
        else:  # Both navigation and vector axes...
            real_vector = np.empty(self.data.shape, dtype=object)
            nav_positions = self._get_navigation_positions(flatten=True)
            nav_positions = nav_positions[..., nav_indexes]
            for ind, nav_pos in zip(np.ndindex(self.data.shape), nav_positions):
                vectors = self.data[ind][:, sig_indexes]
                real = vectors * sig_scales + sig_offsets
                real_vector[ind] = np.array([np.append(nav_pos, v) for v in real])
        if flatten:  # Unpack object into one array
            real_vector = np.array(
                [v for ind in np.ndindex(self.data.shape) for v in real_vector[ind]]
            )

        return real_vector

    def flatten(self, inplace=True):
        """ Converts selected navigation axes to vectors.

        Parameters
        ----------
        inplace

        Returns
        -------

        """
        if inplace:
            vectors = self
        else:
            vectors = self.deepcopy()
        new_vectors = self.get_real_vectors(real_units=False, flatten=True)
        for axis in vectors.axes_manager._axes:
            if not isinstance(axis, VectorDataAxis):
                axis.convert_to_vector_axis()
            axis.navigate = False
        new = np.empty(1, dtype=object)
        new[0] = new_vectors
        vectors.data = new
        return vectors

    def _get_navigation_positions(self, flatten=False, real_units=True):
        nav_indexes = np.array(list(np.ndindex(self.axes_manager._navigation_shape_in_array)))
        np.zeros(shape=self.axes_manager.navigation_shape)
        if not real_units:
            scales = [1 for a in self.axes_manager.navigation_axes]
            offsets = [0 for a in self.axes_manager.navigation_axes]
        else:
            scales = [a.scale for a in self.axes_manager.navigation_axes]
            offsets = [a.offset for a in self.axes_manager.navigation_axes]

        if flatten:
            real_nav = np.array(
                [
                    np.array(ind) * scales + offsets
                    for ind in np.array(list(nav_indexes))
                ]
            )
        else:
            real_nav = np.reshape(
                [
                    np.array(ind) * scales + offsets
                    for ind in np.array(list(nav_indexes))
                ],
                self.axes_manager._navigation_shape_in_array + (-1,),
            )
        return real_nav

    def cluster(self, method,
                axis=None,
                real_units=True,
                inplace=True,):
        """
        Clusters the vectors based on the clustering object given. The clustering
        object should have a 'fit' method which
        Parameters
        ----------
        method: func
            A function which returns labels for each of the
            vectors clustering. See sklearn.cluster
        navigation_to_vector
            Coverts each of the navigation axes to a vector
            to cluster in higher dimensions.
        **kwargs:
            Any additional keyword arguments for the method.
        Returns
        -------
        vector: Vector
            A collection of vectors clustered based on the function passed.
        """
        if inplace:
            vectors = self
        else:
            vectors = self.deepcopy()

        # This copies the vector array...
        all_vectors = vectors.get_real_vectors(axis=None,
                                               real_units=False,
                                               flatten=True)
        sub_vectors = vectors.get_real_vectors(axis=axis,
                                               real_units=real_units,
                                               flatten=True)

        cluster = method.fit(sub_vectors)
        labels = cluster.labels_
        groups = np.array(
            [all_vectors[labels == l] for l in range(max(labels))], dtype=object
        )
        vectors.data = groups
        vectors.learning_results = cluster
        return vectors

    def to_markers(self, x_axis=(-2,), y_axis=(-1,), **kwargs):
        """ Converts the vector to a set of markers given the axes.

        Parameters
        ----------
        x_axis: int
            The index for the x axis
        y_axis: int
            The index for the y axis
        """
        x_vectors = self.get_real_vectors(axis=x_axis).T
        y_vectors = self.get_real_vectors(axis=y_axis).T
        return Point(x_vectors, y_vectors, **kwargs)


class LazyBaseVectorSignal(LazySignal, BaseVectorSignal):
    _lazy = True
