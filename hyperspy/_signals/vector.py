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
from hyperspy.drawing._markers.point import Point


class Vector(BaseSignal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector = True

    def get_real_vectors(self):
        signal_axes = self.axes_manager.signal_axes
        scales = np.array([a.scale for a in signal_axes])
        offsets = np.array([a.offset for a in signal_axes])
        real_vector = np.empty(self.data.shape, dtype=object)
        for ind in np.ndindex(self.data.shape):
            vectors = self.data[ind]
            print(vectors)
            real = vectors*scales+offsets
            real_vector[ind] = real
        return real_vector

    def get_vector_axis(self, axis=0):
        signal_axes = self.axes_manager.signal_axes
        scales = signal_axes[axis].scale
        offsets = signal_axes[axis].offset
        real_vector = np.empty(self.data.shape, dtype=object)
        for ind in np.ndindex(self.data.shape):
            vectors = self.data[ind][:, axis]
            real = vectors*scales+offsets
            real_vector[ind] = real
        return real_vector

    def to_markers(self, xaxis=0, yaxis=1):
        x_vectors = self.get_vector_axis(axis=xaxis).T
        y_vectors = self.get_vector_axis(axis=yaxis).T
        return Point(x_vectors, y_vectors)

    def vector_to_navigation(self, axes=None):
        """Convert some of the navigation dimensions to a navigation dimension.

        For use when creating markers

        Returns
        -------
        vector_list: Vector
            A collection of all of vectors as pure vectors.  The navigation dimension is
            reduced to one dimension.
        """

        return

    def navigation_to_vector(self, inplace=False):
        """Convert the navigation dimension to a set of vectors.  Used when clustering
        based on more than just kx,ky.

        Returns
        -------
        vector_list: Vector
            A collection of all of vectors as pure vectors.  The navigation dimension is
            reduced to one dimension.
        """
        if inplace:
            vectors = self
        else:
            vectors = self.deepcopy()
        nav_indexes = np.ndindex(vectors.axes_manager.navigation_shape)
        scales = [a.scale for a in vectors.axes_manager.navigation_axes]
        offsets = [a.offset for a in vectors.axes_manager.navigation_axes]
        real_nav = [np.array(ind)*scales+offsets for ind in np.array(list(nav_indexes))]
        print(nav_indexes)
        new_v = [tuple(r) + tuple(vector) for i, r in zip(np.ndindex(vectors.axes_manager.navigation_shape),
                                                          real_nav) for vector in vectors.data[i]]
        return new_v

    def cluster(self,
                method,
                navigation_to_vector=False,
                real_units=True,
                inplace=True,
                **kwargs):
        """
        Clusters the vectors based on the method.

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
        if real_units:
            real_vectors = vectors.get_real_vectors()
        labels = method(real_vectors, **kwargs)
        groups = np.array([real_vectors[labels==l] for l in range(max(labels))],dtype=object)
        vectors.data = groups
        return vectors
