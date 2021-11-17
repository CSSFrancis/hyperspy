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
from hyperspy.misc.slicing import FancySlicing


class VectorComponent:

    def __init__(self,
                 vectors,
                 isnav=False,
                 axes=None):
        self.is_vector=True
        return

    def to_matrix(self, radius=1):
        """
        Parameters
        ----------
        radius: int
            The radius of the matrix to be used.

        Returns
        -------

        """


class Decomposition():
    """ The decomposition class is a simplified representation of some signal.

    This means that a signal can be created from calling the `_reconstruct`
    function.  For all intents the Decomposition acts like a signal and is
    computed when necessary.
    """
    def __init__(self,
                 navigation_components,
                 signal_components,
                 intensity=None,
                 metadata=None,
                 axes=None):
        """
        Parameters
        ---------
        navigation_components: list
            The list of navigation components.  If navigation components is
            a vector then each component is a vector of dimension equal
            to the navigation dimensions. Otherwise the navigation component
            spans the navigation dimension.

        signal_components:
            The list of signal components.  If signal components is
            a vector then each component is a vector of dimension equal
            to the signal dimensions. Otherwise the signal component
            spans the navigation dimension.
        """
        self.navigation_components = navigation_components
        self.is_nav_vector = len(np.shape(self.navigation_components)) > 1

        self.signal_components = signal_components
        self.is_sig_vector = len(np.shape(self.signal_components)) > 1
        self.intensity = intensity
        self.axes_manager = axes
        self.isig = SpecialSlicersDecomposition(self, False)
        self.inav = SpecialSlicersDecomposition(self, True)

    def plot(self):
        """Plots the paired navigation and signal components
        """
        return

    def _reconstruct(self, lazy=False):
        """Returns the signal created from the decomposition.  If lazy then the signal is
        only calculated when necessary.
        """
        if self.

        return

    def plot_signal(self, lazy=False, **kwargs):
        """Plots the reconstructed signal """
        self._reconstruct(lazy=lazy).plot(**kwargs)

    def sum(self, axis):
        """Sum of the data along some axis. If the data is vectorized then this gives the number of
        vectors for every pixel"""
        return

    def mean(self, axis):
        """mean of the data along some axis. If the data is vectorized then this gives the
        average number of vectors for every pixel"""
        return

    def save(self):
        """Saves the data using the .hspy format as a decomposition"""
        return