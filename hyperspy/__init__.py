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

import logging
import os
from hyperspy import docstrings

_logger = logging.getLogger(__name__)


try:
    from dask.widgets import TEMPLATE_PATHS
    templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "widgets/templates")
    TEMPLATE_PATHS.append(templates_path)
except ModuleNotFoundError:
    _logger.info("Dask widgets not loaded (dask >=2021.11.1 is required)")

__doc__ = """
HyperSpy: a multi-dimensional data analysis package for Python
==============================================================

Documentation is available in the docstrings and online at
http://hyperspy.org/hyperspy-doc/current/index.html.

All public packages, functions and classes are in :mod:`~hyperspy.api`. All
other packages and modules are for internal consumption and should not be
needed for data analysis.

%s

More details in the :mod:`~hyperspy.api` docstring.

""" % docstrings.START_HSPY


from . import Release

__all__ = ["api"]
__version__ = Release.version
