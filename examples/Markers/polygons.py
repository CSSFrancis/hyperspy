"""
Filled Polygon Markers
======================

This is good for identifying regions of interest on a plot. Usually used with something
like scipy.spaital.ConvexHull to describe the extent of some feature.

"""
import hyperspy.api as hs
import matplotlib as mpl
from scipy.spatial import ConvexHull
import numpy as np

# Create a Signal2D with 2 navigation dimensions
rng = np.random.default_rng()
s = hs.signals.Signal2D(rng.random((25, 25, 100, 100)))

# Define the position of the circles
offsets = rng.random((10, 2)) * 100


m = hs.plot.markers.Points(
    sizes=10,
    offsets=offsets,
    )

s.plot()
s.add_marker(m)

