"""
=================================
Markers with the anyplotlib backend
=================================

Markers overlay annotations on HyperSpy signal plots.  This example shows
several marker types rendered via the ``anyplotlib`` backend, which maps
all marker geometry to native anyplotlib scatter/line collections rather
than matplotlib ``PathCollection`` objects.

The marker API is identical regardless of the active backend.

.. note::

   ``anyplotlib`` requires the ``anyplotlib`` package::

       pip install anyplotlib

"""

import numpy as np
import hyperspy.api as hs

# %%
# Create a simple spectrum image with two peaks that move across the map.

rng = np.random.default_rng(7)
nx, nch = 16, 256
x = np.linspace(0, 10, nch)
data = np.zeros((nx, nch))
centres = np.linspace(2, 8, nx)
for i, c in enumerate(centres):
    data[i] = np.exp(-0.5 * ((x - c) / 0.3) ** 2)
data += rng.random((nx, nch)) * 0.02

s = hs.signals.Signal1D(data)
s.axes_manager[0].name = "position"
s.axes_manager[0].scale = 1.0
s.axes_manager[1].name = "energy"
s.axes_manager[1].units = "eV"
s.axes_manager[1].scale = x[1] - x[0]
s.axes_manager[1].offset = x[0]

# %%
# Add a vertical-line marker that tracks the peak centre for each position.
# Navigating markers require a dtype=object array with navigation_shape,
# where each element is an array of positions for that nav coordinate.

offsets_vlines = np.empty(nx, dtype=object)
for i in range(nx):
    offsets_vlines[i] = np.array([centres[i]])

vlines = hs.plot.markers.VerticalLines(
    offsets=offsets_vlines,
    colors="crimson",
    linewidths=1.5,
)
s.add_marker(vlines, permanent=True)

# %%
# Add point markers at the peak maxima.

offsets_points = np.empty(nx, dtype=object)
for i in range(nx):
    offsets_points[i] = np.array([[centres[i], 1.0]])  # shape (1, 2): one point (x, y)

points = hs.plot.markers.Points(
    offsets=offsets_points,
    color="gold",
    sizes=30,
)
s.add_marker(points, permanent=True)

# %%
# Plot with anyplotlib.

try:
    import anyplotlib  # noqa: F401
    hs.preferences.Plot.backend = "anyplotlib"
    s.plot()
    # Expose figure for gallery scraper.
    _apl_fig = s._plot.signal_plot.figure
    _apl_fig = getattr(_apl_fig, "_real_fig", _apl_fig)
    hs.preferences.Plot.backend = "matplotlib"
except ImportError:
    print("anyplotlib not installed — skipping WebGL plot.  "
          "Install with:  pip install anyplotlib")
    s.plot()  # fall back to matplotlib so the gallery thumbnail is generated
