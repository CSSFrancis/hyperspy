"""
======================================
2-D Spectrum Image with anyplotlib
======================================

This example demonstrates navigating a 2-D spectrum image (4-D dataset)
using the ``anyplotlib`` backend.  Navigation and signal figures are
rendered as WebGL canvases, providing smooth panning and zooming without
requiring a GUI toolkit.

.. note::

   ``anyplotlib`` requires the ``anyplotlib`` package::

       pip install anyplotlib

"""

import numpy as np
import hyperspy.api as hs

# %%
# Build a synthetic spectrum image: 8×8 navigation, 128-point spectrum.
# Each pixel contains a Gaussian peak whose centre drifts across the map.

rng = np.random.default_rng(42)
nx, ny, nch = 8, 8, 128
x = np.linspace(0, 10, nch)
data = np.zeros((ny, nx, nch))
for iy in range(ny):
    for ix in range(nx):
        centre = 3 + (ix + iy) * 0.25
        amplitude = 1 + rng.random() * 0.5
        data[iy, ix] = amplitude * np.exp(-0.5 * ((x - centre) / 0.4) ** 2)
data += rng.random((ny, nx, nch)) * 0.05  # small noise floor

s = hs.signals.Signal1D(data)
s.axes_manager[0].name = "y"
s.axes_manager[1].name = "x"
s.axes_manager[2].name = "energy"
s.axes_manager[2].units = "eV"
s.axes_manager[2].scale = x[1] - x[0]
s.axes_manager[2].offset = x[0]

# %%
# Switch to the anyplotlib backend and plot.

try:
    import anyplotlib  # noqa: F401
    hs.preferences.Plot.backend = "anyplotlib"
    s.plot()
    hs.preferences.Plot.backend = "matplotlib"
except ImportError:
    print("anyplotlib not installed — skipping WebGL plot.  "
          "Install with:  pip install anyplotlib")
    s.plot()  # fall back to matplotlib so the gallery thumbnail is generated
