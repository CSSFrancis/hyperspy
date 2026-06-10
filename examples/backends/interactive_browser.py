"""
====================================
Interactive HyperSpy in the Browser
====================================

This example is designed to run **interactively in the browser** via
JupyterLite / Pyodide.  Click the *Run this example in the browser* button
at the top of the page to launch a fully-functional HyperSpy session without
installing anything locally.

What this demonstrates
----------------------

* Creating a synthetic hyperspectral dataset
* Switching to the ``anyplotlib`` backend
* Calling ``.plot()`` to get a linked navigator + signal display
* Moving the navigation pointer to update the signal in real time

The anyplotlib backend renders into a WebGL canvas via ``jupyter_rfb``, which
works in JupyterLite just like it does in a local Jupyter notebook.

.. note::

   This example requires ``anyplotlib`` and ``jupyterlite-sphinx``.  When
   viewed on the HyperSpy documentation website the *Run in browser* button
   is injected automatically by ``jupyterlite-sphinx``.

"""

# %%
# Setup
# -----
# Import HyperSpy and configure the anyplotlib backend.

import numpy as np
import hyperspy.api as hs

hs.preferences.Plot.backend = "anyplotlib"

# %%
# Create a synthetic spectrum image
# ----------------------------------
# 12×12 navigation, 256-channel spectrum.  Each pixel contains a Gaussian
# whose centre position encodes the spatial coordinate.

rng = np.random.default_rng(2024)

nav_y, nav_x, n_channels = 12, 12, 256
energy = np.linspace(0, 10, n_channels)
data = np.zeros((nav_y, nav_x, n_channels))

for iy in range(nav_y):
    for ix in range(nav_x):
        centre = 2.0 + ix * 0.4 + iy * 0.2
        width = 0.3 + rng.random() * 0.2
        amplitude = 0.8 + rng.random() * 0.4
        data[iy, ix] = amplitude * np.exp(-0.5 * ((energy - centre) / width) ** 2)

data += rng.random((nav_y, nav_x, n_channels)) * 0.03

s = hs.signals.Signal1D(data)
s.axes_manager[0].name = "y"
s.axes_manager[0].scale = 0.5
s.axes_manager[0].units = "nm"
s.axes_manager[1].name = "x"
s.axes_manager[1].scale = 0.5
s.axes_manager[1].units = "nm"
s.axes_manager[2].name = "energy"
s.axes_manager[2].scale = energy[1] - energy[0]
s.axes_manager[2].offset = energy[0]
s.axes_manager[2].units = "eV"
s.metadata.General.title = "Synthetic EELS spectrum image"

# %%
# Add a marker that tracks the peak centre for every pixel.
# The peak centre shifts with x — adding a VerticalLines marker makes this
# drift visible as you navigate the map.

peak_centres = 2.0 + np.arange(nav_x) * 0.4   # centre varies with x only

# Navigating markers need a dtype=object array with navigation_shape,
# where each element is an array of line positions for that nav coordinate.
offsets_vlines = np.empty((nav_y, nav_x), dtype=object)
for iy in range(nav_y):
    for ix in range(nav_x):
        offsets_vlines[iy, ix] = np.array([peak_centres[ix]])

vlines = hs.plot.markers.VerticalLines(
    offsets=offsets_vlines,
    colors="#FF6400B3",  # orange with ~70% opacity (works in both mpl and anyplotlib)
    linewidths=1.5,
)
s.add_marker(vlines, permanent=True)

# %%
# Plot
# ----
# A navigator (summed intensity map) appears on the left; the spectrum at
# the current navigation position appears on the right.  Click on the
# navigator to move the cursor and watch the spectrum update.

try:
    import anyplotlib  # noqa: F401
    hs.preferences.Plot.backend = "anyplotlib"
    s.plot()
    # Expose the combined figure widget for the gallery scraper.
    # The "# Interactive" tag tells AnywidgetScraper to embed the full Python
    # source so the Pyodide bridge can re-run the example live in the browser.
    _apl_fig = s._plot.signal_plot.figure
    _apl_fig = getattr(_apl_fig, "_real_fig", _apl_fig)  # Interactive
    hs.preferences.Plot.backend = "matplotlib"
except ImportError:
    print("anyplotlib not installed — skipping WebGL plot.  "
          "Install with:  pip install anyplotlib")
    s.plot()  # fall back to matplotlib so the gallery thumbnail is generated
