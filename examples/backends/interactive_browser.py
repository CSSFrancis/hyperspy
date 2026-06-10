"""
====================================
Interactive HyperSpy in the Browser
====================================

Click the **⚡** badge on the figure below to launch this example live in
your browser via Pyodide — no installation required.

After a short bootstrap (~10 s the first time, cached on repeat visits),
the navigator on the left becomes fully interactive: click or drag to move
the crosshair and watch the spectrum on the right update in real time.

What this demonstrates
----------------------

* Building a synthetic EELS-style spectrum image entirely in NumPy
* Attaching a navigating :class:`~hyperspy.api.plot.markers.VerticalLines`
  marker whose position tracks the Gaussian peak centre across the map
* Calling :meth:`~.api.signals.BaseSignal.plot` to get a linked
  navigator + signal display via the ``anyplotlib`` backend
* How the ⚡ Pyodide bridge re-runs the full example in WASM and wires
  navigation events back to Python for live updates

"""

# Pyodide environment declarations
# ---------------------------------
# _PYODIDE_PACKAGES: pre-built Pyodide WASM wheels to load before execution.
# _PYODIDE_MOCK_PACKAGES: packages that aren't available in Pyodide but are
#   listed in HyperSpy's metadata — register as stubs so micropip skips them
#   during dependency resolution.
_PYODIDE_PACKAGES = ["scipy", "sympy", "matplotlib", "pyyaml"]
_PYODIDE_MOCK_PACKAGES = [
    "dask",
    "distributed",
    "rosettasciio",
    "cloudpickle",
    "pint",
    "natsort",
    "prettytable",
    "traits",
    "tqdm",
    "packaging",
    "importlib_metadata",
]

# %%
# Setup
# -----
# Import HyperSpy and switch to the anyplotlib backend.

import numpy as np
import hyperspy.api as hs

hs.preferences.Plot.backend = "anyplotlib"

# %%
# Synthetic spectrum image
# -------------------------
# 10 × 10 navigation map, 128-channel energy axis.  Each pixel contains a
# Gaussian whose centre drifts linearly with the x-position — the kind of
# peak shift you'd see in a real EELS dataset.

rng = np.random.default_rng(2024)

nav_y, nav_x, n_ch = 10, 10, 128
energy = np.linspace(0, 10, n_ch)
data = np.zeros((nav_y, nav_x, n_ch))

for iy in range(nav_y):
    for ix in range(nav_x):
        centre = 2.0 + ix * 0.5 + iy * 0.1
        width = 0.35 + rng.random() * 0.15
        amp = 0.8 + rng.random() * 0.4
        data[iy, ix] = amp * np.exp(-0.5 * ((energy - centre) / width) ** 2)

data += rng.random((nav_y, nav_x, n_ch)) * 0.02

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
s.metadata.General.title = "Synthetic EELS"

# %%
# Navigating marker
# ------------------
# A :class:`~hyperspy.api.plot.markers.VerticalLines` marker that shows the
# theoretical peak centre for each pixel.  Because the centre varies with x,
# the orange line shifts left-to-right as you navigate across the map.
#
# Navigating markers use a ``dtype=object`` array with the navigation shape,
# where each element is an array of positions for that navigation coordinate.

peak_centres = 2.0 + np.arange(nav_x) * 0.5  # centre varies with x

offsets_vlines = np.empty((nav_y, nav_x), dtype=object)
for iy in range(nav_y):
    for ix in range(nav_x):
        offsets_vlines[iy, ix] = np.array([peak_centres[ix]])

vlines = hs.plot.markers.VerticalLines(
    offsets=offsets_vlines,
    colors="#FF6400CC",
    linewidths=2.0,
)
s.add_marker(vlines, permanent=True)

# %%
# Plot — click ⚡ to make it live
# ---------------------------------
# The navigator (summed intensity map) is on the left; the spectrum at the
# current position is on the right.  After clicking ⚡, clicking or dragging
# on the navigator updates the spectrum in real time via Pyodide.

try:
    import anyplotlib  # noqa: F401
    hs.preferences.Plot.backend = "anyplotlib"
    s.plot()
    _apl_fig = s._plot.signal_plot.figure
    _apl_fig = getattr(_apl_fig, "_real_fig", _apl_fig)  # Interactive
    hs.preferences.Plot.backend = "matplotlib"
except ImportError:
    print("anyplotlib not installed — falling back to matplotlib.")
    s.plot()
