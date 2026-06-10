"""
====================
Switching Backends
====================

HyperSpy separates data exploration from rendering through a pluggable
backend system.  Two backends ship with HyperSpy:

* ``"matplotlib"`` — the default, backed by `matplotlib`
* ``"anyplotlib"`` — a WebGL backend built on `anyplotlib`

The active backend is selected once, before any plotting call.  All
subsequent :meth:`~.api.signals.BaseSignal.plot` calls use it
automatically.

.. note::

   ``anyplotlib`` requires the ``anyplotlib`` package::

       pip install anyplotlib

"""

import numpy as np
import hyperspy.api as hs

# %%
# Matplotlib backend (default)
# ----------------------------
# No special setup is required — matplotlib is used automatically.

rng = np.random.default_rng(0)
s = hs.signals.Signal1D(rng.random((4, 256)))
s.axes_manager[0].name = "position"
s.axes_manager[1].name = "energy"
s.axes_manager[1].units = "eV"

s.plot()

# %%
# Anyplotlib backend
# ------------------
# Select the backend before calling ``.plot()``.  The preference persists
# for the remainder of the session (or until changed again).

try:
    import anyplotlib  # noqa: F401
    _HAS_ANYPLOTLIB = True
except ImportError:
    _HAS_ANYPLOTLIB = False

if _HAS_ANYPLOTLIB:
    hs.preferences.Plot.backend = "anyplotlib"

    s2 = hs.signals.Signal1D(rng.random((4, 256)))
    s2.axes_manager[0].name = "position"
    s2.axes_manager[1].name = "energy"
    s2.axes_manager[1].units = "eV"

    s2.plot()
    # Expose the combined anyplotlib figure so the gallery scraper captures it.
    _apl_fig = s2._plot.signal_plot.figure
    _apl_fig = getattr(_apl_fig, "_real_fig", _apl_fig)  # unwrap proxy
else:
    print("anyplotlib not installed — skipping WebGL example.  "
          "Install with:  pip install anyplotlib")

# %%
# Restore matplotlib for other examples in this gallery
hs.preferences.Plot.backend = "matplotlib"
