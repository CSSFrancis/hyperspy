"""
Colormaps
=========

:class:`~hyperspy.viewer.viewer2d.Viewer2D` supports every colormap
available in **matplotlib** and, when **colorcet** is installed, all
perceptually-uniform maps from that library as well.

The colormap affects both the image canvas **and** the colourbar in the
histogram panel, and it composites correctly with the intensity scale mode
(linear / log / symlog) and the display clim window.
"""

import numpy as np
from hyperspy.viewer.viewer2d import Viewer2D

rng = np.random.default_rng(0)

# %%
# Test image — a combination of smooth gradients and sharp features
# -----------------------------------------------------------------
# We build a 256×256 image with several distinct regions so that the
# colormap differences are clearly visible.

x = np.linspace(-3, 3, 256)
X, Y = np.meshgrid(x, x)
data = (
    np.exp(-(X**2 + Y**2) / 4)               # broad Gaussian blob
    + 0.4 * np.sin(3 * X) * np.cos(3 * Y)    # oscillating texture
    + 0.05 * rng.standard_normal((256, 256))  # noise
)
data = (data - data.min()) / (data.max() - data.min())  # → [0, 1]

# %%
# Default — ``'gray'``
# --------------------

v_gray = Viewer2D(data)

# %%
# Perceptually-uniform sequential — ``'viridis'``
# ------------------------------------------------
# ``viridis`` is the matplotlib default for new colormaps: perceptually
# uniform and readable by people with the most common forms of colour
# vision deficiency.

v_viridis = Viewer2D(data)
v_viridis.set_colormap('viridis')

# %%
# Other matplotlib sequential colormaps
# ---------------------------------------

v_plasma  = Viewer2D(data)
v_plasma.set_colormap('plasma')

v_inferno = Viewer2D(data)
v_inferno.set_colormap('inferno')

v_magma   = Viewer2D(data)
v_magma.set_colormap('magma')

v_hot     = Viewer2D(data)
v_hot.set_colormap('hot')

# %%
# Diverging colormap — ``'RdBu_r'``
# -----------------------------------
# Useful when the data has a natural midpoint (zero, mean, etc.).
# We centre the display window on 0.5 so both wings of the diverging
# map are used symmetrically.

v_div = Viewer2D(data)
v_div.set_colormap('RdBu_r')
v_div.set_clim(vmin=0.1, vmax=0.9)

# %%
# Cyclic colormap — ``'hsv'``
# ----------------------------
# Good for phase or orientation data where 0 and 1 represent the same value.

v_hsv = Viewer2D(data)
v_hsv.set_colormap('hsv')

# %%
# Colorcet perceptually-uniform maps (requires ``pip install colorcet``)
# -----------------------------------------------------------------------
# Colorcet provides a large family of maps that are linear in perceived
# brightness.  The short aliases are used below; full ``CET-*`` names also
# work.

try:
    import colorcet  # noqa: F401 — check availability

    v_fire = Viewer2D(data)
    v_fire.set_colormap('fire')        # CET-L3 — warm sequential

    v_bmy  = Viewer2D(data)
    v_bmy.set_colormap('bmy')          # CET-L8 — blue→magenta→yellow

    v_cet_d = Viewer2D(data)
    v_cet_d.set_colormap('CET-D1')     # diverging blue-red

    v_cet_r = Viewer2D(data)
    v_cet_r.set_colormap('CET-R1')     # rainbow perceptually uniform

    print("colorcet maps applied successfully.")
except ImportError:
    print("colorcet not installed — skipping colorcet examples.")
    print("Install with:  pip install colorcet")

# %%
# Listing available colormaps
# ----------------------------
# :meth:`~Viewer2D.list_colormaps` returns a sorted list of all names
# known to the installed libraries.  An optional substring filter narrows
# the search.

all_maps   = Viewer2D.list_colormaps()
seq_maps   = Viewer2D.list_colormaps('viridis')   # exact search
cet_maps   = Viewer2D.list_colormaps('cet')        # all colorcet maps

print(f"Total colormaps available: {len(all_maps)}")
print(f"Maps matching 'viridis':  {seq_maps}")
print(f"Maps matching 'cet' ({len(cet_maps)} total): {cet_maps[:8]} …")

# %%
# Combining colormap with log scale and clim
# -------------------------------------------
# The colormap, scale mode, and contrast window are fully independent and
# compose correctly.  Here ``'magma'`` is combined with log-intensity
# scaling to reveal detail in the faint regions of the image.

v_combo = Viewer2D(data)
v_combo.set_colormap('magma')
v_combo.set_scale_mode('log')
v_combo.set_clim(vmin=max(data.min() + 1e-3, 1e-3), vmax=data.max())

# %%
# Changing the colormap on an existing viewer
# --------------------------------------------
# Calling :meth:`~Viewer2D.set_colormap` on a live widget updates both the
# image and the colourbar immediately — no need to recreate the viewer.

v_live = Viewer2D(data)
v_live.set_colormap('plasma')   # first choice
# ... later, switch without losing zoom/pan/clim state:
v_live.set_colormap('viridis')

# %%
# sphinx_gallery_thumbnail_number = 2

