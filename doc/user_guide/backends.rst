.. _plotting-backends:

Plotting Backends
*****************

HyperSpy separates data exploration from rendering through a pluggable backend
system.  Every call to :meth:`~.api.signals.BaseSignal.plot` goes through the
active backend rather than importing matplotlib directly, making it possible to
use any rendering library without modifying HyperSpy's core data logic.

Two backends ship with HyperSpy:

* **matplotlib** (default) — the familiar pyplot-based renderer used in the
  rest of the documentation.
* **anyplotlib** — a WebGL-accelerated backend built on `anyplotlib
  <https://github.com/fastplotlib/anyplotlib>`_.  It renders the same
  HyperSpy interface but draws with a GPU-friendly canvas, making it
  well-suited to Jupyter environments and large datasets.

Third-party packages can add additional backends via Python entry points (see
:ref:`custom-backends` below).

.. _selecting-backend:

Selecting a backend
===================

Set the backend preference before calling :meth:`~.api.signals.BaseSignal.plot`:

.. code-block:: python

    import hyperspy.api as hs
    hs.preferences.Plot.backend = "anyplotlib"

    s = hs.signals.Signal1D(...)
    s.plot()                           # rendered by anyplotlib

The preference is session-scoped.  To make it persistent across sessions, save
it to HyperSpy's configuration file:

.. code-block:: python

    hs.preferences.save()

To restore matplotlib:

.. code-block:: python

    hs.preferences.Plot.backend = "matplotlib"

.. _anyplotlib-backend:

The anyplotlib backend
======================

`anyplotlib <https://github.com/fastplotlib/anyplotlib>`_ is a high-level
plotting library that targets multiple rendering backends (currently fastplotlib
/ pygfx / WGPU).  Unlike matplotlib it renders into a ``jupyter_rfb``
canvas rather than a Qt/Tk/Agg window, which means:

* The plot lives inside a Jupyter notebook cell as a widget.
* Panning, zooming, and scrolling are handled by the GPU without recomputing
  the data.
* The navigator and signal figures can optionally share a single panel with
  a combined layout.

Installation::

    pip install anyplotlib

Feature comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 20

   * - Feature
     - matplotlib
     - anyplotlib
   * - 1-D signal plot
     - ✓
     - ✓
   * - 2-D image plot
     - ✓
     - ✓
   * - Markers (all types)
     - ✓
     - ✓
   * - Colorbar
     - ✓
     - ✓
   * - Blitting / animation
     - ✓
     - n/a (GPU repaint)
   * - Twin / right-axis
     - ✓
     - ✗ (raises ``BackendCapabilityError``)
   * - Span selector
     - ✓
     - ✗ (raises ``BackendCapabilityError``)
   * - Coordinate transforms
     - ✓
     - partial
   * - Jupyter Widget display
     - with ipympl
     - native


Interactive Examples
--------------------

The Sphinx Gallery examples in :ref:`backends-gallery` demonstrate the
anyplotlib backend.  Each example contains a *Run this example in the browser*
button (powered by JupyterLite / Pyodide) so you can try it without installing
anything locally.

To run the examples interactively in a local Jupyter session:

.. code-block:: python

    import hyperspy.api as hs
    hs.preferences.Plot.backend = "anyplotlib"

    import numpy as np
    s = hs.signals.Signal1D(np.random.default_rng(0).random((8, 256)))
    s.plot()

.. _custom-backends:

Writing a custom backend
========================

Any plotting library can be connected to HyperSpy by:

1. Implementing the :class:`~hyperspy.drawing.backends._protocol.PlottingBackend`
   protocol (see :mod:`hyperspy.drawing.backends._stub` for a copy-paste
   starting point).
2. Registering the class as a Python entry point under the
   ``"hyperspy.backends"`` group.

Minimal ``pyproject.toml`` entry::

    [project.entry-points."hyperspy.backends"]
    mybackend = "mypackage.plotting:MyBackend"

After installing your package, users can select your backend:

.. code-block:: python

    hs.preferences.Plot.backend = "mybackend"

For in-process registration (useful during development):

.. code-block:: python

    from hyperspy.drawing.backends import register_backend
    from mypackage.plotting import MyBackend
    register_backend("mybackend", MyBackend)

See :ref:`backends-gallery` for a worked example, and
``hyperspy/drawing/backends/_stub.py`` for the full method reference.

Protocol reference
------------------

.. autoclass:: hyperspy.drawing.backends._protocol.PlottingBackend
   :members:
   :undoc-members:

.. autoclass:: hyperspy.drawing.backends._protocol.BlitMixin
   :members:

.. autoclass:: hyperspy.drawing.backends._protocol.PointerMixin
   :members:

.. autoexception:: hyperspy.drawing.backends._protocol.BackendCapabilityError
