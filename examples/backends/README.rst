.. _backends-gallery:

Plotting Backends
=================

HyperSpy ships with a pluggable plotting backend system.  The default backend
is ``matplotlib``.  An ``anyplotlib`` backend is also included and demonstrates
how any rendering library can be connected to HyperSpy.

Switching backends::

    import hyperspy.api as hs
    hs.preferences.Plot.backend = "anyplotlib"

These examples show how the backend API is used and how to write your own.
