<!-- Parent: ../AGENTS.md -->
<!-- Updated: 2026-06-10 -->

# drawing

## Purpose
Backend-agnostic interactive plotting engine.  Handles the `signal.plot()` infrastructure, including multi-panel figures, navigator/signal separation, interactive widgets, and the markers system.  All rendering goes through a `PlottingBackend` protocol so that alternative backends (anyplotlib, pyqtgraph, fastplotlib, …) can replace matplotlib without touching the core drawing logic.

## Key Files

| File | Description |
|------|-------------|
| `signal.py` | Top-level signal plotting entry point |
| `signal1d.py` | 1D signal panel rendering (`Signal1DFigure`, `Signal1DLine`) |
| `image.py` | 2D image panel rendering (`ImagePlot`) |
| `figure.py` | `BlittedFigure` (blit animation base) + `AbstractSignal1DFigure` / `AbstractImageFigure` ABCs |
| `he.py` / `hse.py` / `hie.py` | `HyperExplorer` base + 1D / 2D generic explorer classes |
| `norm.py` | Backend-agnostic norm descriptors (`LinearNorm`, `LogNorm`, `PowerNorm`, `SymLogNorm`) |
| `marker_collection.py` | `HyperMarkerCollection` descriptor hierarchy (10 marker types) |
| `markers.py` | `Markers` — navigation-aware marker renderer |
| `widget.py` | `WidgetBase` — navigator widget base |
| `widgets.py` | Concrete interactive widgets (crosshair, range, etc.) |
| `tiles.py` | Tiled signal display |
| `utils.py` | Plot utility helpers |

## Backend System

### Key Files

| File | Description |
|------|-------------|
| `backends/_protocol.py` | `PlottingBackend` Protocol + `BackendCapabilityError` + `BlitMixin` + `PointerMixin` |
| `backends/_registry.py` | Entry-point discovery via `importlib.metadata` |
| `backends/__init__.py` | `get_backend()` / `register_backend()` |
| `backends/_stub.py` | Fully documented template for new backend authors |
| `backends/mpl/` | Matplotlib backend (default) |
| `backends/anyplotlib/` | anyplotlib backend |
| `backends/_magic.py` | `%anyplotlib` IPython magic |

### How the active backend is selected

1. `hyperspy/drawing/__init__.py` loads `"matplotlib"` at import time.
2. `preferences.Plot.backend` is an open `Str` trait; changing it fires `_on_backend_pref_change` which calls `load_backend(name)`.
3. `load_backend` looks up the `"hyperspy.backends"` entry-point group; external packages register there.

### Adding a new backend (summary)

1. Create a package with `pyproject.toml`:
   ```toml
   [project.entry-points."hyperspy.backends"]
   mybackend = "hyperspy_mybackend.backend:MyBackend"
   ```
2. Implement every method in `backends/_protocol.py`.  Copy `backends/_stub.py` as your starting point — all REQUIRED methods are clearly marked.
3. **Override `get_explorer`** — return the appropriate `HyperExplorer` subclass for each signal dimension.  The base class default is broken.  Subclass `HyperSignal1D_Explorer` + `HyperImage_Explorer` for 1D/2D; use `HyperExplorer` for 0D.  See `backends/anyplotlib/_explorers.py`.
4. **Handle `_on_figure_window_close`** — `BlittedFigure.create_figure` forwards this kwarg to `backend.create_figure`.  Store it on your figure object and call it from `close_figure` so the HyperExplorer's `close()` fires when the window is destroyed.  See `AnyplotlibBackend.create_figure` for the reference implementation.
5. Your axes objects must accept arbitrary attribute assignment (`ax.hspy_fig = ...`, `ax.figure = ...`); `signal1d.py` and `image.py` set these automatically.
6. Verify: `isinstance(MyBackend(), PlottingBackend)` and `pytest hyperspy/tests/drawing/`.

### Figure close callback threading

HyperSpy uses two close-callback mechanisms that backends must handle:

| Mechanism | How it works |
|-----------|-------------|
| `on_close` kwarg in `create_figure` | The `BlittedFigure`'s own close method; always set by `BlittedFigure.create_figure`. Call it from `close_figure` to tear down the figure object graph. |
| `_on_figure_window_close` kwarg in `create_figure` | The HyperExplorer's `close()` method; closes the entire explorer when the window is destroyed. For backends with native close events, use `connect_close_event` post-construction. For backends without (e.g. anyplotlib), store on the figure and call from `close_figure`. |

### anyplotlib backend — feature status

| Feature | Status |
|---------|--------|
| 1D / 2D plotting | ✅ Full |
| Image display (`imshow`, `pcolormesh`) | ✅ Full |
| Colorbar | ✅ Full |
| Native marker collections | ✅ Full (via `create_markers` + `_translate_marker_kwargs`) |
| Navigation line pointers (1D: vline) | ✅ Full |
| Navigation line pointers (2D: crosshair) | ⚠️ Requires `plot.add_widget` |
| Text annotations | ⚠️ Degrades to `_AplTextHandle` sentinel (canvas-invisible, state preserved) |
| Secondary y-axis (`twinx`) | ❌ `BackendCapabilityError` |
| Span / Polygon selectors | ❌ `BackendCapabilityError` |
| Coordinate transforms | ❌ `BackendCapabilityError` |
| Blit / animation | ❌ Not supported (anyplotlib repaints natively) |

### pyqtgraph / fastplotlib backends

External packages — not part of this repository.  Register under the `"hyperspy.backends"` entry point group.  Use `backends/_stub.py` as the implementation template.

### Known remaining MPL coupling in the generic layer

These items are intentional; a new backend whose axes accept arbitrary attribute assignment will not be broken by them:

- `_markers/` — `HyperMarkerCollection.mpl_collection()` returns a matplotlib `Collection` class; used only on the MPL fallback path.
- `_widgets/line2d.py`, `_widgets/circle.py` — `plt.Line2D` / `plt.Circle` construction is MPL-only; these widgets raise `BackendCapabilityError` on non-MPL backends.
- `signal1d.py:add_line` — color-cycle deduplication tries an exact string match first, then falls back to `matplotlib.colors.to_rgba` behind `try/except (ImportError, ValueError)` for canonical RGBA comparison.
- `signal1d.py:Signal1DLine.plot` — `matplotlib.colors.Normalize` type-check in the norm validation runs only when matplotlib is present; the guard enriches the error message for the common misuse case.
- `markers.py` — `_is_patch()` helper and legacy string-based collection fallback both guard with `try/except ImportError`.  These exist for backward compatibility with files saved before `HyperMarkerCollection`.
- `utils.py` — `create_figure`, `plot_RGB_map`, `plot_images`, `animate_legend` are intentionally MPL-specific public API helpers and not part of the generic rendering path.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `backends/` | Backend implementations and protocol |
| `_markers/` | Individual marker type implementations (Arrow, Circle, Line, etc.) |
| `_widgets/` | Low-level interactive widget implementations |

## For AI Agents

### Working In This Directory

- All rendering goes through the active backend: `get_backend().<method>()`.  **Never call `ax.transData`, `plt.gca()`, `ax.hspy_fig._background = None`, or similar MPL-specific calls from outside `backends/mpl/`.**
- Use `BackendCapabilityError` for features a backend does not yet support; callers in `markers.py` and `hse.py` degrade gracefully on this exception.
- Interactive updates rely on `events.py` — do not poll; connect/disconnect event handlers.
- Use `backend.render_figure_from_ax(ax)` for performance-critical repaints; use `backend.invalidate_blit_background(ax)` before `draw_patch()` when a patch is structurally removed/added.
- `BlittedFigure._on_close` is idempotent (guarded by `if self.figure is None: return`); it is safe to call it multiple times via re-entrant close paths.

### Testing Requirements

```bash
uv run pytest hyperspy/tests/drawing/
```

Use `matplotlib.use('Agg')` or the `mpl_cleanup` fixture for non-interactive test rendering.

## Dependencies

### Internal
- `hyperspy/events.py` — reactive event system
- `hyperspy/roi.py` — ROI widgets connect to drawing widgets

### External
- `matplotlib` — default rendering stack (via `backends/mpl/`)
- `anyplotlib` — optional alternative backend (via `backends/anyplotlib/`)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
