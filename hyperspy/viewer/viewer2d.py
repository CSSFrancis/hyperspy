"""
viewer2d.py
===========

Standalone 2-D image viewer widget backed by `anywidget` / JavaScript canvas.

Features
--------
* Optional physical ``x_axis`` / ``y_axis`` coordinate arrays.
* Scale bar (uniform axes) or X/Y rulers (non-uniform axes).
* Smooth zoom (mouse wheel) and pan (drag).
* Pixel-perfect rendering — no interpolation/antialiasing.
* Resizable canvas (drag the corner handle).
* Intensity histogram with optional log scale and color bar.
* Overlay *widgets* (circle, rectangle, annular, polygon, label) that are
  draggable and resizable directly on the canvas.
* Read-only *marker overlays* (circles, arrows, ellipses, lines, rectangles,
  squares, texts) with optional hover tooltips per-marker and/or per-collection.
* PyCharm / static-notebook fallback via ``image/png`` MIME bundle.
"""

from __future__ import annotations

import base64
import io
import json
import uuid as _uuid

import anywidget
import numpy as np
import traitlets

__all__ = ["Viewer2D"]


def _is_uniform(axis: np.ndarray, rtol: float = 1e-3) -> bool:
    """Return ``True`` if *axis* has a constant step size (uniform spacing)."""
    if len(axis) < 2:
        return True
    steps = np.diff(axis)
    return bool(np.allclose(steps, steps[0], rtol=rtol))


def _equal_scale(x_axis: np.ndarray, y_axis: np.ndarray, rtol: float = 1e-3) -> bool:
    """Return ``True`` when both axes are uniformly spaced with the same pixel size."""
    if not (_is_uniform(x_axis) and _is_uniform(y_axis)):
        return False
    dx = (x_axis[-1] - x_axis[0]) / (len(x_axis) - 1) if len(x_axis) > 1 else 1.0
    dy = (y_axis[-1] - y_axis[0]) / (len(y_axis) - 1) if len(y_axis) > 1 else 1.0
    return bool(np.isclose(abs(dx), abs(dy), rtol=rtol))


class Viewer2D(anywidget.AnyWidget):
    """Interactive 2-D image viewer that works directly with NumPy arrays.

    The widget renders in Jupyter / JupyterLab via ``anywidget``'s JavaScript
    runtime, and falls back to a static PNG thumbnail in environments that
    only support ``image/png`` (e.g. PyCharm's notebook preview).

    Parameters
    ----------
    data : np.ndarray
        2-D grayscale image of shape ``(H, W)``.  A 3-D array
        ``(H, W, C)`` is accepted; only the first channel is used.
    x_axis : array-like, optional
        Physical coordinates for each column (length ``W``).
        Defaults to ``np.arange(W)``.
    y_axis : array-like, optional
        Physical coordinates for each row (length ``H``).
        Defaults to ``np.arange(H)``.
    units : str, optional
        Physical unit label shown on the scale bar / axes (default ``'px'``).

    Examples
    --------
    >>> import numpy as np
    >>> from hyperspy.viewer.viewer2d import Viewer2D
    >>> v = Viewer2D(np.random.rand(256, 256))
    >>> v  # display in a Jupyter cell
    """

    # ------------------------------------------------------------------ traits
    # Image data — raw uint8 bytes (grayscale, row-major) synced to JS.
    image_bytes  = traitlets.Bytes(b"").tag(sync=True)
    image_width  = traitlets.Int(256).tag(sync=True)
    image_height = traitlets.Int(256).tag(sync=True)

    # Canvas display size in CSS pixels.
    viewer_width  = traitlets.Int(256).tag(sync=True)
    viewer_height = traitlets.Int(256).tag(sync=True)

    # Axis coordinate arrays (JSON-encoded float lists) and unit label.
    x_axis_json = traitlets.Unicode("[]").tag(sync=True)
    y_axis_json = traitlets.Unicode("[]").tag(sync=True)
    units       = traitlets.Unicode("px").tag(sync=True)

    # Scale bar vs. axis rulers.
    use_scalebar = traitlets.Bool(True).tag(sync=True)
    scale_x      = traitlets.Float(1.0).tag(sync=True)
    scale_y      = traitlets.Float(1.0).tag(sync=True)

    # Histogram panel.
    histogram_data    = traitlets.Unicode('{"bins":[],"counts":[]}').tag(sync=True)
    hist_min          = traitlets.Float(0.0).tag(sync=True)
    hist_max          = traitlets.Float(255.0).tag(sync=True)
    # Current display window — draggable in the histogram panel.
    display_min       = traitlets.Float(0.0).tag(sync=True)
    display_max       = traitlets.Float(255.0).tag(sync=True)
    # 'linear' | 'log' | 'symlog'
    scale_mode        = traitlets.Unicode("linear").tag(sync=True)
    histogram_visible = traitlets.Bool(True).tag(sync=True)
    log_scale         = traitlets.Bool(False).tag(sync=True)
    show_colorbar     = traitlets.Bool(True).tag(sync=True)
    colorbar_width    = traitlets.Int(20).tag(sync=True)
    histogram_width   = traitlets.Int(120).tag(sync=True)
    gap               = traitlets.Int(10).tag(sync=True)

    # Colormap — name for display, data as a JSON [[r,g,b],…] array of 256
    # uint8 triples resolved on the Python side from matplotlib / colorcet.
    colormap_name = traitlets.Unicode("gray").tag(sync=True)
    colormap_data = traitlets.Unicode("[]").tag(sync=True)

    # Zoom / pan state.
    zoom     = traitlets.Float(1.0).tag(sync=True)
    center_x = traitlets.Float(0.5).tag(sync=True)
    center_y = traitlets.Float(0.5).tag(sync=True)

    # Overlay widgets — JSON list of moveable/resizable shape dicts.
    # Shapes: circle, rectangle, annular, polygon, label.
    overlay_widgets = traitlets.Unicode("[]").tag(sync=True)

    # Marker overlays — JSON list of read-only marker-set dicts.
    # Each dict: { id, type, offsets/segments/…, color, linewidth,
    #              label?, labels? }
    # Types: circles, arrows, ellipses, lines, rectangles, squares, texts.
    markers_json = traitlets.Unicode("[]").tag(sync=True)

    # ------------------------------------------------------------------ JS
    _esm = r"""
    function render({ model, el }) {
      const dpr       = window.devicePixelRatio || 1;
      const AXIS_SIZE = 40;

      // ── theme detection ────────────────────────────────────────────────────
      // Reads the host environment's colour scheme by sampling the computed
      // background of the nearest scrollable/opaque ancestor.  Falls back to
      // prefers-color-scheme.  Returns a plain object with all UI colours.
      function _isDarkBg(el) {
        // Walk up the DOM to find a non-transparent background.
        let node = el.parentElement;
        while (node && node !== document.body) {
          const bg = window.getComputedStyle(node).backgroundColor;
          const m  = bg.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
          if (m) {
            const [r, g, b] = [+m[1], +m[2], +m[3]];
            // Skip fully-transparent
            if (!(r === 0 && g === 0 && b === 0 && bg.includes('0)'))) {
              // Perceived luminance (sRGB)
              return (0.299 * r + 0.587 * g + 0.114 * b) < 128;
            }
          }
          node = node.parentElement;
        }
        // Fallback: OS-level preference
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
      }

      function _makeTheme(dark) {
        return dark ? {
          bg:          '#1e1e2e',
          bgCanvas:    '#181825',
          bgHist:      '#181825',
          border:      '#44475a',
          axisBg:      '#1e1e2e',
          tickStroke:  '#6272a4',
          tickText:    '#cdd6f4',
          unitText:    '#6272a4',
          gridStroke:  'rgba(98,114,164,0.25)',
          dark:        true,
        } : {
          bg:          '#f0f0f0',
          bgCanvas:    '#ffffff',
          bgHist:      '#ffffff',
          border:      '#cccccc',
          axisBg:      '#f0f0f0',
          tickStroke:  '#666666',
          tickText:    '#333333',
          unitText:    '#888888',
          gridStroke:  'rgba(0,0,0,0.08)',
          dark:        false,
        };
      }

      let theme = _makeTheme(_isDarkBg(el));

      function _applyTheme() {
        theme = _makeTheme(_isDarkBg(el));
        container.style.background    = theme.bg;
        imageCanvas.style.borderColor = theme.border;
        xAxisCanvas.style.background  = theme.axisBg;
        yAxisCanvas.style.background  = theme.axisBg;
        histCanvas.style.borderColor  = theme.border;
        histCanvas.style.background   = theme.bgHist;
        drawAxes();
        drawHistogram();
      }

      // Re-theme when the host switches (e.g. JupyterLab toggle)
      const _mq = window.matchMedia('(prefers-color-scheme: dark)');
      _mq.addEventListener('change', _applyTheme);
      const _themeObserver = new MutationObserver(_applyTheme);
      _themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-jp-theme-name', 'data-vscode-theme-kind', 'class'] });

      // ── DOM ────────────────────────────────────────────────────────────────
      const outerContainer = document.createElement('div');
      outerContainer.style.cssText = 'position:relative;display:inline-block;';

      const container = document.createElement('div');
      container.style.cssText =
        `display:flex;flex-direction:row;gap:${model.get('gap')}px;` +
        `background:${theme.bg};padding:10px;border-radius:4px;position:relative;`;

      const histWidth = model.get('histogram_width');

      // Canvas wrapper
      const canvasWrapper = document.createElement('div');
      canvasWrapper.style.cssText = 'position:relative;display:inline-block;';

      // Image canvas
      const imageCanvas = document.createElement('canvas');
      imageCanvas.tabIndex = 1;
      imageCanvas.style.cssText = `outline:none;cursor:default;background:${theme.bgCanvas};border:1px solid ${theme.border};border-radius:2px;`;
      const imgCtx = imageCanvas.getContext('2d');
      imageCanvas.addEventListener('focus', () => { imageCanvas.style.boxShadow = '0 0 0 2px rgba(76,175,80,0.5)'; });
      imageCanvas.addEventListener('blur',  () => { imageCanvas.style.boxShadow = 'none'; });

      // Axis canvases
      const xAxisCanvas = document.createElement('canvas');
      xAxisCanvas.style.cssText = `display:block;background:${theme.axisBg};`;
      const xCtx = xAxisCanvas.getContext('2d');

      const yAxisCanvas = document.createElement('canvas');
      yAxisCanvas.style.cssText = `display:block;background:${theme.axisBg};`;
      const yCtx = yAxisCanvas.getContext('2d');

      // Layout: y-axis | (image / x-axis)
      const imageRow = document.createElement('div');
      imageRow.style.cssText = 'display:flex;flex-direction:row;align-items:flex-start;';
      const imageCol = document.createElement('div');
      imageCol.style.cssText = 'display:flex;flex-direction:column;align-items:flex-start;';
      imageRow.appendChild(yAxisCanvas);
      imageRow.appendChild(imageCol);
      imageCol.appendChild(imageCanvas);
      imageCol.appendChild(xAxisCanvas);
      canvasWrapper.appendChild(imageRow);

      // Overlay canvas – absolutely positioned on top of imageCanvas
      const overlayCanvas = document.createElement('canvas');
      overlayCanvas.style.cssText =
        'position:absolute;top:0;left:0;pointer-events:none;z-index:5;';
      imageCol.style.position = 'relative';
      imageCol.appendChild(overlayCanvas);
      const ovCtx = overlayCanvas.getContext('2d');

      // Markers canvas – sits above overlay canvas, also pointer-events:none
      const markersCanvas = document.createElement('canvas');
      markersCanvas.style.cssText =
        'position:absolute;top:0;left:0;pointer-events:none;z-index:6;';
      imageCol.appendChild(markersCanvas);
      const mkCtx = markersCanvas.getContext('2d');

      // Scale bar
      const scaleBar = document.createElement('div');
      scaleBar.style.cssText =
        'position:absolute;bottom:15px;right:15px;background:rgba(0,0,0,0.55);' +
        'padding:4px 8px;border-radius:4px;box-shadow:0 2px 4px rgba(0,0,0,0.4);pointer-events:none;';
      const scaleBarLine = document.createElement('div');
      scaleBarLine.style.cssText = 'background:white;height:4px;width:100px;margin-bottom:2px;';
      const scaleBarLabel = document.createElement('div');
      scaleBarLabel.style.cssText = 'color:white;font-size:11px;font-weight:bold;text-align:center;';
      scaleBarLabel.textContent = '100 px';
      scaleBar.appendChild(scaleBarLine);
      scaleBar.appendChild(scaleBarLabel);
      canvasWrapper.appendChild(scaleBar);

      // Status overlay — x / y / value readout, top-right corner while hovering.
      const statusBar = document.createElement('div');
      statusBar.style.cssText =
        'position:absolute;top:8px;right:8px;padding:2px 7px;' +
        'background:rgba(0,0,0,0.55);color:white;font-size:10px;font-family:monospace;' +
        'border-radius:4px;box-shadow:0 2px 4px rgba(0,0,0,0.4);pointer-events:none;' +
        'white-space:nowrap;display:none;z-index:8;';
      canvasWrapper.appendChild(statusBar);

      // Marker tooltip
      const markerTooltip = document.createElement('div');
      markerTooltip.style.cssText =
        'position:fixed;padding:5px 9px;font-size:12px;font-family:sans-serif;' +
        'background:rgba(30,30,30,0.92);color:#fff;border-radius:4px;' +
        'pointer-events:none;white-space:pre;display:none;z-index:9999;' +
        'box-shadow:0 2px 6px rgba(0,0,0,0.4);max-width:260px;';
      document.body.appendChild(markerTooltip);

      // Histogram canvas
      const histCanvas = document.createElement('canvas');
      histCanvas.style.cssText =
        `background:${theme.bgHist};border:1px solid ${theme.border};border-radius:2px;` +
        `display:${model.get('histogram_visible') ? 'block' : 'none'};`;
      const histCtx = histCanvas.getContext('2d');

      // Resize handle
      const resizeHandle = document.createElement('div');
      resizeHandle.style.cssText =
        'position:absolute;bottom:5px;right:5px;width:16px;height:16px;cursor:nwse-resize;' +
        'background:linear-gradient(135deg,transparent 50%,#888 50%);' +
        'border-radius:0 0 4px 0;z-index:10;';
      resizeHandle.title = 'Drag to resize';

      const sizeLabel = document.createElement('div');
      sizeLabel.style.cssText =
        'position:absolute;bottom:25px;right:25px;padding:4px 8px;background:rgba(0,0,0,0.7);' +
        'color:white;font-size:12px;border-radius:4px;display:none;pointer-events:none;z-index:10;';

      container.appendChild(canvasWrapper);
      container.appendChild(histCanvas);
      outerContainer.appendChild(container);
      outerContainer.appendChild(resizeHandle);
      outerContainer.appendChild(sizeLabel);

      el.appendChild(outerContainer);

      // ── state ──────────────────────────────────────────────────────────────
      let isResizing = false, isPanning = false;
      let startX, startY, startWidth, startHeight;
      let panStartX, panStartY, panStartCenterX, panStartCenterY;
      let _suppressSyncResize = false;

      // ── syncCanvasSizes ────────────────────────────────────────────────────
      function syncCanvasSizes() {
        // Suppressed while a resize drag is committing both traits at once.
        if (_suppressSyncResize) return;

        const useScalebar = model.get('use_scalebar');
        const w = model.get('viewer_width');
        const h = model.get('viewer_height');

        // Only resize if both traits have actually settled to the same values
        // that are already on the canvas — guards against the partial-update
        // race where change:viewer_width fires before change:viewer_height.
        const curW = parseInt(imageCanvas.style.width)  || 0;
        const curH = parseInt(imageCanvas.style.height) || 0;
        if (curW === w && curH === h) return;   // already correct, nothing to do

        _applyCanvasSize(w, h);

        if (useScalebar) {
          xAxisCanvas.style.display = 'none';
          yAxisCanvas.style.display = 'none';
          scaleBar.style.display    = 'block';
        } else {
          xAxisCanvas.width  = w * dpr;  xAxisCanvas.height = AXIS_SIZE * dpr;
          xAxisCanvas.style.width = w + 'px';  xAxisCanvas.style.height = AXIS_SIZE + 'px';
          xAxisCanvas.style.display = 'block';
          yAxisCanvas.width  = AXIS_SIZE * dpr;  yAxisCanvas.height = h * dpr;
          yAxisCanvas.style.width = AXIS_SIZE + 'px';  yAxisCanvas.style.height = h + 'px';
          yAxisCanvas.style.display = 'block';
          xCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
          yCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
          scaleBar.style.display = 'none';
        }

        histCanvas.width  = histWidth * dpr;  histCanvas.height = h * dpr;
        histCanvas.style.width = histWidth + 'px';  histCanvas.style.height = h + 'px';
        histCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
      syncCanvasSizes();

      // ── helpers ────────────────────────────────────────────────────────────
      function findNice(target) {
        if (target <= 0) return 1;
        const mag = Math.pow(10, Math.floor(Math.log10(target)));
        let best = mag, bestDiff = Math.abs(target - mag);
        for (const n of [1, 2, 3, 5, 7, 10]) {
          for (const m of [1, 10]) {
            const v = n * mag * m, d = Math.abs(target - v);
            if (d < bestDiff) { best = v; bestDiff = d; }
          }
        }
        return best;
      }

      function fmtVal(v) {
        if (Math.abs(v) >= 1000) return (v / 1000).toFixed(1) + 'k';
        if (Math.abs(v) >= 10)   return v.toFixed(0);
        if (Math.abs(v) >= 1)    return v.toFixed(1);
        return v.toFixed(2);
      }

      // ── axis coordinate helpers (uniform & non-uniform) ───────────────────
      /** True when all steps are within 0.1 % of the first step. */
      function _axisIsUniform(arr) {
        if (arr.length < 2) return true;
        const step0 = arr[1] - arr[0];
        if (step0 === 0) return false;
        for (let i = 2; i < arr.length; i++) {
          if (Math.abs((arr[i] - arr[i-1]) / step0 - 1) > 1e-3) return false;
        }
        return true;
      }

      /**
       * Physical axis value → fractional image position [0, 1].
       * Binary-search + linear interpolation — works for any spacing.
       */
      function _axisValToFrac(arr, val) {
        if (arr.length < 2) return 0;
        const n = arr.length;
        const asc = arr[n-1] >= arr[0];
        if (asc ? val <= arr[0]   : val >= arr[0])   return 0;
        if (asc ? val >= arr[n-1] : val <= arr[n-1]) return 1;
        let lo = 0, hi = n - 2;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          const inSeg = asc ? (arr[mid] <= val && val < arr[mid+1])
                            : (arr[mid] >= val && val > arr[mid+1]);
          if (inSeg) { lo = mid; break; }
          if (asc ? arr[mid+1] <= val : arr[mid+1] >= val) lo = mid + 1;
          else hi = mid;
        }
        const t = (val - arr[lo]) / (arr[lo+1] - arr[lo]);
        return (lo + t) / (n - 1);
      }

      /**
       * Fractional image position [0, 1] → physical axis value.
       * Exact inverse of _axisValToFrac.
       */
      function _axisFracToVal(arr, frac) {
        if (arr.length < 2) return arr.length ? arr[0] : 0;
        const n   = arr.length;
        const pos = Math.max(0, Math.min(1, frac)) * (n - 1);
        const lo  = Math.min(Math.floor(pos), n - 2);
        const t   = pos - lo;
        return arr[lo] + t * (arr[lo+1] - arr[lo]);
      }

      // ── scale bar ──────────────────────────────────────────────────────────
      function drawScaleBar() {
        const cw    = parseInt(imageCanvas.style.width)  || model.get('viewer_width');
        const imgW  = model.get('image_width');
        const units = model.get('units');
        const zoom  = model.get('zoom');
        const cx    = model.get('center_x');
        const targetPx = cw / 4;

        if (units === 'px') {
          const ppsp = imgW / cw;
          const nice = findNice(targetPx * ppsp / zoom);
          scaleBarLine.style.width  = (nice / ppsp * zoom) + 'px';
          scaleBarLabel.textContent = fmtVal(nice) + ' px';
        } else {
          const xArr = JSON.parse(model.get('x_axis_json'));
          if (xArr.length < 2) return;
          const fracPerPx = zoom >= 1.0 ? (1 / zoom) / cw : 1 / (cw * zoom);
          const cx0 = zoom >= 1.0 ? Math.max(0.5/zoom, Math.min(1-0.5/zoom, cx)) : 0.5;
          const physPerPx = Math.abs(
            _axisFracToVal(xArr, Math.min(1, cx0 + fracPerPx)) - _axisFracToVal(xArr, cx0)
          );
          if (physPerPx <= 0) return;
          const nice = findNice(targetPx * physPerPx);
          scaleBarLine.style.width  = (nice / physPerPx) + 'px';
          scaleBarLabel.textContent = fmtVal(nice) + ' ' + units;
        }
      }

      // ── axis rulers (uniform & non-uniform) ───────────────────────────────
      function drawAxes() {
        const xArr  = JSON.parse(model.get('x_axis_json'));
        const yArr  = JSON.parse(model.get('y_axis_json'));
        const units = model.get('units');
        const cw    = parseInt(imageCanvas.style.width)  || model.get('viewer_width');
        const ch    = parseInt(imageCanvas.style.height) || model.get('viewer_height');
        const zoom  = model.get('zoom');
        const cx    = model.get('center_x'), cy = model.get('center_y');

        // Visible fractional range given current zoom/pan.
        function _visFrac(zoom, centre) {
          if (zoom >= 1.0) {
            const half = 0.5 / zoom;
            const c = Math.max(half, Math.min(1 - half, centre));
            return [c - half, c + half];
          }
          return [0, 1];
        }

        // ── X axis ──────────────────────────────────────────────────────────
        xCtx.clearRect(0, 0, cw, AXIS_SIZE);
        xCtx.fillStyle = theme.axisBg;
        xCtx.fillRect(0, 0, cw, AXIS_SIZE);
        if (xArr.length >= 2) {
          const [xF0, xF1] = _visFrac(zoom, cx);
          const xVMin = _axisFracToVal(xArr, xF0);
          const xVMax = _axisFracToVal(xArr, xF1);
          const xRange = xVMax - xVMin || 1;
          const step = findNice(xRange / Math.max(3, Math.floor(cw / 60)));
          xCtx.strokeStyle = theme.tickStroke; xCtx.lineWidth = 1;
          xCtx.fillStyle = theme.tickText; xCtx.font = '10px sans-serif'; xCtx.textAlign = 'center';
          xCtx.beginPath(); xCtx.moveTo(0, 0); xCtx.lineTo(cw, 0); xCtx.stroke();
          for (let v = Math.ceil(xVMin / step) * step; v <= xVMax + step * 0.01; v += step) {
            const frac = _axisValToFrac(xArr, v);
            let px;
            if (zoom >= 1.0) {
              const half = 0.5 / zoom;
              const c = Math.max(half, Math.min(1 - half, cx));
              px = (frac - (c - half)) / (2 * half) * cw;
            } else {
              px = (cw - cw * zoom) / 2 + frac * cw * zoom;
            }
            if (px < 0 || px > cw) continue;
            xCtx.beginPath(); xCtx.moveTo(px, 0); xCtx.lineTo(px, 6); xCtx.stroke();
            xCtx.fillText(fmtVal(v), px, 18);
          }
          xCtx.textAlign = 'right'; xCtx.fillStyle = theme.unitText;
          xCtx.fillText(units, cw - 2, AXIS_SIZE - 4);
          if (!_axisIsUniform(xArr)) {
            xCtx.textAlign = 'left'; xCtx.fillStyle = 'rgba(255,140,0,0.85)';
            xCtx.font = '8px sans-serif';
            xCtx.fillText('non-uniform', 2, AXIS_SIZE - 4);
          }
        }

        // ── Y axis ──────────────────────────────────────────────────────────
        yCtx.clearRect(0, 0, AXIS_SIZE, ch);
        yCtx.fillStyle = theme.axisBg;
        yCtx.fillRect(0, 0, AXIS_SIZE, ch);
        if (yArr.length >= 2) {
          const [yF0, yF1] = _visFrac(zoom, cy);
          const yVMin = _axisFracToVal(yArr, yF0);
          const yVMax = _axisFracToVal(yArr, yF1);
          const yRange = yVMax - yVMin || 1;
          const step = findNice(yRange / Math.max(3, Math.floor(ch / 60)));
          yCtx.strokeStyle = theme.tickStroke; yCtx.lineWidth = 1;
          yCtx.fillStyle = theme.tickText; yCtx.font = '10px sans-serif'; yCtx.textAlign = 'right';
          yCtx.beginPath(); yCtx.moveTo(AXIS_SIZE, 0); yCtx.lineTo(AXIS_SIZE, ch); yCtx.stroke();
          for (let v = Math.ceil(yVMin / step) * step; v <= yVMax + step * 0.01; v += step) {
            const frac = _axisValToFrac(yArr, v);
            let py;
            if (zoom >= 1.0) {
              const half = 0.5 / zoom;
              const c = Math.max(half, Math.min(1 - half, cy));
              py = (frac - (c - half)) / (2 * half) * ch;
            } else {
              py = (ch - ch * zoom) / 2 + frac * ch * zoom;
            }
            if (py < 0 || py > ch) continue;
            yCtx.beginPath(); yCtx.moveTo(AXIS_SIZE, py); yCtx.lineTo(AXIS_SIZE - 6, py); yCtx.stroke();
            yCtx.fillText(fmtVal(v), AXIS_SIZE - 8, py + 4);
          }
          if (!_axisIsUniform(yArr)) {
            yCtx.save();
            yCtx.translate(2, ch - 2);
            yCtx.rotate(-Math.PI / 2);
            yCtx.textAlign = 'left'; yCtx.fillStyle = 'rgba(255,140,0,0.85)';
            yCtx.font = '8px sans-serif';
            yCtx.fillText('non-uniform', 0, 0);
            yCtx.restore();
          }
        }
      }

      // ── draw image ─────────────────────────────────────────────────────────
      function drawImage() {
        const raw = model.get('image_bytes');
        let bytes;
        if (raw instanceof Uint8Array)                     bytes = raw;
        else if (raw instanceof ArrayBuffer)               bytes = new Uint8Array(raw);
        else if (raw && raw.buffer instanceof ArrayBuffer) bytes = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
        else                                               bytes = new Uint8Array(0);

        const iw = model.get('image_width'),  ih = model.get('image_height');
        const cw = parseInt(imageCanvas.style.width)  || model.get('viewer_width');
        const ch = parseInt(imageCanvas.style.height) || model.get('viewer_height');

        imgCtx.clearRect(0, 0, cw, ch);
        if (bytes.length === 0) return;

        imgCtx.imageSmoothingEnabled = false;

        // ── intensity LUT: raw uint8 → remapped uint8 [0,255] ───────────────
        const dMin  = model.get('display_min');
        const dMax  = model.get('display_max');
        const hMin  = model.get('hist_min');
        const hMax  = model.get('hist_max');
        const mode  = model.get('scale_mode');
        const range = hMax - hMin || 1;

        const intensityLut = new Uint8Array(256);
        for (let raw8 = 0; raw8 < 256; raw8++) {
          const val = hMin + (raw8 / 255) * range;
          let t;
          if (mode === 'log') {
            const dMinC = Math.max(dMin, 1e-10), dMaxC = Math.max(dMax, dMinC + 1e-10);
            t = (Math.log10(Math.max(val, 1e-10)) - Math.log10(dMinC)) /
                (Math.log10(dMaxC) - Math.log10(dMinC));
          } else if (mode === 'symlog') {
            const linThresh = Math.max((dMax - dMin) * 0.01, 1e-10);
            const sl = v => v >= 0
              ? (v <= linThresh ? v / linThresh : 1 + Math.log10(v / linThresh))
              : -(Math.abs(v) <= linThresh ? Math.abs(v) / linThresh : 1 + Math.log10(Math.abs(v) / linThresh));
            t = (sl(val) - sl(dMin)) / ((sl(dMax) - sl(dMin)) || 1);
          } else {
            t = (val - dMin) / ((dMax - dMin) || 1);
          }
          intensityLut[raw8] = Math.max(0, Math.min(255, Math.round(t * 255)));
        }

        // ── colormap LUT: uint8 [0,255] → [r, g, b] ─────────────────────────
        let cmapData;
        try { cmapData = JSON.parse(model.get('colormap_data')); } catch (_) { cmapData = []; }
        const hasCmap = Array.isArray(cmapData) && cmapData.length === 256;

        // ── build RGBA image ─────────────────────────────────────────────────
        const imageData = imgCtx.createImageData(iw, ih);
        for (let i = 0; i < bytes.length; i++) {
          const mapped = intensityLut[bytes[i]];
          const base   = i * 4;
          if (hasCmap) {
            imageData.data[base]     = cmapData[mapped][0];
            imageData.data[base + 1] = cmapData[mapped][1];
            imageData.data[base + 2] = cmapData[mapped][2];
          } else {
            imageData.data[base]     = mapped;
            imageData.data[base + 1] = mapped;
            imageData.data[base + 2] = mapped;
          }
          imageData.data[base + 3] = 255;
        }

        const tmp = document.createElement('canvas');
        tmp.width = iw; tmp.height = ih;
        tmp.getContext('2d').putImageData(imageData, 0, 0);

        const zoom = model.get('zoom');
        const cx   = model.get('center_x'), cy = model.get('center_y');
        if (zoom >= 1.0) {
          const visW = iw / zoom, visH = ih / zoom;
          const srcX = Math.max(0, Math.min(iw - visW, cx * iw - visW / 2));
          const srcY = Math.max(0, Math.min(ih - visH, cy * ih - visH / 2));
          imgCtx.drawImage(tmp, srcX, srcY, visW, visH, 0, 0, cw, ch);
        } else {
          const dstW = cw * zoom, dstH = ch * zoom;
          imgCtx.fillStyle = theme.bgCanvas;
          imgCtx.fillRect(0, 0, cw, ch);
          imgCtx.drawImage(tmp, 0, 0, iw, ih, (cw - dstW) / 2, (ch - dstH) / 2, dstW, dstH);
        }

        if (model.get('use_scalebar')) drawScaleBar(); else drawAxes();
        drawOverlay();
        drawMarkers();
      }

      // ── histogram ──────────────────────────────────────────────────────────
      // Histogram drag state
      let histDrag = null;   // null | 'min' | 'max'

      // Convert a data value to a y-pixel position within the histogram chart area.
      function _dataToHistY(val, h) {
        const hMin = model.get('hist_min'), hMax = model.get('hist_max');
        return h - (val - hMin) / ((hMax - hMin) || 1) * h;
      }

      // Convert a y-pixel within the histogram chart area to a data value.
      function _histYToData(py, h) {
        const hMin = model.get('hist_min'), hMax = model.get('hist_max');
        return hMin + (1 - py / h) * ((hMax - hMin) || 1);
      }

      function drawHistogram() {
        if (!model.get('histogram_visible')) return;
        const histData  = JSON.parse(model.get('histogram_data'));
        const counts    = histData.counts;
        const w         = parseInt(histCanvas.style.width)  || histWidth;
        const h         = parseInt(histCanvas.style.height) || model.get('viewer_height');
        const showCB    = model.get('show_colorbar');
        const cbW       = model.get('colorbar_width');
        const mode      = model.get('scale_mode');
        const dMin      = model.get('display_min');
        const dMax      = model.get('display_max');
        const hMin      = model.get('hist_min');
        const hMax      = model.get('hist_max');

        histCtx.clearRect(0, 0, w, h);
        if (!counts || counts.length === 0) return;

        // ── bars ────────────────────────────────────────────────────────────
        // The bars always represent pixel-count frequency; log_scale (the 'h'
        // toggle) compresses tall spikes in the *count* axis.  scale_mode
        // (l / s keys) only affects how image intensities are mapped and is
        // intentionally NOT applied to the bars themselves.
        const useLogCounts = model.get('log_scale');
        let proc = counts.slice();
        if (useLogCounts) proc = counts.map(c => c > 0 ? Math.log10(c + 1) : 0);
        const maxC   = Math.max(...proc, 1);
        const chartX = showCB ? cbW + 5 : 0;
        const chartW = w - chartX;
        const barH   = h / counts.length;

        histCtx.fillStyle = '#4CAF50';
        histCtx.strokeStyle = '#2E7D32';
        histCtx.lineWidth = 0.5;
        for (let i = 0; i < counts.length; i++) {
          const idx = counts.length - 1 - i;
          const bw  = (proc[idx] / maxC) * chartW;
          const y   = i * barH;
          if (bw > 0) {
            histCtx.fillRect(chartX, y, bw, barH);
            if (barH > 2) histCtx.strokeRect(chartX, y, bw, barH);
          }
        }

        // ── colorbar (reflects current scale mode + colormap) ───────────────
        if (showCB) {
          let cmapData;
          try { cmapData = JSON.parse(model.get('colormap_data')); } catch (_) { cmapData = []; }
          const hasCmap = Array.isArray(cmapData) && cmapData.length === 256;

          const g = histCtx.createLinearGradient(0, 0, 0, h);
          const steps = 64;
          for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            // position in data space (top = hMax, bottom = hMin)
            const val = hMin + (1 - t) * (hMax - hMin);
            // normalise val through display window + scale mode → [0,1]
            let mapped;
            if (mode === 'log') {
              const dMinC = Math.max(dMin, 1e-10), dMaxC = Math.max(dMax, dMinC + 1e-10);
              const valC  = Math.max(val,  1e-10);
              mapped = (Math.log10(valC) - Math.log10(dMinC)) / (Math.log10(dMaxC) - Math.log10(dMinC));
            } else if (mode === 'symlog') {
              const linThresh = Math.max((dMax - dMin) * 0.01, 1e-10);
              const sl = v => v >= 0
                ? (v <= linThresh ? v / linThresh : 1 + Math.log10(v / linThresh))
                : -(Math.abs(v) <= linThresh ? Math.abs(v) / linThresh : 1 + Math.log10(Math.abs(v) / linThresh));
              const slRange = (sl(dMax) - sl(dMin)) || 1;
              mapped = (sl(val) - sl(dMin)) / slRange;
            } else {
              mapped = (val - dMin) / ((dMax - dMin) || 1);
            }
            const idx = Math.max(0, Math.min(255, Math.round(mapped * 255)));
            let colStr;
            if (hasCmap) {
              const [r, gv, b] = cmapData[idx];
              colStr = `rgb(${r},${gv},${b})`;
            } else {
              const grey = idx;
              colStr = `rgb(${grey},${grey},${grey})`;
            }
            g.addColorStop(t, colStr);
          }
          histCtx.fillStyle = g;
          histCtx.fillRect(0, 0, cbW, h);
          histCtx.strokeStyle = theme.tickStroke;
          histCtx.lineWidth = 1;
          histCtx.strokeRect(0, 0, cbW, h);

          // Colormap name badge along the left edge of the colorbar.
          const cmapName = model.get('colormap_name');
          if (cmapName && cmapName !== 'gray') {
            histCtx.save();
            histCtx.translate(cbW / 2, h - 4);
            histCtx.rotate(-Math.PI / 2);
            histCtx.font = 'bold 8px sans-serif';
            histCtx.textAlign = 'left';
            histCtx.fillStyle = 'rgba(255,255,255,0.85)';
            histCtx.fillText(cmapName, 0, 0);
            histCtx.restore();
          }
        }

        // ── min/max labels ──────────────────────────────────────────────────
        histCtx.fillStyle = theme.tickText;
        histCtx.font = '10px monospace';
        histCtx.textAlign = 'left';
        histCtx.fillText(hMax.toFixed(2), chartX + 2, 12);
        histCtx.fillText(hMin.toFixed(2), chartX + 2, h - 3);

        // ── draggable display_min / display_max lines ───────────────────────
        const yMax = _dataToHistY(dMax, h);
        const yMin = _dataToHistY(dMin, h);

        // Clamp display lines to visible area
        const yMaxCl = Math.max(2,  Math.min(h - 2, yMax));
        const yMinCl = Math.max(2,  Math.min(h - 2, yMin));

        // Max line (bright white + label)
        histCtx.save();
        histCtx.strokeStyle = '#ffffff';
        histCtx.lineWidth = 2;
        histCtx.setLineDash([4, 3]);
        histCtx.beginPath();
        histCtx.moveTo(chartX, yMaxCl);
        histCtx.lineTo(w, yMaxCl);
        histCtx.stroke();
        histCtx.setLineDash([]);
        histCtx.fillStyle = 'rgba(0,0,0,0.6)';
        const maxLabel = dMax.toFixed(2);
        const maxLW = histCtx.measureText(maxLabel).width + 6;
        histCtx.fillRect(w - maxLW - 2, yMaxCl - 13, maxLW + 2, 13);
        histCtx.fillStyle = '#fff';
        histCtx.font = '9px monospace';
        histCtx.textAlign = 'right';
        histCtx.fillText(maxLabel, w - 3, yMaxCl - 2);
        histCtx.restore();

        // Min line
        histCtx.save();
        histCtx.strokeStyle = '#aaaaff';
        histCtx.lineWidth = 2;
        histCtx.setLineDash([4, 3]);
        histCtx.beginPath();
        histCtx.moveTo(chartX, yMinCl);
        histCtx.lineTo(w, yMinCl);
        histCtx.stroke();
        histCtx.setLineDash([]);
        histCtx.fillStyle = 'rgba(0,0,0,0.6)';
        const minLabel = dMin.toFixed(2);
        const minLW = histCtx.measureText(minLabel).width + 6;
        histCtx.fillRect(w - minLW - 2, yMinCl, minLW + 2, 13);
        histCtx.fillStyle = '#aaaaff';
        histCtx.font = '9px monospace';
        histCtx.textAlign = 'right';
        histCtx.fillText(minLabel, w - 3, yMinCl + 11);
        histCtx.restore();

        // ── scale mode badge ────────────────────────────────────────────────
        if (mode !== 'linear') {
          const badge = mode === 'log' ? 'LOG' : 'SYMLOG';
          histCtx.save();
          histCtx.font = 'bold 9px monospace';
          histCtx.fillStyle = 'rgba(255,180,0,0.9)';
          histCtx.textAlign = 'right';
          histCtx.fillText(badge, w - 3, h - 3);
          histCtx.restore();
        }
      }

      // ── histogram drag (min/max lines) ─────────────────────────────────────
      const HIST_DRAG_TOL = 6;   // px tolerance for grabbing a line

      histCanvas.style.cursor = 'default';

      histCanvas.addEventListener('mousedown', (e) => {
        const rect  = histCanvas.getBoundingClientRect();
        const my    = e.clientY - rect.top;
        const h     = rect.height;
        const yMax  = _dataToHistY(model.get('display_max'), h);
        const yMin  = _dataToHistY(model.get('display_min'), h);
        if (Math.abs(my - yMax) <= HIST_DRAG_TOL) {
          histDrag = 'max'; e.preventDefault();
        } else if (Math.abs(my - yMin) <= HIST_DRAG_TOL) {
          histDrag = 'min'; e.preventDefault();
        }
      });

      document.addEventListener('mousemove', (e) => {
        if (!histDrag) return;
        const rect = histCanvas.getBoundingClientRect();
        const my   = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
        let val    = _histYToData(my, rect.height);
        const hMin = model.get('hist_min'), hMax = model.get('hist_max');
        val = Math.max(hMin, Math.min(hMax, val));

        if (histDrag === 'max') {
          const dMin = model.get('display_min');
          if (val > dMin) {
            model.set('display_max', val);
            model.save_changes();
          }
        } else {
          const dMax = model.get('display_max');
          if (val < dMax) {
            model.set('display_min', val);
            model.save_changes();
          }
        }
        e.preventDefault();
      });

      document.addEventListener('mouseup', () => {
        if (histDrag) { histDrag = null; }
      });

      histCanvas.addEventListener('mousemove', (e) => {
        if (histDrag) return;
        const rect = histCanvas.getBoundingClientRect();
        const my   = e.clientY - rect.top;
        const h    = rect.height;
        const yMax = _dataToHistY(model.get('display_max'), h);
        const yMin = _dataToHistY(model.get('display_min'), h);
        if (Math.abs(my - yMax) <= HIST_DRAG_TOL || Math.abs(my - yMin) <= HIST_DRAG_TOL) {
          histCanvas.style.cursor = 'ns-resize';
        } else {
          histCanvas.style.cursor = 'default';
        }
      });

      histCanvas.addEventListener('mouseleave', () => {
        if (!histDrag) histCanvas.style.cursor = 'default';
      });

      // ── marker overlay ─────────────────────────────────────────────────────
      // markers_json is a JSON array of marker-set objects. Each has a 'type'
      // field and type-specific data, mirroring the HyperSpy Markers API.
      //
      // circles:    { type:'circles',    offsets:[[x,y],...], sizes:[r,...],       color, linewidth }
      // arrows:     { type:'arrows',     offsets:[[x,y],...], U:[...], V:[...],    color, linewidth }
      // ellipses:   { type:'ellipses',   offsets:[[x,y],...], widths:[...], heights:[...], angles:[...], color, linewidth }
      // lines:      { type:'lines',      segments:[[[x1,y1],[x2,y2]],...],         color, linewidth }
      // rectangles: { type:'rectangles', offsets:[[x,y],...], widths:[...], heights:[...], angles:[...], color, linewidth }
      // squares:    { type:'squares',    offsets:[[x,y],...], widths:[...],  angles:[...], color, linewidth }
      // texts:      { type:'texts',      offsets:[[x,y],...], texts:[...],          color, fontsize }
      //
      // All spatial values are in image-pixel units.

      function drawMarkers() {
        const cw = parseInt(markersCanvas.style.width)  || model.get('viewer_width');
        const ch = parseInt(markersCanvas.style.height) || model.get('viewer_height');
        mkCtx.clearRect(0, 0, cw, ch);

        let sets;
        try { sets = JSON.parse(model.get('markers_json')); }
        catch (_) { return; }
        if (!Array.isArray(sets) || sets.length === 0) return;

        const scale = _imgScale();

        for (const ms of sets) {
          const color     = ms.color     || '#ff0000';
          const linewidth = ms.linewidth != null ? ms.linewidth : 1.5;
          const type      = ms.type || 'circles';
          const fillColor = ms.fill_color || null;
          const fillAlpha = ms.fill_alpha != null ? ms.fill_alpha : 0.3;

          mkCtx.save();
          mkCtx.strokeStyle = color;
          mkCtx.fillStyle   = color;
          mkCtx.lineWidth   = linewidth;

          if (type === 'circles') {
            const offsets = ms.offsets || [];
            const sizes   = ms.sizes   || [];
            for (let i = 0; i < offsets.length; i++) {
              const [cx, cy] = _imgToCanvas(offsets[i][0], offsets[i][1]);
              const r = (sizes[i] != null ? sizes[i] : (sizes[0] != null ? sizes[0] : 5)) * scale;
              mkCtx.beginPath();
              mkCtx.arc(cx, cy, Math.max(1, r), 0, Math.PI * 2);
              if (fillColor) {
                mkCtx.save();
                mkCtx.globalAlpha = fillAlpha;
                mkCtx.fillStyle = fillColor;
                mkCtx.fill();
                mkCtx.restore();
              }
              mkCtx.stroke();
            }

          } else if (type === 'arrows') {
            const offsets = ms.offsets || [];
            const Us = ms.U || [], Vs = ms.V || [];
            const headLen = 8; // canvas px
            for (let i = 0; i < offsets.length; i++) {
              const [x1, y1] = _imgToCanvas(offsets[i][0], offsets[i][1]);
              const u = (Us[i] != null ? Us[i] : 0) * scale;
              const v = (Vs[i] != null ? Vs[i] : 0) * scale;
              const x2 = x1 + u, y2 = y1 + v;
              const angle = Math.atan2(y2 - y1, x2 - x1);
              // shaft
              mkCtx.beginPath();
              mkCtx.moveTo(x1, y1);
              mkCtx.lineTo(x2, y2);
              mkCtx.stroke();
              // arrowhead
              mkCtx.beginPath();
              mkCtx.moveTo(x2, y2);
              mkCtx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6),
                           y2 - headLen * Math.sin(angle - Math.PI / 6));
              mkCtx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6),
                           y2 - headLen * Math.sin(angle + Math.PI / 6));
              mkCtx.closePath();
              mkCtx.fill();
            }

          } else if (type === 'ellipses') {
            const offsets  = ms.offsets  || [];
            const widths   = ms.widths   || [];
            const heights  = ms.heights  || [];
            const angles   = ms.angles   || [];
            for (let i = 0; i < offsets.length; i++) {
              const [cx, cy] = _imgToCanvas(offsets[i][0], offsets[i][1]);
              const rw  = (widths[i]  != null ? widths[i]  : (widths[0]  != null ? widths[0]  : 10)) * scale / 2;
              const rh  = (heights[i] != null ? heights[i] : (heights[0] != null ? heights[0] : 10)) * scale / 2;
              const ang = ((angles[i] != null ? angles[i]  : (angles[0]  != null ? angles[0]  : 0)) * Math.PI) / 180;
              mkCtx.beginPath();
              mkCtx.ellipse(cx, cy, Math.max(1, rw), Math.max(1, rh), ang, 0, Math.PI * 2);
              if (fillColor) {
                mkCtx.save();
                mkCtx.globalAlpha = fillAlpha;
                mkCtx.fillStyle = fillColor;
                mkCtx.fill();
                mkCtx.restore();
              }
              mkCtx.stroke();
            }

          } else if (type === 'lines') {
            const segments = ms.segments || [];
            for (const seg of segments) {
              const [x1, y1] = _imgToCanvas(seg[0][0], seg[0][1]);
              const [x2, y2] = _imgToCanvas(seg[1][0], seg[1][1]);
              mkCtx.beginPath();
              mkCtx.moveTo(x1, y1);
              mkCtx.lineTo(x2, y2);
              mkCtx.stroke();
            }

          } else if (type === 'rectangles') {
            const offsets  = ms.offsets  || [];
            const widths   = ms.widths   || [];
            const heights  = ms.heights  || [];
            const angles   = ms.angles   || [];
            for (let i = 0; i < offsets.length; i++) {
              const [cx, cy] = _imgToCanvas(offsets[i][0], offsets[i][1]);
              const rw  = (widths[i]  != null ? widths[i]  : (widths[0]  != null ? widths[0]  : 20)) * scale;
              const rh  = (heights[i] != null ? heights[i] : (heights[0] != null ? heights[0] : 20)) * scale;
              const ang = ((angles[i] != null ? angles[i]  : (angles[0]  != null ? angles[0]  : 0)) * Math.PI) / 180;
              mkCtx.save();
              mkCtx.translate(cx, cy);
              mkCtx.rotate(ang);
              if (fillColor) {
                mkCtx.save();
                mkCtx.globalAlpha = fillAlpha;
                mkCtx.fillStyle = fillColor;
                mkCtx.fillRect(-rw / 2, -rh / 2, rw, rh);
                mkCtx.restore();
              }
              mkCtx.strokeRect(-rw / 2, -rh / 2, rw, rh);
              mkCtx.restore();
            }

          } else if (type === 'squares') {
            const offsets = ms.offsets || [];
            const widths  = ms.widths  || [];
            const angles  = ms.angles  || [];
            for (let i = 0; i < offsets.length; i++) {
              const [cx, cy] = _imgToCanvas(offsets[i][0], offsets[i][1]);
              const side = (widths[i] != null ? widths[i] : (widths[0] != null ? widths[0] : 20)) * scale;
              const ang  = ((angles[i] != null ? angles[i] : (angles[0] != null ? angles[0] : 0)) * Math.PI) / 180;
              mkCtx.save();
              mkCtx.translate(cx, cy);
              mkCtx.rotate(ang);
              if (fillColor) {
                mkCtx.save();
                mkCtx.globalAlpha = fillAlpha;
                mkCtx.fillStyle = fillColor;
                mkCtx.fillRect(-side / 2, -side / 2, side, side);
                mkCtx.restore();
              }
              mkCtx.strokeRect(-side / 2, -side / 2, side, side);
              mkCtx.restore();
            }

          } else if (type === 'texts') {
            const offsets  = ms.offsets || [];
            const texts    = ms.texts   || [];
            const fontsize = ms.fontsize != null ? ms.fontsize : 12;
            mkCtx.font      = `${fontsize}px sans-serif`;
            mkCtx.textAlign = 'left';
            mkCtx.textBaseline = 'top';
            for (let i = 0; i < offsets.length; i++) {
              const [cx, cy] = _imgToCanvas(offsets[i][0], offsets[i][1]);
              const label = texts[i] != null ? String(texts[i]) : '';
              mkCtx.fillText(label, cx, cy);
            }

          } else if (type === 'polygons') {
            // Each entry in ms.vertices_list is one polygon: [[x,y], ...]
            const vertsList = ms.vertices_list || [];
            for (let i = 0; i < vertsList.length; i++) {
              const verts = vertsList[i];
              if (!verts || verts.length < 2) continue;
              mkCtx.beginPath();
              const [px0, py0] = _imgToCanvas(verts[0][0], verts[0][1]);
              mkCtx.moveTo(px0, py0);
              for (let k = 1; k < verts.length; k++) {
                const [px, py] = _imgToCanvas(verts[k][0], verts[k][1]);
                mkCtx.lineTo(px, py);
              }
              mkCtx.closePath();
              if (fillColor) {
                mkCtx.save();
                mkCtx.globalAlpha = fillAlpha;
                mkCtx.fillStyle = fillColor;
                mkCtx.fill();
                mkCtx.restore();
              }
              mkCtx.stroke();
            }
          }

          mkCtx.restore();
        }
      }

      // ── overlay widgets ────────────────────────────────────────────────────
      // Shapes: { id, type:'circle',    cx, cy, r,  color }
      //         { id, type:'rectangle', x,  y,  w,  h, color }
      // All coordinates are in image-pixel space.
      const HANDLE_R    = 7;   // hit-test radius (canvas px)
      const HANDLE_DRAW = 5;   // visual radius

      function _imgToCanvas(ix, iy) {
        const zoom = model.get('zoom'), cx = model.get('center_x'), cy = model.get('center_y');
        const cw = parseInt(overlayCanvas.style.width)  || model.get('viewer_width');
        const ch = parseInt(overlayCanvas.style.height) || model.get('viewer_height');
        const iw = model.get('image_width'), ih = model.get('image_height');
        if (zoom >= 1.0) {
          const visW = iw / zoom, visH = ih / zoom;
          const srcX = Math.max(0, Math.min(iw - visW, cx * iw - visW / 2));
          const srcY = Math.max(0, Math.min(ih - visH, cy * ih - visH / 2));
          return [(ix - srcX) / visW * cw, (iy - srcY) / visH * ch];
        } else {
          const dstW = cw * zoom, dstH = ch * zoom;
          return [(cw - dstW) / 2 + (ix / iw) * dstW,
                  (ch - dstH) / 2 + (iy / ih) * dstH];
        }
      }

      function _canvasToImg(px, py) {
        const zoom = model.get('zoom'), cx = model.get('center_x'), cy = model.get('center_y');
        const cw = parseInt(overlayCanvas.style.width)  || model.get('viewer_width');
        const ch = parseInt(overlayCanvas.style.height) || model.get('viewer_height');
        const iw = model.get('image_width'), ih = model.get('image_height');
        if (zoom >= 1.0) {
          const visW = iw / zoom, visH = ih / zoom;
          const srcX = Math.max(0, Math.min(iw - visW, cx * iw - visW / 2));
          const srcY = Math.max(0, Math.min(ih - visH, cy * ih - visH / 2));
          return [srcX + (px / cw) * visW, srcY + (py / ch) * visH];
        } else {
          const dstW = cw * zoom, dstH = ch * zoom;
          return [((px - (cw - dstW) / 2) / dstW) * iw,
                  ((py - (ch - dstH) / 2) / dstH) * ih];
        }
      }

      function _imgScale() {
        const zoom = model.get('zoom');
        const cw   = parseInt(overlayCanvas.style.width)  || model.get('viewer_width');
        const iw   = model.get('image_width');
        return zoom >= 1.0 ? cw / (iw / zoom) : cw * zoom / iw;
      }

      function _drawHandle(x, y, color) {
        ovCtx.save();
        ovCtx.fillStyle   = '#ffffff';
        ovCtx.strokeStyle = color || '#00e5ff';
        ovCtx.lineWidth   = 1.5;
        ovCtx.beginPath();
        ovCtx.arc(x, y, HANDLE_DRAW, 0, Math.PI * 2);
        ovCtx.fill();
        ovCtx.stroke();
        ovCtx.restore();
      }

      function drawOverlay() {
        const cw = parseInt(overlayCanvas.style.width)  || model.get('viewer_width');
        const ch = parseInt(overlayCanvas.style.height) || model.get('viewer_height');
        ovCtx.clearRect(0, 0, cw, ch);
        const widgets = JSON.parse(model.get('overlay_widgets'));
        const scale   = _imgScale();
        for (const w of widgets) {
          ovCtx.save();
          ovCtx.strokeStyle = w.color || '#00e5ff';
          ovCtx.lineWidth   = 2;

          if (w.type === 'circle') {
            const [ccx, ccy] = _imgToCanvas(w.cx, w.cy);
            const cr = w.r * scale;
            ovCtx.beginPath();
            ovCtx.arc(ccx, ccy, cr, 0, Math.PI * 2);
            ovCtx.stroke();
            _drawHandle(ccx + cr, ccy, w.color);  // outer resize handle (right)

          } else if (w.type === 'annular') {
            // Two concentric circles: outer radius r_outer, inner radius r_inner
            const [ccx, ccy] = _imgToCanvas(w.cx, w.cy);
            const ro = w.r_outer * scale;
            const ri = w.r_inner * scale;
            ovCtx.beginPath();
            ovCtx.arc(ccx, ccy, ro, 0, Math.PI * 2);
            ovCtx.stroke();
            ovCtx.beginPath();
            ovCtx.arc(ccx, ccy, ri, 0, Math.PI * 2);
            ovCtx.stroke();
            _drawHandle(ccx + ro, ccy, w.color);              // outer resize
            _drawHandle(ccx + ri, ccy - ri * 0.3, w.color);  // inner resize (offset so handles don't overlap when ri≈ro)

          } else if (w.type === 'rectangle') {
            const [rx, ry] = _imgToCanvas(w.x, w.y);
            const rw = w.w * scale, rh = w.h * scale;
            ovCtx.strokeRect(rx, ry, rw, rh);
            _drawHandle(rx,      ry,      w.color);
            _drawHandle(rx + rw, ry,      w.color);
            _drawHandle(rx,      ry + rh, w.color);
            _drawHandle(rx + rw, ry + rh, w.color);

          } else if (w.type === 'polygon') {
            const verts = w.vertices || [];
            if (verts.length >= 2) {
              ovCtx.beginPath();
              const [px0, py0] = _imgToCanvas(verts[0][0], verts[0][1]);
              ovCtx.moveTo(px0, py0);
              for (let k = 1; k < verts.length; k++) {
                const [px, py] = _imgToCanvas(verts[k][0], verts[k][1]);
                ovCtx.lineTo(px, py);
              }
              ovCtx.closePath();
              ovCtx.stroke();
              // vertex handles
              for (const v of verts) {
                const [px, py] = _imgToCanvas(v[0], v[1]);
                _drawHandle(px, py, w.color);
              }
            }

          } else if (w.type === 'label') {
            const [lx, ly] = _imgToCanvas(w.x, w.y);
            const fs = (w.fontsize || 14);
            ovCtx.font      = `${fs}px sans-serif`;
            ovCtx.fillStyle = w.color || '#00e5ff';
            ovCtx.textAlign    = 'left';
            ovCtx.textBaseline = 'top';
            ovCtx.fillText(w.text || '', lx, ly);
            // small drag handle at anchor point
            _drawHandle(lx, ly, w.color);

          } else if (w.type === 'crosshair') {
            // "+" widget — full-canvas crosshair through (cx, cy); move-only.
            const [ccx, ccy] = _imgToCanvas(w.cx, w.cy);
            const cw2 = parseInt(overlayCanvas.style.width)  || model.get('viewer_width');
            const ch2 = parseInt(overlayCanvas.style.height) || model.get('viewer_height');
            // Horizontal arm
            ovCtx.beginPath(); ovCtx.moveTo(0, ccy); ovCtx.lineTo(cw2, ccy); ovCtx.stroke();
            // Vertical arm
            ovCtx.beginPath(); ovCtx.moveTo(ccx, 0); ovCtx.lineTo(ccx, ch2); ovCtx.stroke();
            // Centre dot (filled, slightly larger than stroke width)
            const dotR = Math.max(3, ovCtx.lineWidth * 1.5);
            ovCtx.beginPath(); ovCtx.arc(ccx, ccy, dotR, 0, Math.PI * 2);
            ovCtx.fillStyle = w.color || '#00e5ff';
            ovCtx.fill();
          }

          ovCtx.restore();
        }
      }

      // ── overlay hit-test & drag ────────────────────────────────────────────
      let ovDrag = null;

      function _handles(w) {
        const scale = _imgScale();
        if (w.type === 'circle') {
          const [ccx, ccy] = _imgToCanvas(w.cx, w.cy);
          return [{ x: ccx + w.r * scale, y: ccy }];
        }
        if (w.type === 'annular') {
          const [ccx, ccy] = _imgToCanvas(w.cx, w.cy);
          const ro = w.r_outer * scale, ri = w.r_inner * scale;
          return [
            { x: ccx + ro, y: ccy },
            { x: ccx + ri, y: ccy - ri * 0.3 },
          ];
        }
        if (w.type === 'polygon') {
          const verts = w.vertices || [];
          return verts.map(v => { const [px,py]=_imgToCanvas(v[0],v[1]); return {x:px,y:py}; });
        }
        if (w.type === 'label') {
          const [lx, ly] = _imgToCanvas(w.x, w.y);
          return [{ x: lx, y: ly }];
        }
        if (w.type === 'crosshair') {
          // No resize handles — crosshair is move-only.
          return [];
        }
        // rectangle (default)
        const [rx, ry] = _imgToCanvas(w.x, w.y);
        const rw = w.w * scale, rh = w.h * scale;
        return [{ x: rx, y: ry }, { x: rx + rw, y: ry },
                { x: rx, y: ry + rh }, { x: rx + rw, y: ry + rh }];
      }

      function _hitTest(ex, ey) {
        const widgets = JSON.parse(model.get('overlay_widgets'));
        const rect    = imageCanvas.getBoundingClientRect();
        const mx = ex - rect.left, my = ey - rect.top;
        // Handles first (highest priority)
        for (let i = widgets.length - 1; i >= 0; i--) {
          const hs = _handles(widgets[i]);
          for (let hi = 0; hi < hs.length; hi++) {
            const dx = mx - hs[hi].x, dy = my - hs[hi].y;
            if (Math.sqrt(dx * dx + dy * dy) <= HANDLE_R)
              return { idx: i, mode: 'resize', hi };
          }
        }
        // Body hit
        const scale = _imgScale();
        for (let i = widgets.length - 1; i >= 0; i--) {
          const w = widgets[i];
          if (w.type === 'circle') {
            const [ccx, ccy] = _imgToCanvas(w.cx, w.cy);
            const dx = mx - ccx, dy = my - ccy;
            if (Math.sqrt(dx * dx + dy * dy) <= w.r * scale + 4)
              return { idx: i, mode: 'move', hi: -1 };
          } else if (w.type === 'annular') {
            const [ccx, ccy] = _imgToCanvas(w.cx, w.cy);
            const dist = Math.sqrt((mx-ccx)**2 + (my-ccy)**2);
            if (dist <= w.r_outer * scale + 4)
              return { idx: i, mode: 'move', hi: -1 };
          } else if (w.type === 'polygon') {
            // point-in-polygon (ray casting)
            const verts = w.vertices || [];
            let inside = false;
            for (let a = 0, b = verts.length - 1; a < verts.length; b = a++) {
              const [ax, ay] = _imgToCanvas(verts[a][0], verts[a][1]);
              const [bx, by] = _imgToCanvas(verts[b][0], verts[b][1]);
              if (((ay > my) !== (by > my)) && (mx < (bx - ax) * (my - ay) / (by - ay) + ax))
                inside = !inside;
            }
            if (inside) return { idx: i, mode: 'move', hi: -1 };
          } else if (w.type === 'label') {
            const [lx, ly] = _imgToCanvas(w.x, w.y);
            // generous hit box around the anchor handle
            if (Math.abs(mx - lx) <= 20 && Math.abs(my - ly) <= 20)
              return { idx: i, mode: 'move', hi: -1 };
          } else if (w.type === 'crosshair') {
            // Hit within 10 px of the centre point OR along either arm (±4 px).
            const [ccx, ccy] = _imgToCanvas(w.cx, w.cy);
            const nearCentre = Math.sqrt((mx-ccx)**2 + (my-ccy)**2) <= 10;
            const nearH = Math.abs(my - ccy) <= 4;
            const nearV = Math.abs(mx - ccx) <= 4;
            if (nearCentre || nearH || nearV)
              return { idx: i, mode: 'move', hi: -1 };
          } else {
            // rectangle
            const [rx, ry] = _imgToCanvas(w.x, w.y);
            const rw = w.w * scale, rh = w.h * scale;
            if (mx >= rx && mx <= rx + rw && my >= ry && my <= ry + rh)
              return { idx: i, mode: 'move', hi: -1 };
          }
        }
        return null;
      }

      // mousedown – check widgets before pan
      imageCanvas.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        imageCanvas.focus();
        const hit = _hitTest(e.clientX, e.clientY);
        if (hit) {
          e.preventDefault();
          e.stopImmediatePropagation();
          const rect = imageCanvas.getBoundingClientRect();
          const [ix, iy] = _canvasToImg(e.clientX - rect.left, e.clientY - rect.top);
          const widgets = JSON.parse(model.get('overlay_widgets'));
          ovDrag = { ...hit, startIX: ix, startIY: iy,
                     snap: JSON.parse(JSON.stringify(widgets[hit.idx])) };
          imageCanvas.style.cursor = hit.mode === 'resize' ? 'nwse-resize' : 'move';
          return;
        }
        // fall through to pan
        isPanning = true;
        panStartX = e.clientX; panStartY = e.clientY;
        panStartCenterX = model.get('center_x');
        panStartCenterY = model.get('center_y');
        imageCanvas.style.cursor = 'grabbing';
        e.preventDefault();
      }, true);   // capture so we beat the pan listener

      document.addEventListener('mousemove', (e) => {
        if (!ovDrag) return;
        const rect = imageCanvas.getBoundingClientRect();
        const [ix, iy] = _canvasToImg(e.clientX - rect.left, e.clientY - rect.top);
        const dx = ix - ovDrag.startIX, dy = iy - ovDrag.startIY;
        const widgets = JSON.parse(model.get('overlay_widgets'));
        const s = ovDrag.snap;

        if (ovDrag.mode === 'move') {
          if (s.type === 'circle') {
            widgets[ovDrag.idx].cx = s.cx + dx;
            widgets[ovDrag.idx].cy = s.cy + dy;
          } else if (s.type === 'annular') {
            widgets[ovDrag.idx].cx = s.cx + dx;
            widgets[ovDrag.idx].cy = s.cy + dy;
          } else if (s.type === 'crosshair') {
            widgets[ovDrag.idx].cx = s.cx + dx;
            widgets[ovDrag.idx].cy = s.cy + dy;
          } else if (s.type === 'polygon') {
            widgets[ovDrag.idx].vertices = s.vertices.map(v => [v[0]+dx, v[1]+dy]);
          } else if (s.type === 'label') {
            widgets[ovDrag.idx].x = s.x + dx;
            widgets[ovDrag.idx].y = s.y + dy;
          } else {
            widgets[ovDrag.idx].x = s.x + dx;
            widgets[ovDrag.idx].y = s.y + dy;
          }
        } else {
          // resize
          if (s.type === 'circle') {
            widgets[ovDrag.idx].r = Math.max(2,
              Math.sqrt((ix - s.cx) ** 2 + (iy - s.cy) ** 2));
          } else if (s.type === 'annular') {
            const dist = Math.sqrt((ix - s.cx) ** 2 + (iy - s.cy) ** 2);
            if (ovDrag.hi === 0) {
              // outer handle
              widgets[ovDrag.idx].r_outer = Math.max(s.r_inner + 2, dist);
            } else {
              // inner handle
              widgets[ovDrag.idx].r_inner = Math.max(2, Math.min(s.r_outer - 2, dist));
            }
          } else if (s.type === 'polygon') {
            // move the specific vertex
            const verts = s.vertices.map(v => [v[0], v[1]]);
            verts[ovDrag.hi] = [ix, iy];
            widgets[ovDrag.idx].vertices = verts;
          } else if (s.type === 'label') {
            // labels don't have a separate resize handle – treat as move
            widgets[ovDrag.idx].x = s.x + dx;
            widgets[ovDrag.idx].y = s.y + dy;
          } else {
            let nx = s.x, ny = s.y, nw = s.w, nh = s.h;
            if      (ovDrag.hi === 0) { nx = s.x+dx; ny = s.y+dy; nw = s.w-dx; nh = s.h-dy; }
            else if (ovDrag.hi === 1) {               ny = s.y+dy; nw = s.w+dx; nh = s.h-dy; }
            else if (ovDrag.hi === 2) { nx = s.x+dx;               nw = s.w-dx; nh = s.h+dy; }
            else                      {                              nw = s.w+dx; nh = s.h+dy; }
            widgets[ovDrag.idx].x = nx; widgets[ovDrag.idx].y = ny;
            widgets[ovDrag.idx].w = Math.max(4, nw);
            widgets[ovDrag.idx].h = Math.max(4, nh);
          }
        }
        model.set('overlay_widgets', JSON.stringify(widgets));
        model.save_changes();
        drawOverlay();
        e.preventDefault();
      }, true);

      document.addEventListener('mouseup', (e) => {
        if (!ovDrag) return;
        ovDrag = null;
        imageCanvas.style.cursor = 'default';
        e.preventDefault();
      }, true);

      // ── resize ─────────────────────────────────────────────────────────────
      resizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX; startY = e.clientY;
        startWidth  = parseInt(imageCanvas.style.width);
        startHeight = parseInt(imageCanvas.style.height);
        sizeLabel.style.display = 'block';
        e.preventDefault();
      });

      function _applyCanvasSize(nw, nh) {
        // Apply a new canvas size to every canvas in sync.
        nw = Math.round(nw); nh = Math.round(nh);
        imageCanvas.style.width  = nw + 'px'; imageCanvas.style.height = nh + 'px';
        imageCanvas.width = nw * dpr; imageCanvas.height = nh * dpr;
        imgCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        imgCtx.imageSmoothingEnabled = false;
        overlayCanvas.style.width  = nw + 'px'; overlayCanvas.style.height = nh + 'px';
        overlayCanvas.width = nw * dpr; overlayCanvas.height = nh * dpr;
        ovCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        markersCanvas.style.width  = nw + 'px'; markersCanvas.style.height = nh + 'px';
        markersCanvas.width = nw * dpr; markersCanvas.height = nh * dpr;
        mkCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        histCanvas.style.height = nh + 'px'; histCanvas.height = nh * dpr;
        histCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        if (!model.get('use_scalebar')) {
          xAxisCanvas.style.width  = nw + 'px'; xAxisCanvas.width  = nw * dpr;
          yAxisCanvas.style.height = nh + 'px'; yAxisCanvas.height = nh * dpr;
          xCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
          yCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
      }

      document.addEventListener('mousemove', (e) => {
        if (isResizing) {
          const dX = e.clientX - startX, dY = e.clientY - startY;
          const imgW = model.get('image_width'), imgH = model.get('image_height');
          let nw, nh;
          if (imgW > 0 && imgH > 0) {
            const ar = imgW / imgH, avg = (dX + dY) / 2;
            nw = Math.max(128, startWidth + avg);
            nh = nw / ar;
            if (nh < 128) { nh = 128; nw = nh * ar; }
          } else {
            nw = Math.max(128, startWidth  + dX);
            nh = Math.max(128, startHeight + dY);
          }
          _applyCanvasSize(nw, nh);
          sizeLabel.textContent = `${Math.round(nw)} × ${Math.round(nh)}`;
          drawImage(); drawHistogram();
          e.preventDefault();
        } else if (isPanning) {
          const rect = imageCanvas.getBoundingClientRect();
          const z = model.get('zoom');
          model.set('center_x', Math.max(0, Math.min(1, panStartCenterX - (e.clientX - panStartX) / rect.width  / z)));
          model.set('center_y', Math.max(0, Math.min(1, panStartCenterY - (e.clientY - panStartY) / rect.height / z)));
          model.save_changes();
          e.preventDefault();
        }
      });

      document.addEventListener('mouseup', (e) => {
        if (isResizing) {
          isResizing = false;
          sizeLabel.style.display = 'none';
          // Read the final canvas size, recompute height from aspect ratio so
          // the two traits are always consistent before saving.
          const nw = parseInt(imageCanvas.style.width);
          const nh = parseInt(imageCanvas.style.height);
          // Suppress the change listeners while we set both traits at once
          // so syncCanvasSizes doesn't fire mid-update with mismatched values.
          _suppressSyncResize = true;
          model.set('viewer_width',  nw);
          model.set('viewer_height', nh);
          model.save_changes();
          _suppressSyncResize = false;
        }
        if (isPanning) {
          isPanning = false;
          imageCanvas.style.cursor = 'default';
          const rect = imageCanvas.getBoundingClientRect();
          const z = model.get('zoom');
          model.set('center_x', Math.max(0, Math.min(1, panStartCenterX - (e.clientX - panStartX) / rect.width  / z)));
          model.set('center_y', Math.max(0, Math.min(1, panStartCenterY - (e.clientY - panStartY) / rect.height / z)));
          model.save_changes();
        }
      });


      imageCanvas.addEventListener('mouseenter', () => imageCanvas.focus());

      // ── wheel zoom ─────────────────────────────────────────────────────────
      imageCanvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const rect = imageCanvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) / rect.width;
        const my = (e.clientY - rect.top)  / rect.height;
        const curZ = model.get('zoom');
        const newZ = Math.max(0.75, Math.min(100, curZ * (e.deltaY > 0 ? 0.9 : 1.1)));
        const ratio = curZ / newZ;
        model.set('zoom', newZ);
        model.set('center_x', Math.max(0, Math.min(1, model.get('center_x') + (mx - 0.5) * (1 - ratio) / newZ)));
        model.set('center_y', Math.max(0, Math.min(1, model.get('center_y') + (my - 0.5) * (1 - ratio) / newZ)));
        model.save_changes();
      });

      // ── keyboard shortcuts ─────────────────────────────────────────────────
      imageCanvas.addEventListener('keydown', (e) => {
        const key = e.key.toLowerCase();
        if (key === 'r') {
          // Reset zoom, pan, and display clim to defaults.
          model.set('zoom', 1.0);
          model.set('center_x', 0.5);
          model.set('center_y', 0.5);
          model.set('display_min', model.get('hist_min'));
          model.set('display_max', model.get('hist_max'));
          model.save_changes();
          e.preventDefault();
        } else if (key === 'l') {
          // Toggle between linear and log scale.
          const next = model.get('scale_mode') === 'log' ? 'linear' : 'log';
          model.set('scale_mode', next);
          model.save_changes();
          e.preventDefault();
        } else if (key === 's') {
          // Toggle between linear and symlog scale.
          const next = model.get('scale_mode') === 'symlog' ? 'linear' : 'symlog';
          model.set('scale_mode', next);
          model.save_changes();
          e.preventDefault();
        } else if (key === 'h') {
          model.set('histogram_visible', !model.get('histogram_visible'));
          model.save_changes();
          e.preventDefault();
        }
      });

      // ── marker hover / tooltip ─────────────────────────────────────────────
      // Each marker set may carry:
      //   label   : string  – shown when hovering anywhere over the collection
      //   labels  : [str, …] – per-element label; shown instead of / in addition to label
      //
      // Hit-test tolerance in canvas px:
      const MARKER_HIT = 8;

      function _markerHitTest(mx, my) {
        // Returns { collectionLabel, markerLabel, x, y } or null.
        let sets;
        try { sets = JSON.parse(model.get('markers_json')); }
        catch (_) { return null; }
        if (!Array.isArray(sets)) return null;

        const scale = _imgScale();

        // Iterate in reverse so topmost set wins
        for (let si = sets.length - 1; si >= 0; si--) {
          const ms   = sets[si];
          const type = ms.type || 'circles';
          const collLabel = ms.label != null ? String(ms.label) : null;
          const perLabels = Array.isArray(ms.labels) ? ms.labels : null;

          // Skip sets that have neither a collection label nor per-marker labels
          if (collLabel === null && perLabels === null) continue;

          if (type === 'circles') {
            const offsets = ms.offsets || [], sizes = ms.sizes || [];
            for (let i = 0; i < offsets.length; i++) {
              const [cx, cy] = _imgToCanvas(offsets[i][0], offsets[i][1]);
              const r = (sizes[i] != null ? sizes[i] : (sizes[0] != null ? sizes[0] : 5)) * scale;
              if (Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2) <= r + MARKER_HIT)
                return { collectionLabel: collLabel,
                         markerLabel: perLabels ? String(perLabels[i] ?? '') : null,
                         x: cx, y: cy };
            }

          } else if (type === 'arrows') {
            const offsets = ms.offsets || [], Us = ms.U || [], Vs = ms.V || [];
            for (let i = 0; i < offsets.length; i++) {
              const [x1, y1] = _imgToCanvas(offsets[i][0], offsets[i][1]);
              const u = (Us[i] != null ? Us[i] : 0) * scale;
              const v = (Vs[i] != null ? Vs[i] : 0) * scale;
              // hit if within MARKER_HIT px of the shaft midpoint
              const midx = x1 + u / 2, midy = y1 + v / 2;
              if (Math.sqrt((mx - midx) ** 2 + (my - midy) ** 2) <= Math.max(MARKER_HIT, Math.sqrt(u*u+v*v) / 2 + 4))
                return { collectionLabel: collLabel,
                         markerLabel: perLabels ? String(perLabels[i] ?? '') : null,
                         x: midx, y: midy };
            }

          } else if (type === 'ellipses') {
            const offsets = ms.offsets || [], widths = ms.widths || [], heights = ms.heights || [];
            for (let i = 0; i < offsets.length; i++) {
              const [cx, cy] = _imgToCanvas(offsets[i][0], offsets[i][1]);
              const rw = (widths[i]  != null ? widths[i]  : (widths[0]  != null ? widths[0]  : 10)) * scale / 2;
              const rh = (heights[i] != null ? heights[i] : (heights[0] != null ? heights[0] : 10)) * scale / 2;
              const dx = (mx - cx) / Math.max(1, rw), dy = (my - cy) / Math.max(1, rh);
              if (dx * dx + dy * dy <= 1.2)
                return { collectionLabel: collLabel,
                         markerLabel: perLabels ? String(perLabels[i] ?? '') : null,
                         x: cx, y: cy };
            }

          } else if (type === 'lines') {
            const segs = ms.segments || [];
            for (let i = 0; i < segs.length; i++) {
              const [x1, y1] = _imgToCanvas(segs[i][0][0], segs[i][0][1]);
              const [x2, y2] = _imgToCanvas(segs[i][1][0], segs[i][1][1]);
              // point-to-segment distance
              const len2 = (x2-x1)**2 + (y2-y1)**2;
              let t = len2 > 0 ? Math.max(0, Math.min(1, ((mx-x1)*(x2-x1)+(my-y1)*(y2-y1)) / len2)) : 0;
              const px = x1 + t*(x2-x1), py = y1 + t*(y2-y1);
              if (Math.sqrt((mx-px)**2 + (my-py)**2) <= MARKER_HIT)
                return { collectionLabel: collLabel,
                         markerLabel: perLabels ? String(perLabels[i] ?? '') : null,
                         x: (x1+x2)/2, y: (y1+y2)/2 };
            }

          } else if (type === 'rectangles' || type === 'squares') {
            const offsets = ms.offsets || [];
            const widths  = ms.widths  || [];
            const heights = type === 'rectangles' ? (ms.heights || []) : widths;
            for (let i = 0; i < offsets.length; i++) {
              const [cx, cy] = _imgToCanvas(offsets[i][0], offsets[i][1]);
              const hw = (widths[i]  != null ? widths[i]  : (widths[0]  != null ? widths[0]  : 20)) * scale / 2;
              const hh = (heights[i] != null ? heights[i] : (heights[0] != null ? heights[0] : 20)) * scale / 2;
              if (Math.abs(mx - cx) <= hw + MARKER_HIT && Math.abs(my - cy) <= hh + MARKER_HIT)
                return { collectionLabel: collLabel,
                         markerLabel: perLabels ? String(perLabels[i] ?? '') : null,
                         x: cx, y: cy };
            }

          } else if (type === 'texts') {
            const offsets = ms.offsets || [], texts = ms.texts || [];
            for (let i = 0; i < offsets.length; i++) {
              const [cx, cy] = _imgToCanvas(offsets[i][0], offsets[i][1]);
              if (Math.abs(mx - cx) <= MARKER_HIT * 2 && Math.abs(my - cy) <= MARKER_HIT * 2)
                return { collectionLabel: collLabel,
                         markerLabel: perLabels ? String(perLabels[i] ?? '') : (texts[i] != null ? String(texts[i]) : null),
                         x: cx, y: cy };
            }
          }
        }
        return null;
      }

      function _showTooltip(text, clientX, clientY) {
        markerTooltip.textContent = text;
        markerTooltip.style.display = 'block';
        // Position just above-right of cursor, flip if near edge
        const tw = markerTooltip.offsetWidth  || 160;
        const th = markerTooltip.offsetHeight || 28;
        const vw = window.innerWidth, vh = window.innerHeight;
        let lx = clientX + 14, ly = clientY - th - 8;
        if (lx + tw > vw - 8) lx = clientX - tw - 14;
        if (ly < 8)            ly = clientY + 18;
        markerTooltip.style.left = lx + 'px';
        markerTooltip.style.top  = ly + 'px';
      }

      // ── status bar helpers ─────────────────────────────────────────────────
      function _fmtPhys(v) {
        // Format a physical coordinate value compactly.
        if (!isFinite(v)) return '?';
        const a = Math.abs(v);
        if (a === 0)      return '0';
        if (a >= 1e4)     return v.toExponential(2);
        if (a >= 100)     return v.toFixed(1);
        if (a >= 1)       return v.toFixed(2);
        if (a >= 1e-2)    return v.toFixed(4);
        return v.toExponential(2);
      }

      function _updateStatusBar(canvasMx, canvasMy) {
        const iw    = model.get('image_width');
        const ih    = model.get('image_height');
        const units = model.get('units');

        // Image-pixel coordinates (float, may be fractional).
        const [ix, iy] = _canvasToImg(canvasMx, canvasMy);
        const pixX = Math.floor(ix), pixY = Math.floor(iy);

        // Clamp to valid image area.
        if (pixX < 0 || pixX >= iw || pixY < 0 || pixY >= ih) {
          statusBar.style.display = 'none';
          return;
        }

        // Physical coordinates via axis arrays.
        const xArr = JSON.parse(model.get('x_axis_json'));
        const yArr = JSON.parse(model.get('y_axis_json'));
        const physX = xArr.length >= 2 ? _axisFracToVal(xArr, ix / iw) : ix;
        const physY = yArr.length >= 2 ? _axisFracToVal(yArr, iy / ih) : iy;

        // Pixel value from the raw uint8 image bytes.
        const raw = model.get('image_bytes');
        let bytes;
        if (raw instanceof Uint8Array)                     bytes = raw;
        else if (raw instanceof ArrayBuffer)               bytes = new Uint8Array(raw);
        else if (raw && raw.buffer instanceof ArrayBuffer) bytes = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
        else                                               bytes = new Uint8Array(0);

        // Recover the original data value from the stored uint8 via hist range.
        let valStr = '–';
        if (bytes.length === iw * ih) {
          const raw8  = bytes[pixY * iw + pixX];
          const hMin  = model.get('hist_min');
          const hMax  = model.get('hist_max');
          const val   = hMin + (raw8 / 255) * (hMax - hMin);
          valStr = _fmtPhys(val);
        }

        // Build the status string.
        const showPhys = units !== 'px';
        let xPart, yPart;
        if (showPhys) {
          xPart = `x: ${_fmtPhys(physX)} ${units} (${pixX} px)`;
          yPart = `y: ${_fmtPhys(physY)} ${units} (${pixY} px)`;
        } else {
          xPart = `x: ${pixX} px`;
          yPart = `y: ${pixY} px`;
        }

        statusBar.textContent = `${xPart}   ${yPart}   value: ${valStr}`;
        statusBar.style.display = 'block';
      }

      imageCanvas.addEventListener('mousemove', (e) => {
        const rect = imageCanvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;

        // Always update the status overlay, even while dragging/panning.
        _updateStatusBar(mx, my);

        if (ovDrag || isPanning || isResizing) {
          markerTooltip.style.display = 'none';
          return;
        }

        // Overlay widget cursor first
        const hit = _hitTest(e.clientX, e.clientY);
        if (hit) {
          imageCanvas.style.cursor = hit.mode === 'resize' ? 'nwse-resize' : 'move';
          markerTooltip.style.display = 'none';
          return;
        }
        imageCanvas.style.cursor = 'default';

        // Marker hover tooltip
        const mhit = _markerHitTest(mx, my);
        if (mhit) {
          const parts = [];
          if (mhit.collectionLabel) parts.push(mhit.collectionLabel);
          if (mhit.markerLabel)     parts.push(mhit.markerLabel);
          if (parts.length > 0) {
            _showTooltip(parts.join('\n'), e.clientX, e.clientY);
            return;
          }
        }
        markerTooltip.style.display = 'none';
      });

      imageCanvas.addEventListener('mouseleave', () => {
        markerTooltip.style.display = 'none';
        statusBar.style.display = 'none';
      });

      // ── model listeners ────────────────────────────────────────────────────
      model.on('change:image_bytes',       drawImage);
      model.on('change:histogram_data',    () => { drawImage(); drawHistogram(); });
      model.on('change:log_scale',         drawHistogram);
      model.on('change:scale_mode',        () => { drawImage(); drawHistogram(); });
      model.on('change:display_min',       () => { drawImage(); drawHistogram(); });
      model.on('change:display_max',       () => { drawImage(); drawHistogram(); });
      model.on('change:colormap_data',     () => { drawImage(); drawHistogram(); });
      model.on('change:show_colorbar',     drawHistogram);
      model.on('change:x_axis_json',       () => { if (!model.get('use_scalebar')) drawAxes(); });
      model.on('change:y_axis_json',       () => { if (!model.get('use_scalebar')) drawAxes(); });
      model.on('change:use_scalebar',      () => { syncCanvasSizes(); drawImage(); drawHistogram(); });
      model.on('change:histogram_visible', () => {
        histCanvas.style.display = model.get('histogram_visible') ? 'block' : 'none';
        container.style.gap = model.get('histogram_visible') ? model.get('gap') + 'px' : '0px';
        drawHistogram();
      });
      model.on('change:viewer_width change:viewer_height', () => {
        if (_suppressSyncResize) return;
        // Both traits must match before we resize — debounce so if width and
        // height fire in the same tick we only run once with both settled.
        const w = model.get('viewer_width');
        const h = model.get('viewer_height');
        const curW = parseInt(imageCanvas.style.width)  || 0;
        const curH = parseInt(imageCanvas.style.height) || 0;
        if (curW === w && curH === h) return;
        syncCanvasSizes(); drawImage(); drawHistogram();
      });
      model.on('change:zoom', () => {
        drawImage();
      });
      model.on('change:center_x', drawImage);
      model.on('change:center_y', drawImage);
      model.on('change:scale_x',  drawScaleBar);
      model.on('change:units',    () => { drawScaleBar(); if (!model.get('use_scalebar')) drawAxes(); });
      model.on('change:overlay_widgets', drawOverlay);
      model.on('change:markers_json',    drawMarkers);

      // ── initial render ─────────────────────────────────────────────────────
      drawImage();
      drawHistogram();
      drawOverlay();
      drawMarkers();
    }

    export default { render };
    """

    # ------------------------------------------------------------------ Python
    def __init__(
        self,
        data: np.ndarray,
        x_axis: np.ndarray | None = None,
        y_axis: np.ndarray | None = None,
        units: str = "px",
    ) -> None:
        """Initialise the viewer with a 2-D image array.

        Parameters
        ----------
        data :
            Grayscale image of shape ``(H, W)``.  A 3-D array ``(H, W, C)``
            is accepted; only the first channel is used.
        x_axis :
            Physical coordinates for each column (length ``W``).
            Defaults to ``np.arange(W)``.
        y_axis :
            Physical coordinates for each row (length ``H``).
            Defaults to ``np.arange(H)``.
        units :
            Unit label shown on the scale bar / axes (default ``'px'``).
        """
        super().__init__()

        data = np.asarray(data)
        if data.ndim == 3:
            data = data[:, :, 0]
        if data.ndim != 2:
            raise ValueError(f"data must be 2-D (H×W), got shape {data.shape}")

        h, w = data.shape

        if x_axis is None:
            x_axis = np.arange(w, dtype=float)
        if y_axis is None:
            y_axis = np.arange(h, dtype=float)

        x_axis = np.asarray(x_axis, dtype=float)
        y_axis = np.asarray(y_axis, dtype=float)

        equal = _equal_scale(x_axis, y_axis)
        dx = (x_axis[-1] - x_axis[0]) / (len(x_axis) - 1) if len(x_axis) > 1 else 1.0

        img = data.astype(float)
        vmin, vmax = np.nanmin(img), np.nanmax(img)
        if vmax > vmin:
            img_u8 = ((img - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            img_u8 = np.zeros_like(img, dtype=np.uint8)

        counts, edges = np.histogram(data.flatten(), bins=256)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        histogram_json = json.dumps(
            {
                "bins": bin_centers.tolist(),
                "counts": counts.tolist(),
            }
        )

        aspect = w / h if h > 0 else 1.0
        if aspect >= 1.0:
            vw, vh = 256, max(64, int(256 / aspect))
        else:
            vh, vw = 256, max(64, int(256 * aspect))

        with self.hold_trait_notifications():
            self.image_bytes = img_u8.tobytes()
            self.image_width = w
            self.image_height = h
            self.viewer_width = vw
            self.viewer_height = vh
            self.units = units
            self.use_scalebar = equal
            self.scale_x = float(abs(dx))
            self.scale_y = float(abs(dx))
            self.x_axis_json = json.dumps(x_axis.tolist())
            self.y_axis_json = json.dumps(y_axis.tolist())
            self.histogram_data = histogram_json
            self.hist_min = float(vmin)
            self.hist_max = float(vmax)
            self.display_min = float(vmin)
            self.display_max = float(vmax)
            self.scale_mode = "linear"
            self.colormap_name = "gray"
            self.colormap_data = json.dumps(self._build_colormap_lut("gray"))

    # ------------------------------------------------------------------
    def _to_png_bytes(self) -> bytes:
        """Render the current image as a PNG byte string.

        Applies the current colormap so the PyCharm static preview matches
        the live widget.  Falls back gracefully if PIL is not installed.
        """
        from PIL import Image as _PILImage

        arr = np.frombuffer(self.image_bytes, dtype=np.uint8).reshape(
            self.image_height, self.image_width
        )
        # Apply colormap if set and not plain gray.
        try:
            lut = json.loads(self.colormap_data)
            if isinstance(lut, list) and len(lut) == 256:
                lut_arr = np.array(lut, dtype=np.uint8)   # (256, 3)
                rgb = lut_arr[arr]                          # (H, W, 3)
                img = _PILImage.fromarray(rgb, mode="RGB")
            else:
                raise ValueError
        except Exception:
            img = _PILImage.fromarray(arr, mode="L")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _repr_mimebundle_(self, **kwargs: object) -> dict:
        """Return the anywidget MIME bundle plus a ``image/png`` fallback."""
        bundle: dict = super()._repr_mimebundle_(**kwargs) or {}
        try:
            bundle["image/png"] = base64.b64encode(self._to_png_bytes()).decode("ascii")
        except Exception:
            pass
        return bundle

    # ------------------------------------------------------------------
    def update(
        self,
        data: np.ndarray,
        x_axis: np.ndarray | None = None,
        y_axis: np.ndarray | None = None,
        units: str | None = None,
    ) -> None:
        """Replace the displayed image with new data.

        Existing zoom / pan state is preserved.  Marker overlays and overlay
        widgets are *not* cleared — call :meth:`clear_markers` or
        :meth:`clear_widgets` explicitly if needed.

        Parameters
        ----------
        data :
            New 2-D (or 3-D) image array.
        x_axis :
            Column coordinates (length ``W``).  Re-uses the previous axis if
            ``None`` and the shape is unchanged; falls back to
            ``np.arange(W)`` otherwise.
        y_axis :
            Row coordinates (length ``H``).  Same fallback logic as
            *x_axis*.
        units :
            Unit label.  Unchanged if ``None``.
        """
        data = np.asarray(data)
        if data.ndim == 3:
            data = data[:, :, 0]
        if data.ndim != 2:
            raise ValueError(f"data must be 2-D (H×W), got shape {data.shape}")

        h, w = data.shape

        if x_axis is None:
            x_axis = np.array(json.loads(self.x_axis_json))
            if len(x_axis) != w:
                x_axis = np.arange(w, dtype=float)
        if y_axis is None:
            y_axis = np.array(json.loads(self.y_axis_json))
            if len(y_axis) != h:
                y_axis = np.arange(h, dtype=float)

        x_axis = np.asarray(x_axis, dtype=float)
        y_axis = np.asarray(y_axis, dtype=float)

        equal = _equal_scale(x_axis, y_axis)
        dx = float(np.median(np.diff(x_axis))) if len(x_axis) > 1 else 1.0
        dy = float(np.median(np.diff(y_axis))) if len(y_axis) > 1 else 1.0

        img = data.astype(float)
        vmin, vmax = np.nanmin(img), np.nanmax(img)
        if vmax > vmin:
            img_u8 = ((img - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            img_u8 = np.zeros_like(img, dtype=np.uint8)

        counts, edges = np.histogram(data.flatten(), bins=256)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        histogram_json = json.dumps(
            {"bins": bin_centers.tolist(), "counts": counts.tolist()}
        )

        with self.hold_trait_notifications():
            self.image_bytes    = img_u8.tobytes()
            self.image_width    = w
            self.image_height   = h
            self.use_scalebar   = equal
            self.scale_x        = float(abs(dx))
            self.scale_y        = float(abs(dy))
            self.x_axis_json    = json.dumps(x_axis.tolist())
            self.y_axis_json    = json.dumps(y_axis.tolist())
            self.histogram_data = histogram_json
            self.hist_min       = float(vmin)
            self.hist_max       = float(vmax)
            self.display_min    = float(vmin)
            self.display_max    = float(vmax)
            # Re-apply the current colormap to match the new data range.
            self.colormap_data  = json.dumps(
                self._build_colormap_lut(self.colormap_name)
            )
            if units is not None:
                self.units = units

    # ------------------------------------------------------------------
    @staticmethod
    def _build_colormap_lut(name: str) -> list[list[int]]:
        """Resolve *name* to a 256×3 uint8 RGB lookup table.

        Resolution order
        ----------------
        1. ``colorcet`` — if installed, ``cc.cm.<name>`` is tried first so
           that colorcet short aliases (``'fire'``, ``'bmy'``, ``'CET-L1'``,
           etc.) work directly.
        2. ``matplotlib.cm.get_cmap(name)`` — covers all built-in matplotlib
           colormaps as well as any registered third-party colormaps.

        Parameters
        ----------
        name :
            Colormap name string.  Case-sensitive for colorcet names;
            case-insensitive for matplotlib names.

        Returns
        -------
        list of [r, g, b]
            256 entries, each a list of three ``int`` values in ``[0, 255]``.

        Raises
        ------
        ValueError
            If the name cannot be resolved by either library.
        """
        import matplotlib.cm as _mcm

        cmap = None

        # 1. Try colorcet first (optional dependency).
        try:
            import colorcet as _cc  # noqa: F401
            if hasattr(_cc.cm, name):
                cmap = getattr(_cc.cm, name)
        except ImportError:
            pass  # colorcet not installed — fall through to matplotlib

        # 2. Fall back to matplotlib (covers built-ins + any registered maps).
        if cmap is None:
            try:
                cmap = _mcm.get_cmap(name)
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Colormap {name!r} not found in matplotlib or colorcet.\n"
                    f"Install colorcet for additional perceptually-uniform maps: "
                    f"pip install colorcet"
                ) from exc

        xs = np.linspace(0.0, 1.0, 256)
        rgba = cmap(xs)                                        # (256, 4) float [0,1]
        rgb8 = (rgba[:, :3] * 255).round().astype(np.uint8)   # (256, 3) uint8
        return rgb8.tolist()

    def set_colormap(self, name: str) -> None:
        """Set the display colormap.

        Supports every colormap available in **matplotlib** and, if installed,
        **colorcet** (perceptually-uniform maps).

        Parameters
        ----------
        name :
            Colormap name string.  Examples:

            *Matplotlib built-ins*
            ``'gray'``, ``'viridis'``, ``'plasma'``, ``'inferno'``,
            ``'magma'``, ``'cividis'``, ``'hot'``, ``'jet'``, ``'RdBu'``,
            ``'coolwarm'``, ``'turbo'``, …

            *Colorcet (install with* ``pip install colorcet`` *)*
            ``'fire'``, ``'bmy'``, ``'CET-L1'``, ``'CET-D1'``,
            ``'CET-R1'``, ``'isolum'``, ``'rainbow'``, …

        Raises
        ------
        ValueError
            If *name* cannot be resolved by either library.

        Examples
        --------
        >>> v.set_colormap('viridis')
        >>> v.set_colormap('fire')      # colorcet
        >>> v.set_colormap('RdBu')      # diverging
        """
        lut = self._build_colormap_lut(name)
        with self.hold_trait_notifications():
            self.colormap_name = name
            self.colormap_data = json.dumps(lut)

    @staticmethod
    def list_colormaps(filter: str | None = None) -> list[str]:
        """Return a sorted list of available colormap names.

        Includes all **matplotlib** colormaps and, if **colorcet** is
        installed, all colorcet colormaps registered in matplotlib.

        Parameters
        ----------
        filter :
            Optional substring filter (case-insensitive).  Only names
            containing this string are returned.  For example
            ``filter='seq'`` returns sequential colormaps whose name
            contains ``'seq'``.

        Returns
        -------
        list of str
            Sorted list of colormap name strings.

        Examples
        --------
        >>> Viewer2D.list_colormaps()                    # all colormaps
        >>> Viewer2D.list_colormaps('viridis')           # exact match search
        >>> Viewer2D.list_colormaps('cet')               # all colorcet maps
        """
        import matplotlib.cm as _mcm
        names: set[str] = set(_mcm._colormaps)

        try:
            import colorcet as _cc
            for attr in dir(_cc.cm):
                if not attr.startswith('_'):
                    names.add(attr)
        except ImportError:
            pass

        result = sorted(names)
        if filter is not None:
            fl = filter.lower()
            result = [n for n in result if fl in n.lower()]
        return result

    def set_clim(
        self,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """Set the display contrast limits (colour window).

        Values outside ``[vmin, vmax]`` are clipped to black or white.
        Both limits must lie within the data range ``[hist_min, hist_max]``.

        Parameters
        ----------
        vmin :
            Lower display bound.  Defaults to the current ``display_min``.
        vmax :
            Upper display bound.  Defaults to the current ``display_max``.

        Raises
        ------
        ValueError
            If ``vmin >= vmax``.

        Examples
        --------
        >>> v.set_clim(0.1, 0.9)   # show only the central 80 % of the range
        >>> v.set_clim(vmax=0.5)   # lower the ceiling without changing the floor
        """
        new_min = float(vmin) if vmin is not None else self.display_min
        new_max = float(vmax) if vmax is not None else self.display_max
        if new_min >= new_max:
            raise ValueError(f"vmin ({new_min}) must be less than vmax ({new_max})")
        with self.hold_trait_notifications():
            self.display_min = new_min
            self.display_max = new_max

    def set_scale_mode(self, mode: str) -> None:
        """Set the intensity mapping scale mode.

        Parameters
        ----------
        mode :
            One of:

            * ``'linear'`` – uniform linear mapping (default).
            * ``'log'``    – base-10 logarithmic mapping.  Values ≤ 0 are
              clamped to a small positive epsilon.
            * ``'symlog'`` – symmetric-log: linear near zero (within 1 % of
              the display range), logarithmic outside.  Handles negative
              values and zero correctly.

        Raises
        ------
        ValueError
            If *mode* is not one of the supported values.

        Notes
        -----
        The keyboard shortcuts ``l`` (log) and ``s`` (symlog) toggle the same
        mode interactively from within the viewer canvas.  Press the same key
        again to return to ``'linear'``.
        """
        valid = ("linear", "log", "symlog")
        if mode not in valid:
            raise ValueError(f"mode must be one of {valid}, got {mode!r}")
        self.scale_mode = mode

    def add_widget(self, kind: str, color: str = "#00e5ff", **kwargs: object) -> str:
        """Add a moveable overlay widget to the viewer.

        Parameters
        ----------
        kind :
            One of ``'circle'``, ``'rectangle'``, ``'annular'``,
            ``'polygon'``, ``'label'``, or ``'crosshair'``.
        color :
            CSS color string (default cyan ``'#00e5ff'``).
        **kwargs
            Shape-specific keyword arguments:

            * **circle** – ``cx``, ``cy`` (center, image px), ``r`` (radius).
            * **rectangle** – ``x``, ``y`` (top-left, image px), ``w``, ``h``.
            * **annular** – ``cx``, ``cy``, ``r_outer``, ``r_inner``.
            * **polygon** – ``vertices`` — list of ``[x, y]`` image-px points
              (minimum 3).
            * **label** – ``x``, ``y``, ``text``, ``fontsize``.
            * **crosshair** – ``cx``, ``cy`` (center, image px).
              Move-only; draws a full-canvas ``+`` through the centre point.

        Returns
        -------
        str
            Unique widget ID.  Pass to :meth:`get_widget`,
            :meth:`remove_widget`, or :meth:`set_polygon_vertices`.

        Raises
        ------
        ValueError
            If *kind* is unrecognized, or if a polygon has fewer than 3
            vertices, or if ``r_inner >= r_outer`` for an annular widget.
        """
        kind = kind.lower()
        valid = ("circle", "rectangle", "annular", "polygon", "label", "crosshair")
        if kind not in valid:
            raise ValueError(f"kind must be one of {valid}, got {kind!r}")
        iw, ih = self.image_width, self.image_height
        wid = str(_uuid.uuid4())[:8]

        def _f(key: str, default: float) -> float:
            return float(kwargs.get(key, default))  # type: ignore[arg-type]

        def _i(key: str, default: int) -> int:
            return int(kwargs.get(key, default))  # type: ignore[arg-type]

        if kind == "circle":
            entry: dict = {
                "id": wid, "type": "circle",
                "cx": _f("cx", iw / 2),
                "cy": _f("cy", ih / 2),
                "r":  _f("r",  iw * 0.1),
                "color": color,
            }
        elif kind == "rectangle":
            entry = {
                "id": wid, "type": "rectangle",
                "x": _f("x", iw * 0.25),
                "y": _f("y", ih * 0.25),
                "w": _f("w", iw * 0.5),
                "h": _f("h", ih * 0.5),
                "color": color,
            }
        elif kind == "annular":
            r_outer = _f("r_outer", iw * 0.2)
            r_inner = _f("r_inner", iw * 0.1)
            if r_inner >= r_outer:
                raise ValueError("r_inner must be less than r_outer")
            entry = {
                "id": wid, "type": "annular",
                "cx":      _f("cx", iw / 2),
                "cy":      _f("cy", ih / 2),
                "r_outer": r_outer,
                "r_inner": r_inner,
                "color":   color,
            }
        elif kind == "polygon":
            raw_verts = kwargs.get("vertices", [
                [iw * 0.25, ih * 0.25],
                [iw * 0.75, ih * 0.25],
                [iw * 0.75, ih * 0.75],
                [iw * 0.25, ih * 0.75],
            ])
            verts = [[float(x), float(y)] for x, y in raw_verts]  # type: ignore[union-attr]
            if len(verts) < 3:
                raise ValueError("polygon needs at least 3 vertices")
            entry = {
                "id": wid, "type": "polygon",
                "vertices": verts,
                "color": color,
            }
        elif kind == "crosshair":
            entry = {
                "id":    wid,
                "type":  "crosshair",
                "cx":    _f("cx", iw / 2),
                "cy":    _f("cy", ih / 2),
                "color": color,
            }
        else:  # label
            entry = {
                "id":       wid,
                "type":     "label",
                "x":        _f("x", iw * 0.1),
                "y":        _f("y", ih * 0.1),
                "text":     str(kwargs.get("text", "Label")),
                "fontsize": _i("fontsize", 14),
                "color":    color,
            }

        widgets = json.loads(self.overlay_widgets)
        widgets.append(entry)
        self.overlay_widgets = json.dumps(widgets)
        return wid

    # ------------------------------------------------------------------
    def add_annular_widget(
        self,
        cx: float | None = None,
        cy: float | None = None,
        r_outer: float | None = None,
        r_inner: float | None = None,
        color: str = "#00e5ff",
    ) -> str:
        """Add an annular (ring) overlay widget.

        Mirrors :class:`~hyperspy.drawing._widgets.circle.CircleWidget` with
        both an outer and an inner radius.

        Parameters
        ----------
        cx, cy :
            Centre position in image-pixel units.
            Defaults to the image centre.
        r_outer :
            Outer radius in image-pixel units (default 20 % of image width).
        r_inner :
            Inner radius in image-pixel units (default 10 % of image width).
            Must be strictly less than *r_outer*.
        color :
            CSS colour string.

        Returns
        -------
        str
            Widget ID.
        """
        return self.add_widget(
            "annular",
            color=color,
            cx=cx if cx is not None else self.image_width / 2,
            cy=cy if cy is not None else self.image_height / 2,
            r_outer=r_outer if r_outer is not None else self.image_width * 0.2,
            r_inner=r_inner if r_inner is not None else self.image_width * 0.1,
        )

    def add_polygon_widget(
        self,
        vertices: list[list[float]] | None = None,
        color: str = "#00e5ff",
    ) -> str:
        """Add a polygon overlay widget.

        Mirrors :class:`~hyperspy.drawing._widgets.polygon.PolygonWidget`.
        Every vertex is represented by a drag handle; dragging inside the
        polygon moves the whole shape.

        Parameters
        ----------
        vertices :
            ``[[x, y], ...]`` vertex positions in image-pixel units.
            Must contain at least 3 points.
            Defaults to a centred square covering 50 % of the image.
        color :
            CSS colour string.

        Returns
        -------
        str
            Widget ID.
        """
        kwargs: dict = {} if vertices is None else {"vertices": vertices}
        return self.add_widget("polygon", color=color, **kwargs)

    def set_polygon_vertices(self, wid: str, vertices: list[list[float]]) -> None:
        """Replace the vertices of an existing polygon widget in-place.

        Parameters
        ----------
        wid :
            Widget ID returned by :meth:`add_polygon_widget`.
        vertices :
            New ``[[x, y], ...]`` vertex list in image-pixel units.
            Must contain at least 3 points.

        Raises
        ------
        ValueError
            If fewer than 3 vertices are supplied.
        KeyError
            If *wid* does not correspond to a known widget.
        """
        verts = [[float(x), float(y)] for x, y in vertices]
        if len(verts) < 3:
            raise ValueError("polygon needs at least 3 vertices")
        widgets = json.loads(self.overlay_widgets)
        for w in widgets:
            if w["id"] == wid:
                w["vertices"] = verts
                self.overlay_widgets = json.dumps(widgets)
                return
        raise KeyError(f"No widget with id {wid!r}")

    def add_label_widget(
        self,
        text: str = "Label",
        x: float | None = None,
        y: float | None = None,
        fontsize: int = 14,
        color: str = "#00e5ff",
    ) -> str:
        """Add a draggable text label overlay widget.

        Mirrors :class:`~hyperspy.drawing._widgets.label.LabelWidget`.

        Parameters
        ----------
        text :
            Displayed string (default ``'Label'``).
        x, y :
            Anchor position in image-pixel units.
            Defaults to the top-left area of the image.
        fontsize :
            Font size in CSS pixels (default ``14``).
        color :
            CSS colour string.

        Returns
        -------
        str
            Widget ID.
        """
        return self.add_widget(
            "label",
            color=color,
            x=x if x is not None else self.image_width * 0.1,
            y=y if y is not None else self.image_height * 0.1,
            text=text,
            fontsize=fontsize,
        )

    def add_crosshair_widget(
        self,
        cx: float | None = None,
        cy: float | None = None,
        color: str = "#00e5ff",
    ) -> str:
        """Add a crosshair (``+``) overlay widget.

        The crosshair draws two full-canvas lines — one horizontal and one
        vertical — intersecting at ``(cx, cy)``.  It is **move-only**: there
        are no resize handles and no resize interaction.

        Dragging anywhere on either arm (within 4 canvas px) or near the
        centre point (within 10 canvas px) moves the crosshair.

        Parameters
        ----------
        cx :
            Horizontal centre position in image-pixel units.
            Defaults to the image centre.
        cy :
            Vertical centre position in image-pixel units.
            Defaults to the image centre.
        color :
            CSS colour string (default cyan ``'#00e5ff'``).

        Returns
        -------
        str
            Widget ID.

        Examples
        --------
        >>> wid = v.add_crosshair_widget(cx=128, cy=128)
        >>> v.get_widget(wid)
        {'id': ..., 'type': 'crosshair', 'cx': 128.0, 'cy': 128.0, 'color': '#00e5ff'}
        """
        return self.add_widget(
            "crosshair",
            color=color,
            cx=cx if cx is not None else self.image_width / 2,
            cy=cy if cy is not None else self.image_height / 2,
        )

    def remove_widget(self, wid: str) -> None:
        """Remove an overlay widget by ID.

        Parameters
        ----------
        wid :
            Widget ID returned by any ``add_*_widget`` call.

        Raises
        ------
        KeyError
            If *wid* does not correspond to a known widget.
        """
        widgets = json.loads(self.overlay_widgets)
        new_widgets = [w for w in widgets if w["id"] != wid]
        if len(new_widgets) == len(widgets):
            raise KeyError(f"No overlay widget with id {wid!r}")
        self.overlay_widgets = json.dumps(new_widgets)

    def clear_widgets(self) -> None:
        """Remove all overlay widgets."""
        self.overlay_widgets = "[]"

    def get_widget(self, wid: str) -> dict:
        """Return the current state of an overlay widget as a plain dict.

        All position and size values are in image-pixel units.

        Parameters
        ----------
        wid :
            Widget ID returned by any ``add_*_widget`` call.

        Returns
        -------
        dict
            A copy of the widget's internal state dictionary.

        Raises
        ------
        KeyError
            If *wid* does not correspond to a known widget.
        """
        for w in json.loads(self.overlay_widgets):
            if w["id"] == wid:
                return dict(w)
        raise KeyError(f"No overlay widget with id {wid!r}")

    # ================================================================== markers
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _new_marker_id(self) -> str:
        """Return a new short unique marker-set ID."""
        return str(_uuid.uuid4())[:8]

    def _write_marker(self, ms: dict, marker_id: str | None) -> str:
        """Insert or replace a marker set by ID.

        Parameters
        ----------
        ms :
            Marker-set dict (without ``"id"``; this method fills it in).
            May contain optional ``"label"`` and/or ``"labels"`` fields.
        marker_id :
            If ``None``, generate a new ID and append the set.
            If given, replace the matching set in-place; append if not found.

        Returns
        -------
        str
            The final marker ID (new or supplied).
        """
        sets = json.loads(self.markers_json)
        if marker_id is None:
            marker_id = self._new_marker_id()
            ms["id"] = marker_id
            sets.append(ms)
        else:
            ms["id"] = marker_id
            for i, s in enumerate(sets):
                if s.get("id") == marker_id:
                    sets[i] = ms
                    break
            else:
                sets.append(ms)
        self.markers_json = json.dumps(sets)
        return marker_id

    @staticmethod
    def _broadcast_1d(arr: object, n: int, name: str) -> list[float]:
        """Broadcast *arr* to a flat Python list of length *n*.

        A scalar value is repeated *n* times.  A 1-D array must already have
        exactly *n* elements.

        Parameters
        ----------
        arr :
            Scalar or 1-D array-like of numeric values.
        n :
            Required length.
        name :
            Parameter name used in error messages.

        Returns
        -------
        list of float

        Raises
        ------
        ValueError
            If *arr* is 1-D but has the wrong length.
        """
        a = np.asarray(arr, dtype=float)
        if a.ndim == 0:
            return np.full(n, float(a)).tolist()
        if a.ndim != 1 or len(a) != n:
            raise ValueError(f"'{name}' must be a scalar or 1-D array of length {n}")
        return a.tolist()

    @staticmethod
    def _check_offsets(offsets: object) -> np.ndarray:
        """Validate and normalise *offsets* to shape ``(N, 2)``.

        A single ``[x, y]`` pair (shape ``(2,)``) is promoted to ``(1, 2)``.

        Parameters
        ----------
        offsets :
            Array-like of shape ``(N, 2)`` or ``(2,)``.

        Returns
        -------
        np.ndarray of shape (N, 2)

        Raises
        ------
        ValueError
            If the shape is not compatible with ``(N, 2)``.
        """
        arr = np.asarray(offsets, dtype=float)
        if arr.ndim == 1 and arr.shape[0] == 2:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("offsets must be shape (N, 2)")
        return arr

    @staticmethod
    def _opt_labels(
        label: str | None,
        labels: list[str] | None,
        n: int,
    ) -> dict:
        """Build the optional ``label`` / ``labels`` sub-dict for a marker set.

        Parameters
        ----------
        label :
            Collection-level hover tooltip, or ``None``.
        labels :
            Per-marker hover tooltips (length must equal *n*), or ``None``.
        n :
            Number of markers in the set.

        Returns
        -------
        dict
            Empty dict, ``{"label": …}``, ``{"labels": […]}``, or both.

        Raises
        ------
        ValueError
            If *labels* is supplied but has a length other than *n*.
        """
        extra: dict = {}
        if label is not None:
            extra["label"] = str(label)
        if labels is not None:
            ls = list(labels)
            if len(ls) != n:
                raise ValueError(
                    f"len(labels) must equal the number of markers ({n}), "
                    f"got {len(ls)}"
                )
            extra["labels"] = [str(l) for l in ls]
        return extra

    # ------------------------------------------------------------------
    # Circles
    # ------------------------------------------------------------------
    def add_circles(self, offsets, sizes, color="#ff0000", linewidth=1.5,
                    fill_color=None, fill_alpha=0.3,
                    marker_id=None, label=None, labels=None) -> str:
        """Add or replace a set of circle markers.

        Mirrors :class:`~hyperspy.drawing._markers.circles.Circles`.

        Parameters
        ----------
        offsets : array-like (N, 2)
            ``[[x, y], ...]`` centre positions in image-pixel space.
        sizes : array-like or scalar
            Radius of each circle in image-pixel units.
        color : str, optional
            Stroke colour (default ``'#ff0000'``).
        linewidth : float, optional
            Stroke width in canvas pixels.
        fill_color : str or None, optional
            Fill colour.  ``None`` (default) draws outline only.
        fill_alpha : float, optional
            Fill opacity in ``[0, 1]`` (default ``0.3``).
        marker_id : str, optional
            Replace an existing set if supplied; otherwise append a new one.
        label : str, optional
            Tooltip shown when hovering anywhere over this collection.
        labels : list of str, optional
            Per-marker tooltip, one entry per offset.

        Returns
        -------
        str  Marker ID.
        """
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms: dict = {"type": "circles", "offsets": offsets.tolist(),
                    "sizes": self._broadcast_1d(sizes, n, "sizes"),
                    "color": color, "linewidth": linewidth,
                    **self._opt_labels(label, labels, n)}
        if fill_color is not None:
            ms["fill_color"] = fill_color
            ms["fill_alpha"] = float(fill_alpha)
        return self._write_marker(ms, marker_id)

    def set_circles(self, offsets, sizes, color="#ff0000", linewidth=1.5,
                    fill_color=None, fill_alpha=0.3,
                    marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_circles`."""
        return self.add_circles(offsets, sizes, color=color, linewidth=linewidth,
                                fill_color=fill_color, fill_alpha=fill_alpha,
                                marker_id=marker_id, label=label, labels=labels)

    # ------------------------------------------------------------------
    # Arrows
    # ------------------------------------------------------------------
    def add_arrows(self, offsets, U, V, color="#ff0000", linewidth=1.5,
                   marker_id=None, label=None, labels=None) -> str:
        """Add or replace a set of arrow markers.

        Mirrors :class:`~hyperspy.drawing._markers.arrows.Arrows`.

        Parameters
        ----------
        offsets : array-like (N, 2)  Arrow tail positions.
        U, V : array-like or scalar  Horizontal / vertical components (image px).
        color : str, optional
        linewidth : float, optional
        marker_id : str, optional  Replace existing set if given.
        label : str, optional      Collection hover tooltip.
        labels : list of str, optional  Per-arrow hover tooltips.

        Returns
        -------
        str  Marker ID.
        """
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms = {"type": "arrows", "offsets": offsets.tolist(),
              "U": self._broadcast_1d(U, n, "U"),
              "V": self._broadcast_1d(V, n, "V"),
              "color": color, "linewidth": linewidth,
              **self._opt_labels(label, labels, n)}
        return self._write_marker(ms, marker_id)

    def set_arrows(self, offsets, U, V, color="#ff0000", linewidth=1.5,
                   marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_arrows`."""
        return self.add_arrows(offsets, U, V, color=color, linewidth=linewidth,
                               marker_id=marker_id, label=label, labels=labels)

    # ------------------------------------------------------------------
    # Ellipses
    # ------------------------------------------------------------------
    def add_ellipses(self, offsets, widths, heights, angles=0,
                     color="#ff0000", linewidth=1.5,
                     fill_color=None, fill_alpha=0.3,
                     marker_id=None, label=None, labels=None) -> str:
        """Add or replace a set of ellipse markers.

        Mirrors :class:`~hyperspy.drawing._markers.ellipses.Ellipses`.

        Parameters
        ----------
        offsets : array-like (N, 2)
        widths, heights : array-like or scalar  Full extents in image px.
        angles : array-like or scalar           Rotation in degrees.
        color : str, optional
        linewidth : float, optional
        fill_color : str or None, optional
            Fill colour.  ``None`` draws outline only.
        fill_alpha : float, optional
            Fill opacity in ``[0, 1]`` (default ``0.3``).
        marker_id : str, optional
        label : str, optional
        labels : list of str, optional

        Returns
        -------
        str  Marker ID.
        """
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms: dict = {"type": "ellipses", "offsets": offsets.tolist(),
                    "widths":  self._broadcast_1d(widths,  n, "widths"),
                    "heights": self._broadcast_1d(heights, n, "heights"),
                    "angles":  self._broadcast_1d(angles,  n, "angles"),
                    "color": color, "linewidth": linewidth,
                    **self._opt_labels(label, labels, n)}
        if fill_color is not None:
            ms["fill_color"] = fill_color
            ms["fill_alpha"] = float(fill_alpha)
        return self._write_marker(ms, marker_id)

    def set_ellipses(self, offsets, widths, heights, angles=0,
                     color="#ff0000", linewidth=1.5,
                     fill_color=None, fill_alpha=0.3,
                     marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_ellipses`."""
        return self.add_ellipses(offsets, widths, heights, angles=angles,
                                 color=color, linewidth=linewidth,
                                 fill_color=fill_color, fill_alpha=fill_alpha,
                                 marker_id=marker_id, label=label, labels=labels)

    # ------------------------------------------------------------------
    # Lines
    # ------------------------------------------------------------------
    def add_lines(self, segments, color="#ff0000", linewidth=1.5,
                  marker_id=None, label=None, labels=None) -> str:
        """Add or replace a set of line-segment markers.

        Mirrors :class:`~hyperspy.drawing._markers.lines.Lines`.

        Parameters
        ----------
        segments : array-like (N, 2, 2)  ``[[[x1,y1],[x2,y2]], ...]``.
        color : str, optional
        linewidth : float, optional
        marker_id : str, optional
        label : str, optional
        labels : list of str, optional  One per segment.

        Returns
        -------
        str  Marker ID.
        """
        segments = np.asarray(segments, dtype=float)
        if segments.ndim == 2 and segments.shape == (2, 2):
            segments = segments[np.newaxis]
        if segments.ndim != 3 or segments.shape[1:] != (2, 2):
            raise ValueError("segments must be shape (N, 2, 2)")
        n = len(segments)
        ms = {"type": "lines", "segments": segments.tolist(),
              "color": color, "linewidth": linewidth,
              **self._opt_labels(label, labels, n)}
        return self._write_marker(ms, marker_id)

    def set_lines(self, segments, color="#ff0000", linewidth=1.5,
                  marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_lines`."""
        return self.add_lines(segments, color=color, linewidth=linewidth,
                              marker_id=marker_id, label=label, labels=labels)

    # ------------------------------------------------------------------
    # Rectangles
    # ------------------------------------------------------------------
    def add_rectangles(self, offsets, widths, heights, angles=0,
                       color="#ff0000", linewidth=1.5,
                       fill_color=None, fill_alpha=0.3,
                       marker_id=None, label=None, labels=None) -> str:
        """Add or replace a set of rectangle markers.

        Parameters
        ----------
        offsets : array-like (N, 2)
        widths, heights : array-like or scalar
        angles : array-like or scalar  Rotation in degrees.
        color : str, optional
        linewidth : float, optional
        fill_color : str or None, optional
            Fill colour.  ``None`` draws outline only.
        fill_alpha : float, optional
            Fill opacity in ``[0, 1]`` (default ``0.3``).
        marker_id : str, optional
        label : str, optional
        labels : list of str, optional

        Returns
        -------
        str  Marker ID.
        """
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms: dict = {"type": "rectangles", "offsets": offsets.tolist(),
                    "widths":  self._broadcast_1d(widths,  n, "widths"),
                    "heights": self._broadcast_1d(heights, n, "heights"),
                    "angles":  self._broadcast_1d(angles,  n, "angles"),
                    "color": color, "linewidth": linewidth,
                    **self._opt_labels(label, labels, n)}
        if fill_color is not None:
            ms["fill_color"] = fill_color
            ms["fill_alpha"] = float(fill_alpha)
        return self._write_marker(ms, marker_id)

    def set_rectangles(self, offsets, widths, heights, angles=0,
                       color="#ff0000", linewidth=1.5,
                       fill_color=None, fill_alpha=0.3,
                       marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_rectangles`."""
        return self.add_rectangles(offsets, widths, heights, angles=angles,
                                   color=color, linewidth=linewidth,
                                   fill_color=fill_color, fill_alpha=fill_alpha,
                                   marker_id=marker_id, label=label, labels=labels)

    # ------------------------------------------------------------------
    # Squares
    # ------------------------------------------------------------------
    def add_squares(self, offsets, widths, angles=0,
                    color="#ff0000", linewidth=1.5,
                    fill_color=None, fill_alpha=0.3,
                    marker_id=None, label=None, labels=None) -> str:
        """Add or replace a set of square markers.

        Mirrors :class:`~hyperspy.drawing._markers.squares.Squares`.

        Parameters
        ----------
        offsets : array-like (N, 2)
        widths : array-like or scalar  Side length in image px.
        angles : array-like or scalar  Rotation in degrees.
        color : str, optional
        linewidth : float, optional
        fill_color : str or None, optional
            Fill colour.  ``None`` draws outline only.
        fill_alpha : float, optional
            Fill opacity in ``[0, 1]`` (default ``0.3``).
        marker_id : str, optional
        label : str, optional
        labels : list of str, optional

        Returns
        -------
        str  Marker ID.
        """
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms: dict = {"type": "squares", "offsets": offsets.tolist(),
                    "widths": self._broadcast_1d(widths, n, "widths"),
                    "angles": self._broadcast_1d(angles, n, "angles"),
                    "color": color, "linewidth": linewidth,
                    **self._opt_labels(label, labels, n)}
        if fill_color is not None:
            ms["fill_color"] = fill_color
            ms["fill_alpha"] = float(fill_alpha)
        return self._write_marker(ms, marker_id)

    def set_squares(self, offsets, widths, angles=0,
                    color="#ff0000", linewidth=1.5,
                    fill_color=None, fill_alpha=0.3,
                    marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_squares`."""
        return self.add_squares(offsets, widths, angles=angles,
                                color=color, linewidth=linewidth,
                                fill_color=fill_color, fill_alpha=fill_alpha,
                                marker_id=marker_id, label=label, labels=labels)

    # ------------------------------------------------------------------
    # Polygons
    # ------------------------------------------------------------------
    def add_polygons(self, vertices_list, color="#ff0000", linewidth=1.5,
                     fill_color=None, fill_alpha=0.3,
                     marker_id=None, label=None, labels=None) -> str:
        """Add or replace a set of polygon markers.

        Unlike the interactive overlay :meth:`add_polygon_widget`, these are
        read-only markers drawn on the markers canvas.

        Parameters
        ----------
        vertices_list : list of array-like
            Each element is one polygon: ``[[x, y], ...]`` in image-pixel
            units.  Minimum 3 vertices per polygon.
        color : str, optional
            Stroke colour (default ``'#ff0000'``).
        linewidth : float, optional
            Stroke width in canvas pixels.
        fill_color : str or None, optional
            Fill colour.  ``None`` draws outline only.
        fill_alpha : float, optional
            Fill opacity in ``[0, 1]`` (default ``0.3``).
        marker_id : str, optional
            Replace an existing set if supplied; otherwise append a new one.
        label : str, optional
            Tooltip shown when hovering over the collection.
        labels : list of str, optional
            Per-polygon hover tooltip.

        Returns
        -------
        str  Marker ID.

        Examples
        --------
        >>> v.add_polygons([[[10,10],[50,10],[30,40]], [[60,60],[100,60],[80,100]]],
        ...                color='#00ff00', fill_color='#00ff00', fill_alpha=0.2)
        """
        # Normalise each polygon to a plain list of [float, float] pairs.
        vlist = []
        for poly in vertices_list:
            arr = np.asarray(poly, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("each polygon must be shape (N, 2)")
            if len(arr) < 3:
                raise ValueError("each polygon needs at least 3 vertices")
            vlist.append(arr.tolist())
        n = len(vlist)
        ms: dict = {"type": "polygons", "vertices_list": vlist,
                    "color": color, "linewidth": linewidth,
                    **self._opt_labels(label, labels, n)}
        if fill_color is not None:
            ms["fill_color"] = fill_color
            ms["fill_alpha"] = float(fill_alpha)
        return self._write_marker(ms, marker_id)

    def set_polygons(self, vertices_list, color="#ff0000", linewidth=1.5,
                     fill_color=None, fill_alpha=0.3,
                     marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_polygons`."""
        return self.add_polygons(vertices_list, color=color, linewidth=linewidth,
                                 fill_color=fill_color, fill_alpha=fill_alpha,
                                 marker_id=marker_id, label=label, labels=labels)

    # ------------------------------------------------------------------
    # Texts
    # ------------------------------------------------------------------
    def add_texts(self, offsets, texts, color="#ff0000", fontsize=12,
                  marker_id=None, label=None, labels=None) -> str:
        """Add or replace a set of text markers.

        Mirrors :class:`~hyperspy.drawing._markers.texts.Texts`.

        Parameters
        ----------
        offsets : array-like (N, 2)
        texts : list of str           Displayed text per position.
        color : str, optional
        fontsize : int, optional
        marker_id : str, optional
        label : str, optional         Collection hover tooltip (distinct from displayed texts).
        labels : list of str, optional  Per-marker hover tooltip; defaults to ``texts`` if omitted.

        Returns
        -------
        str  Marker ID.
        """
        offsets = self._check_offsets(offsets)
        texts = list(texts)
        n = len(offsets)
        if len(texts) != n:
            raise ValueError("len(texts) must equal len(offsets)")
        ms = {"type": "texts", "offsets": offsets.tolist(),
              "texts": texts, "color": color, "fontsize": fontsize,
              **self._opt_labels(label, labels, n)}
        return self._write_marker(ms, marker_id)

    def set_texts(self, offsets, texts, color="#ff0000", fontsize=12,
                  marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_texts`."""
        return self.add_texts(offsets, texts, color=color, fontsize=fontsize,
                              marker_id=marker_id, label=label, labels=labels)

    # ------------------------------------------------------------------
    # Legacy aliases
    # ------------------------------------------------------------------
    def set_markers(self, offsets, sizes, color="#ff0000", linewidth=1.5,
                    fill_color=None, fill_alpha=0.3,
                    marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_circles` (backwards compatibility)."""
        return self.add_circles(offsets, sizes, color=color, linewidth=linewidth,
                                fill_color=fill_color, fill_alpha=fill_alpha,
                                marker_id=marker_id, label=label, labels=labels)

    def add_markers(self, offsets, sizes, color="#ff0000", linewidth=1.5,
                    fill_color=None, fill_alpha=0.3,
                    marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_circles` (backwards compatibility)."""
        return self.add_circles(offsets, sizes, color=color, linewidth=linewidth,
                                fill_color=fill_color, fill_alpha=fill_alpha,
                                marker_id=marker_id, label=label, labels=labels)

    # ------------------------------------------------------------------
    # Inspection / removal
    # ------------------------------------------------------------------
    def get_marker(self, marker_id: str) -> dict:
        """Return a copy of the marker-set dict for *marker_id*.

        Parameters
        ----------
        marker_id :
            ID returned by any ``add_*`` / ``set_*`` call.

        Returns
        -------
        dict
            Current state of the marker set — includes ``type``, spatial
            data, ``color``, ``id``, and optionally ``label`` / ``labels``.

        Raises
        ------
        KeyError
            If *marker_id* is not in the current marker list.
        """
        for ms in json.loads(self.markers_json):
            if ms.get("id") == marker_id:
                return dict(ms)
        raise KeyError(f"No marker set with id {marker_id!r}")

    def remove_marker(self, marker_id: str) -> None:
        """Remove a single marker set by ID.

        Parameters
        ----------
        marker_id :
            ID returned by any ``add_*`` / ``set_*`` call.

        Raises
        ------
        KeyError
            If *marker_id* is not in the current marker list.
        """
        sets = json.loads(self.markers_json)
        new_sets = [ms for ms in sets if ms.get("id") != marker_id]
        if len(new_sets) == len(sets):
            raise KeyError(f"No marker set with id {marker_id!r}")
        self.markers_json = json.dumps(new_sets)

    def list_markers(self) -> list[dict]:
        """Return a summary of every current marker set.

        Returns
        -------
        list of dict
            Each entry contains:

            * ``'id'`` – marker-set ID string.
            * ``'type'`` – marker type (e.g. ``'circles'``).
            * ``'color'`` – CSS colour string.
            * ``'n'`` – number of individual marks / segments.
            * ``'label'`` – collection tooltip (only present if set).
            * ``'labels'`` – per-marker tooltips (only present if set).
        """
        out: list[dict] = []
        for ms in json.loads(self.markers_json):
            t = ms.get("type", "?")
            n = len(ms.get("segments", [])) if t == "lines" else len(ms.get("offsets", []))
            entry: dict = {
                "id": ms.get("id"),
                "type": t,
                "color": ms.get("color"),
                "n": n,
            }
            if "label"  in ms:
                entry["label"]  = ms["label"]
            if "labels" in ms:
                entry["labels"] = ms["labels"]
            out.append(entry)
        return out

    def clear_markers(self) -> None:
        """Remove all marker overlays."""
        self.markers_json = "[]"


