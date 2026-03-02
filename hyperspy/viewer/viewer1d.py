"""
viewer1d.py
===========

Standalone 1-D line-plot viewer widget backed by ``anywidget`` / JavaScript
canvas.

Features
--------
* Accepts a 1-D NumPy array plus an optional physical ``x_axis`` array.
* X/Y axis rulers with tick labels (uniform or non-uniform x spacing).
* Smooth horizontal zoom (mouse wheel) and pan (drag).
* Resizable canvas (drag the corner handle).
* Press ``r`` to reset zoom and pan to the full data range.
* Status overlay (top-right corner) showing cursor ``x`` / ``y`` values.
* Multiple line series on the same plot via :meth:`add_line`.
* Vertical / horizontal span markers.
* PyCharm / static-notebook fallback via ``image/png`` MIME bundle.

Examples
--------
>>> import numpy as np
>>> from hyperspy.viewer.viewer1d import Viewer1D
>>> x = np.linspace(0, 10, 512)
>>> v = Viewer1D(np.sin(x), x_axis=x, units='s')
>>> v  # display in a Jupyter cell
"""

from __future__ import annotations

import base64
import io
import json
import uuid as _uuid

import anywidget
import numpy as np
import traitlets

__all__ = ["Viewer1D"]


class Viewer1D(anywidget.AnyWidget):
    """Interactive 1-D line-plot viewer widget.

    The widget renders in Jupyter / JupyterLab via ``anywidget``'s JavaScript
    runtime, and falls back to a static PNG line-plot in environments that
    only support ``image/png`` (e.g. PyCharm's notebook preview).

    Parameters
    ----------
    data : np.ndarray
        1-D signal array of shape ``(N,)``.
    x_axis : array-like, optional
        Physical coordinates for each sample (length ``N``).
        Defaults to ``np.arange(N)``.
    units : str, optional
        Physical unit label shown on the X axis (default ``'px'``).
    y_units : str, optional
        Unit label shown on the Y axis (default ``''``).
    color : str, optional
        CSS colour of the primary line (default ``'#4fc3f7'``).
    linewidth : float, optional
        Stroke width of the primary line in canvas pixels (default ``1.5``).
    label : str, optional
        Legend label for the primary line (default ``''``).

    Examples
    --------
    >>> import numpy as np
    >>> from hyperspy.viewer.viewer1d import Viewer1D
    >>> v = Viewer1D(np.random.rand(256), units='eV')
    >>> v
    """

    # ------------------------------------------------------------------ traits

    # Primary line: raw float64 values as JSON list.
    data_json   = traitlets.Unicode("[]").tag(sync=True)
    x_axis_json = traitlets.Unicode("[]").tag(sync=True)
    units       = traitlets.Unicode("px").tag(sync=True)
    y_units     = traitlets.Unicode("").tag(sync=True)

    # Canvas display size in CSS pixels.
    viewer_width  = traitlets.Int(480).tag(sync=True)
    viewer_height = traitlets.Int(256).tag(sync=True)

    # Data range (for y-axis scaling).
    data_min = traitlets.Float(0.0).tag(sync=True)
    data_max = traitlets.Float(1.0).tag(sync=True)

    # Zoom / pan over the x domain.  Both in fractional [0, 1] of x_axis span.
    view_x0 = traitlets.Float(0.0).tag(sync=True)   # left edge fraction
    view_x1 = traitlets.Float(1.0).tag(sync=True)   # right edge fraction

    # Additional line series — JSON list of {data, x_axis, color, linewidth, label}.
    extra_lines_json = traitlets.Unicode("[]").tag(sync=True)

    # Vertical / horizontal span markers — JSON list of span dicts.
    spans_json = traitlets.Unicode("[]").tag(sync=True)

    # Primary line style.
    line_color     = traitlets.Unicode("#4fc3f7").tag(sync=True)
    line_linewidth = traitlets.Float(1.5).tag(sync=True)
    line_label     = traitlets.Unicode("").tag(sync=True)

    # Marker overlays — JSON list of read-only marker-set dicts.
    # Each dict: { id, type, offsets/segments/…, color, linewidth,
    #              fill_color?, fill_alpha?, label?, labels? }
    # Offsets are [x] or [x, y] in data/axis units.
    # Types: points, vlines, hlines, lines, texts, rectangles, ellipses, polygons.
    markers_json = traitlets.Unicode("[]").tag(sync=True)

    # Interactive overlay widgets — JSON list of widget dicts.
    # Types and fields:
    #   vline:  { id, type:'vline',  x,       color }  x in data units
    #   hline:  { id, type:'hline',  y,       color }  y in data units
    #   range:  { id, type:'range',  x0, x1,  color }  x0/x1 in data units
    overlay_widgets_json = traitlets.Unicode("[]").tag(sync=True)

    # ------------------------------------------------------------------ JS
    _esm = r"""
    function render({ model, el }) {
      const dpr      = window.devicePixelRatio || 1;
      const PAD_L    = 58;   // left  padding (y-axis tick labels + rotated unit label)
      const PAD_R    = 12;   // right padding
      const PAD_T    = 12;   // top   padding
      const PAD_B    = 36;   // bottom padding (x-axis labels)

      // ── theme detection ────────────────────────────────────────────────────
      function _isDarkBg(el) {
        let node = el.parentElement;
        while (node && node !== document.body) {
          const bg = window.getComputedStyle(node).backgroundColor;
          const m  = bg.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
          if (m) {
            const [r, g, b] = [+m[1], +m[2], +m[3]];
            if (!(r === 0 && g === 0 && b === 0 && bg.includes('0)')))
              return (0.299 * r + 0.587 * g + 0.114 * b) < 128;
          }
          node = node.parentElement;
        }
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
      }

      function _makeTheme(dark) {
        return dark ? {
          bg:         '#1e1e2e',
          bgPlot:     '#181825',
          axisStroke: 'rgba(98,114,164,0.5)',
          gridStroke: 'rgba(98,114,164,0.18)',
          tickText:   'rgba(205,214,244,0.75)',
          unitText:   'rgba(98,114,164,0.85)',
          dark:       true,
        } : {
          bg:         '#f0f0f0',
          bgPlot:     '#ffffff',
          axisStroke: 'rgba(0,0,0,0.35)',
          gridStroke: 'rgba(0,0,0,0.07)',
          tickText:   'rgba(40,40,40,0.85)',
          unitText:   'rgba(100,100,100,0.85)',
          dark:       false,
        };
      }

      let theme = _makeTheme(_isDarkBg(el));

      function _applyTheme() {
        theme = _makeTheme(_isDarkBg(el));
        container.style.background = theme.bg;
        draw();
      }

      const _mq = window.matchMedia('(prefers-color-scheme: dark)');
      _mq.addEventListener('change', _applyTheme);
      const _themeObserver = new MutationObserver(_applyTheme);
      _themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-jp-theme-name', 'data-vscode-theme-kind', 'class'] });

      // ── DOM ────────────────────────────────────────────────────────────────
      const outerContainer = document.createElement('div');
      outerContainer.style.cssText = 'position:relative;display:inline-block;';

      const container = document.createElement('div');
      container.style.cssText =
        `background:${theme.bg};padding:10px;border-radius:4px;position:relative;display:inline-block;`;
      outerContainer.appendChild(container);

      // Plot canvas
      const canvas = document.createElement('canvas');
      canvas.tabIndex = 1;
      canvas.style.cssText =
        'outline:none;cursor:crosshair;display:block;border-radius:2px;';
      canvas.addEventListener('focus', () => { canvas.style.boxShadow = '0 0 0 2px rgba(76,175,80,0.5)'; });
      canvas.addEventListener('blur',  () => { canvas.style.boxShadow = 'none'; });

      // canvasWrap — zero-padding relative wrapper so all three canvases
      // share the exact same top-left origin (no container-padding offset).
      const canvasWrap = document.createElement('div');
      canvasWrap.style.cssText = 'position:relative;display:inline-block;line-height:0;';
      canvasWrap.appendChild(canvas);
      container.appendChild(canvasWrap);
      const ctx = canvas.getContext('2d');

      // Overlay canvas — interactive widgets (vline, hline, range).
      // pointer-events enabled; sits on top of the plot canvas.
      const overlayCanvas = document.createElement('canvas');
      overlayCanvas.style.cssText =
        'position:absolute;top:0;left:0;z-index:5;cursor:crosshair;';
      canvasWrap.appendChild(overlayCanvas);
      const ovCtx = overlayCanvas.getContext('2d');

      // Markers canvas — read-only marker overlays, pointer-events:none.
      const markersCanvas = document.createElement('canvas');
      markersCanvas.style.cssText =
        'position:absolute;top:0;left:0;pointer-events:none;z-index:6;';
      canvasWrap.appendChild(markersCanvas);
      const mkCtx = markersCanvas.getContext('2d');

      // Marker tooltip — fixed position, follows mouse.
      const markerTooltip = document.createElement('div');
      markerTooltip.style.cssText =
        'position:fixed;padding:5px 9px;font-size:12px;font-family:sans-serif;' +
        'background:rgba(30,30,30,0.92);color:#fff;border-radius:4px;' +
        'pointer-events:none;white-space:pre;display:none;z-index:9999;' +
        'box-shadow:0 2px 6px rgba(0,0,0,0.4);max-width:260px;';
      document.body.appendChild(markerTooltip);

      // Status overlay — cursor x / y, top-right corner.
      const statusBar = document.createElement('div');
      statusBar.style.cssText =
        'position:absolute;top:18px;right:18px;padding:2px 7px;' +
        'background:rgba(0,0,0,0.55);color:white;font-size:10px;font-family:monospace;' +
        'border-radius:4px;box-shadow:0 2px 4px rgba(0,0,0,0.4);pointer-events:none;' +
        'white-space:nowrap;display:none;z-index:8;';
      outerContainer.appendChild(statusBar);

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

      outerContainer.appendChild(resizeHandle);
      outerContainer.appendChild(sizeLabel);
      el.appendChild(outerContainer);

      // ── state ──────────────────────────────────────────────────────────────
      let isResizing = false, isPanning = false;
      let startX, startY, startW, startH;
      let panStartMx, panStartX0, panStartX1;
      let _suppressResize = false;

      // ── canvas sizing ──────────────────────────────────────────────────────
      function _applySize(w, h) {
        w = Math.round(w); h = Math.round(h);
        canvas.style.width  = w + 'px';
        canvas.style.height = h + 'px';
        canvas.width  = w * dpr;
        canvas.height = h * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        // Keep overlay canvas in sync.
        overlayCanvas.style.width  = w + 'px';
        overlayCanvas.style.height = h + 'px';
        overlayCanvas.width  = w * dpr;
        overlayCanvas.height = h * dpr;
        ovCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        // Keep markers canvas in sync.
        markersCanvas.style.width  = w + 'px';
        markersCanvas.style.height = h + 'px';
        markersCanvas.width  = w * dpr;
        markersCanvas.height = h * dpr;
        mkCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }

      function syncSize() {
        if (_suppressResize) return;
        const w = model.get('viewer_width'), h = model.get('viewer_height');
        const curW = parseInt(canvas.style.width)  || 0;
        const curH = parseInt(canvas.style.height) || 0;
        if (curW === w && curH === h) return;
        _applySize(w, h);
      }
      _applySize(model.get('viewer_width'), model.get('viewer_height'));

      // ── helpers ────────────────────────────────────────────────────────────
      function findNice(target) {
        if (target <= 0) return 1;
        const mag = Math.pow(10, Math.floor(Math.log10(target)));
        let best = mag, bestDiff = Math.abs(target - mag);
        for (const n of [1, 2, 2.5, 5, 10]) {
          const v = n * mag, d = Math.abs(target - v);
          if (d < bestDiff) { best = v; bestDiff = d; }
        }
        return best;
      }

      function fmtTick(v) {
        const a = Math.abs(v);
        if (a === 0)      return '0';
        if (a >= 1e4)     return v.toExponential(1);
        if (a >= 100)     return v.toFixed(0);
        if (a >= 1)       return v.toFixed(2);
        if (a >= 1e-2)    return v.toFixed(4);
        return v.toExponential(1);
      }

      /** Fractional x position [0,1] within x_axis array → physical value. */
      function _fracToX(xArr, frac) {
        if (xArr.length < 2) return xArr.length ? xArr[0] : 0;
        const n   = xArr.length;
        const pos = Math.max(0, Math.min(1, frac)) * (n - 1);
        const lo  = Math.min(Math.floor(pos), n - 2);
        const t   = pos - lo;
        return xArr[lo] + t * (xArr[lo + 1] - xArr[lo]);
      }

      /** Physical x value → fractional [0,1] position within x_axis. */
      function _xToFrac(xArr, val) {
        if (xArr.length < 2) return 0;
        const n   = xArr.length;
        const asc = xArr[n - 1] >= xArr[0];
        if (asc ? val <= xArr[0]     : val >= xArr[0])     return 0;
        if (asc ? val >= xArr[n - 1] : val <= xArr[n - 1]) return 1;
        let lo = 0, hi = n - 2;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          const inSeg = asc ? (xArr[mid] <= val && val < xArr[mid + 1])
                            : (xArr[mid] >= val && val > xArr[mid + 1]);
          if (inSeg) { lo = mid; break; }
          if (asc ? xArr[mid + 1] <= val : xArr[mid + 1] >= val) lo = mid + 1;
          else hi = mid;
        }
        return (lo + (val - xArr[lo]) / (xArr[lo + 1] - xArr[lo])) / (n - 1);
      }

      // ── plot geometry helpers ──────────────────────────────────────────────
      // Returns the plot area rect in CSS pixels.
      function _plotRect() {
        const w = parseInt(canvas.style.width)  || model.get('viewer_width');
        const h = parseInt(canvas.style.height) || model.get('viewer_height');
        return {
          x: PAD_L, y: PAD_T,
          w: Math.max(1, w - PAD_L - PAD_R),
          h: Math.max(1, h - PAD_T - PAD_B),
        };
      }

      // x fraction [0,1] of data → canvas pixel x within plot area.
      function _fracToCanvasX(frac) {
        const r  = _plotRect();
        const x0 = model.get('view_x0'), x1 = model.get('view_x1');
        return r.x + ((frac - x0) / ((x1 - x0) || 1)) * r.w;
      }

      // canvas pixel x → x fraction [0,1] of data.
      function _canvasXToFrac(px) {
        const r  = _plotRect();
        const x0 = model.get('view_x0'), x1 = model.get('view_x1');
        return x0 + ((px - r.x) / (r.w || 1)) * (x1 - x0);
      }

      // data value (raw) → canvas pixel y within plot area.
      function _valToCanvasY(val) {
        const r    = _plotRect();
        const dMin = model.get('data_min'), dMax = model.get('data_max');
        const range = (dMax - dMin) || 1;
        return r.y + r.h - ((val - dMin) / range) * r.h;
      }

      // ── draw ──────────────────────────────────────────────────────────────
      function draw() {
        const cw = parseInt(canvas.style.width)  || model.get('viewer_width');
        const ch = parseInt(canvas.style.height) || model.get('viewer_height');
        ctx.clearRect(0, 0, cw, ch);

        // Background
        ctx.fillStyle = theme.bg;
        ctx.fillRect(0, 0, cw, ch);

        const r    = _plotRect();
        const xArr = JSON.parse(model.get('x_axis_json'));
        const x0   = model.get('view_x0'), x1 = model.get('view_x1');
        const dMin = model.get('data_min'), dMax = model.get('data_max');
        const units  = model.get('units');
        const yUnits = model.get('y_units');

        // Plot area background
        ctx.fillStyle = theme.bgPlot;
        ctx.fillRect(r.x, r.y, r.w, r.h);

        // ── grid ──────────────────────────────────────────────────────────
        ctx.strokeStyle = theme.gridStroke;
        ctx.lineWidth   = 1;

        // X grid
        if (xArr.length >= 2) {
          const xVMin = _fracToX(xArr, x0), xVMax = _fracToX(xArr, x1);
          const xRange = xVMax - xVMin || 1;
          const xStep  = findNice(xRange / Math.max(2, Math.floor(r.w / 70)));
          for (let v = Math.ceil(xVMin / xStep) * xStep; v <= xVMax + xStep * 0.01; v += xStep) {
            const px = _fracToCanvasX(_xToFrac(xArr, v));
            if (px < r.x || px > r.x + r.w) continue;
            ctx.beginPath(); ctx.moveTo(px, r.y); ctx.lineTo(px, r.y + r.h); ctx.stroke();
          }
        }

        // Y grid
        const yRange = (dMax - dMin) || 1;
        const yStep  = findNice(yRange / Math.max(2, Math.floor(r.h / 40)));
        for (let v = Math.ceil(dMin / yStep) * yStep; v <= dMax + yStep * 0.01; v += yStep) {
          const py = _valToCanvasY(v);
          if (py < r.y || py > r.y + r.h) continue;
          ctx.beginPath(); ctx.moveTo(r.x, py); ctx.lineTo(r.x + r.w, py); ctx.stroke();
        }

        // ── spans ─────────────────────────────────────────────────────────
        let spans;
        try { spans = JSON.parse(model.get('spans_json')); } catch (_) { spans = []; }
        for (const sp of spans) {
          const color = sp.color || (theme.dark ? 'rgba(255,255,100,0.15)' : 'rgba(200,160,0,0.15)');
          ctx.fillStyle = color;
          if (sp.axis === 'x') {
            const f0 = _xToFrac(xArr, sp.v0), f1 = _xToFrac(xArr, sp.v1);
            const px0 = _fracToCanvasX(f0), px1 = _fracToCanvasX(f1);
            ctx.fillRect(px0, r.y, px1 - px0, r.h);
          } else {
            const py0 = _valToCanvasY(sp.v1), py1 = _valToCanvasY(sp.v0);
            ctx.fillRect(r.x, py0, r.w, py1 - py0);
          }
        }

        // ── clip to plot area ─────────────────────────────────────────────
        ctx.save();
        ctx.beginPath();
        ctx.rect(r.x, r.y, r.w, r.h);
        ctx.clip();

        // ── draw a single line series ──────────────────────────────────────
        function _drawLine(yData, lineXArr, color, lw) {
          if (!yData || yData.length === 0) return;
          const n = yData.length;
          ctx.beginPath();
          ctx.strokeStyle = color;
          ctx.lineWidth   = lw;
          ctx.lineJoin    = 'round';
          let first = true;
          for (let i = 0; i < n; i++) {
            const xFrac = lineXArr.length >= 2
              ? (lineXArr[i] - lineXArr[0]) / ((lineXArr[lineXArr.length - 1] - lineXArr[0]) || 1)
              : i / ((n - 1) || 1);
            const px = _fracToCanvasX(xFrac);
            const py = _valToCanvasY(yData[i]);
            if (first) { ctx.moveTo(px, py); first = false; }
            else        { ctx.lineTo(px, py); }
          }
          ctx.stroke();
        }

        // Primary line
        const yData = JSON.parse(model.get('data_json'));
        _drawLine(yData, xArr, model.get('line_color'), model.get('line_linewidth'));

        // Extra lines
        let extras;
        try { extras = JSON.parse(model.get('extra_lines_json')); } catch (_) { extras = []; }
        for (const ex of extras) {
          _drawLine(ex.data || [], ex.x_axis || xArr,
                    ex.color || (theme.dark ? '#ffffff' : '#333333'),
                    ex.linewidth || 1.5);
        }

        ctx.restore();   // un-clip

        // ── axes ──────────────────────────────────────────────────────────
        ctx.strokeStyle = theme.axisStroke;
        ctx.lineWidth   = 1;
        ctx.beginPath(); ctx.moveTo(r.x, r.y + r.h); ctx.lineTo(r.x + r.w, r.y + r.h); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(r.x, r.y); ctx.lineTo(r.x, r.y + r.h); ctx.stroke();

        ctx.fillStyle = theme.tickText;
        ctx.font      = '10px monospace';

        // X tick labels
        if (xArr.length >= 2) {
          const xVMin = _fracToX(xArr, x0), xVMax = _fracToX(xArr, x1);
          const xRange = xVMax - xVMin || 1;
          const xStep  = findNice(xRange / Math.max(2, Math.floor(r.w / 70)));
          ctx.textAlign    = 'center';
          ctx.textBaseline = 'top';
          for (let v = Math.ceil(xVMin / xStep) * xStep; v <= xVMax + xStep * 0.01; v += xStep) {
            const px = _fracToCanvasX(_xToFrac(xArr, v));
            if (px < r.x || px > r.x + r.w) continue;
            ctx.strokeStyle = theme.axisStroke;
            ctx.beginPath(); ctx.moveTo(px, r.y + r.h); ctx.lineTo(px, r.y + r.h + 5); ctx.stroke();
            ctx.fillStyle = theme.tickText;
            ctx.fillText(fmtTick(v), px, r.y + r.h + 7);
          }
          ctx.textAlign    = 'right';
          ctx.textBaseline = 'bottom';
          ctx.fillStyle    = theme.unitText;
          ctx.font         = '9px monospace';
          if (units && units !== 'px')
            ctx.fillText(units, r.x + r.w, r.y + r.h + PAD_B - 4);
          ctx.font = '10px monospace';
        }

        // Y tick labels
        ctx.font         = '10px monospace';
        ctx.textAlign    = 'right';
        ctx.textBaseline = 'middle';

        // Measure the widest tick string so we can position the unit label safely.
        let maxTickW = 0;
        for (let v = Math.ceil(dMin / yStep) * yStep; v <= dMax + yStep * 0.01; v += yStep) {
          const tw = ctx.measureText(fmtTick(v)).width;
          if (tw > maxTickW) maxTickW = tw;
        }
        // Tick text right-edge sits 8 px left of the axis line.
        const TICK_GAP   = 8;
        const tickRightX = r.x - TICK_GAP;

        for (let v = Math.ceil(dMin / yStep) * yStep; v <= dMax + yStep * 0.01; v += yStep) {
          const py = _valToCanvasY(v);
          if (py < r.y || py > r.y + r.h) continue;
          ctx.strokeStyle = theme.axisStroke;
          ctx.beginPath(); ctx.moveTo(r.x, py); ctx.lineTo(r.x - 5, py); ctx.stroke();
          ctx.fillStyle = theme.tickText;
          ctx.fillText(fmtTick(v), tickRightX, py);
        }

        if (yUnits) {
          ctx.save();
          // Centre the rotated label in the gap between the left canvas edge and
          // the start of the tick text (tickRightX - maxTickW).
          const tickLeftX  = tickRightX - maxTickW;
          const labelCentX = Math.max(8, tickLeftX / 2);
          ctx.translate(labelCentX, r.y + r.h / 2);
          ctx.rotate(-Math.PI / 2);
          ctx.textAlign    = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle    = theme.unitText;
          ctx.font         = '9px monospace';
          ctx.fillText(yUnits, 0, 0);
          ctx.restore();
        }

        // ── legend ────────────────────────────────────────────────────────
        const labels = [];
        if (model.get('line_label'))
          labels.push({ color: model.get('line_color'), text: model.get('line_label') });
        for (const ex of (extras || []))
          if (ex.label) labels.push({ color: ex.color || (theme.dark ? '#ffffff' : '#333333'), text: ex.label });

        if (labels.length > 0) {
          ctx.font = '10px monospace';
          let ly = r.y + 6;
          for (const lbl of labels) {
            ctx.strokeStyle = lbl.color; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(r.x + 8, ly + 5); ctx.lineTo(r.x + 24, ly + 5); ctx.stroke();
            ctx.fillStyle = theme.tickText;
            ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText(lbl.text, r.x + 28, ly);
            ly += 16;
          }
        }
      }

      // ── overlay widget draw/hit/drag ──────────────────────────────────────
      // All widget positions are in data-unit space (same units as x_axis/data).
      // Coordinate converters use the existing _fracToCanvasX / _valToCanvasY /
      // _canvasXToFrac helpers defined above.

      const OV_HANDLE_R    = 7;   // hit-test radius for handles (canvas px)
      const OV_HANDLE_DRAW = 5;   // drawn radius for handles
      const OV_FILL_ALPHA  = 0.15; // range widget fill opacity

      /** Data-unit x value → canvas px (using current view/zoom). */
      function _xUnitToPx(xVal) {
        const xArr = JSON.parse(model.get('x_axis_json'));
        const frac  = xArr.length >= 2 ? _xToFrac(xArr, xVal) : 0;
        return _fracToCanvasX(frac);
      }

      /** Canvas px x → data-unit x value. */
      function _pxToXUnit(px) {
        const xArr = JSON.parse(model.get('x_axis_json'));
        const frac  = _canvasXToFrac(px);
        return xArr.length >= 2 ? _fracToX(xArr, frac) : frac;
      }

      /** Data-unit y value → canvas px y. */
      function _yUnitToPx(yVal) { return _valToCanvasY(yVal); }

      /** Canvas px y → data-unit y value. */
      function _pxToYUnit(py) {
        const r    = _plotRect();
        const dMin = model.get('data_min'), dMax = model.get('data_max');
        return dMax - ((py - r.y) / (r.h || 1)) * (dMax - dMin);
      }

      /** Draw a small circular handle at (hx, hy). */
      function _ovHandle(hx, hy, color) {
        ovCtx.save();
        ovCtx.fillStyle   = color || '#00e5ff';
        ovCtx.strokeStyle = 'rgba(0,0,0,0.45)';
        ovCtx.lineWidth   = 1.5;
        ovCtx.beginPath();
        ovCtx.arc(hx, hy, OV_HANDLE_DRAW, 0, Math.PI * 2);
        ovCtx.fill();
        ovCtx.stroke();
        ovCtx.restore();
      }

      function drawOverlay() {
        const cw = parseInt(overlayCanvas.style.width)  || model.get('viewer_width');
        const ch = parseInt(overlayCanvas.style.height) || model.get('viewer_height');
        ovCtx.clearRect(0, 0, cw, ch);

        let widgets;
        try { widgets = JSON.parse(model.get('overlay_widgets_json')); }
        catch (_) { return; }
        if (!Array.isArray(widgets) || widgets.length === 0) return;

        const r = _plotRect();

        for (const w of widgets) {
          const color = w.color || '#00e5ff';
          ovCtx.save();
          ovCtx.strokeStyle = color;
          ovCtx.lineWidth   = 2;

          if (w.type === 'vline') {
            // Vertical line at w.x, clamped to the plot area.
            const px = _xUnitToPx(w.x);
            ovCtx.setLineDash([5, 3]);
            ovCtx.beginPath();
            ovCtx.moveTo(px, r.y);
            ovCtx.lineTo(px, r.y + r.h);
            ovCtx.stroke();
            ovCtx.setLineDash([]);
            // Handle at the top of the line.
            _ovHandle(px, r.y + OV_HANDLE_DRAW + 2, color);

          } else if (w.type === 'hline') {
            // Horizontal line at w.y.
            const py = _yUnitToPx(w.y);
            ovCtx.setLineDash([5, 3]);
            ovCtx.beginPath();
            ovCtx.moveTo(r.x, py);
            ovCtx.lineTo(r.x + r.w, py);
            ovCtx.stroke();
            ovCtx.setLineDash([]);
            // Handle at the right end.
            _ovHandle(r.x + r.w - OV_HANDLE_DRAW - 2, py, color);

          } else if (w.type === 'range') {
            // Shaded band between w.x0 and w.x1.
            const px0 = _xUnitToPx(w.x0);
            const px1 = _xUnitToPx(w.x1);
            const left  = Math.min(px0, px1);
            const right = Math.max(px0, px1);
            // Fill
            ovCtx.save();
            ovCtx.globalAlpha = OV_FILL_ALPHA;
            ovCtx.fillStyle   = color;
            ovCtx.fillRect(left, r.y, right - left, r.h);
            ovCtx.restore();
            // Left edge
            ovCtx.setLineDash([5, 3]);
            ovCtx.beginPath();
            ovCtx.moveTo(px0, r.y); ovCtx.lineTo(px0, r.y + r.h); ovCtx.stroke();
            // Right edge
            ovCtx.beginPath();
            ovCtx.moveTo(px1, r.y); ovCtx.lineTo(px1, r.y + r.h); ovCtx.stroke();
            ovCtx.setLineDash([]);
            // Handles at top of each edge only — body drag uses the shaded region.
            _ovHandle(px0, r.y + OV_HANDLE_DRAW + 2, color);
            _ovHandle(px1, r.y + OV_HANDLE_DRAW + 2, color);
          }

          ovCtx.restore();
        }
      }

      // ── overlay hit-test ──────────────────────────────────────────────────
      let ovDrag = null;   // { idx, mode:'move'|'edge0'|'edge1', startMX, snapW }

      function _ovHitTest(mx, my) {
        let widgets;
        try { widgets = JSON.parse(model.get('overlay_widgets_json')); }
        catch (_) { return null; }
        if (!Array.isArray(widgets)) return null;

        const r = _plotRect();

        // Reverse order so topmost widget wins.
        for (let i = widgets.length - 1; i >= 0; i--) {
          const w     = widgets[i];
          const color = w.color || '#00e5ff';

          if (w.type === 'vline') {
            const px = _xUnitToPx(w.x);
            const hx = px, hy = r.y + OV_HANDLE_DRAW + 2;
            // Handle
            if (Math.sqrt((mx - hx) ** 2 + (my - hy) ** 2) <= OV_HANDLE_R)
              return { idx: i, mode: 'move', wtype: 'vline', startMX: mx, snapW: { ...w } };
            // Line body
            if (Math.abs(mx - px) <= 5 && my >= r.y && my <= r.y + r.h)
              return { idx: i, mode: 'move', wtype: 'vline', startMX: mx, snapW: { ...w } };

          } else if (w.type === 'hline') {
            const py = _yUnitToPx(w.y);
            const hx = r.x + r.w - OV_HANDLE_DRAW - 2, hy = py;
            // Handle
            if (Math.sqrt((mx - hx) ** 2 + (my - hy) ** 2) <= OV_HANDLE_R)
              return { idx: i, mode: 'move', wtype: 'hline', startMY: my, snapW: { ...w } };
            // Line body
            if (Math.abs(my - py) <= 5 && mx >= r.x && mx <= r.x + r.w)
              return { idx: i, mode: 'move', wtype: 'hline', startMY: my, snapW: { ...w } };

          } else if (w.type === 'range') {
            const px0 = _xUnitToPx(w.x0), px1 = _xUnitToPx(w.x1);
            const left = Math.min(px0, px1), right = Math.max(px0, px1);
            // Left edge handle / edge line
            const lhx = px0, lhy = r.y + OV_HANDLE_DRAW + 2;
            if (Math.sqrt((mx - lhx) ** 2 + (my - lhy) ** 2) <= OV_HANDLE_R
                || (Math.abs(mx - px0) <= 5 && my >= r.y && my <= r.y + r.h))
              return { idx: i, mode: 'edge0', wtype: 'range', startMX: mx, snapW: { ...w } };
            // Right edge handle / edge line
            const rhx = px1, rhy = r.y + OV_HANDLE_DRAW + 2;
            if (Math.sqrt((mx - rhx) ** 2 + (my - rhy) ** 2) <= OV_HANDLE_R
                || (Math.abs(mx - px1) <= 5 && my >= r.y && my <= r.y + r.h))
              return { idx: i, mode: 'edge1', wtype: 'range', startMX: mx, snapW: { ...w } };
            // Body drag — anywhere in the shaded interior
            if (mx >= left && mx <= right && my >= r.y && my <= r.y + r.h)
              return { idx: i, mode: 'move', wtype: 'range', startMX: mx, snapW: { ...w } };
          }
        }
        return null;
      }

      // ── overlay mouse events ──────────────────────────────────────────────
      // overlayCanvas is always on top — it handles ALL mouse interaction:
      //   • widget hit → start widget drag
      //   • no hit     → start pan (same as before on canvas)
      // Wheel zoom is also handled here so it fires regardless of widget presence.

      overlayCanvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const r    = _plotRect();
        const frac = _canvasXToFrac(e.clientX - overlayCanvas.getBoundingClientRect().left);
        const x0   = model.get('view_x0'), x1 = model.get('view_x1');
        const span = x1 - x0;
        const factor = e.deltaY > 0 ? 1.15 : 0.87;
        let newSpan = Math.min(1.0, span * factor);
        let newX0   = frac - (frac - x0) * (newSpan / span);
        let newX1   = newX0 + newSpan;
        if (newX0 < 0) { newX0 = 0; newX1 = newSpan; }
        if (newX1 > 1) { newX1 = 1; newX0 = 1 - newSpan; }
        model.set('view_x0', newX0);
        model.set('view_x1', newX1);
        model.save_changes();
      }, { passive: false });

      overlayCanvas.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        canvas.focus();
        const rect = overlayCanvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const hit = _ovHitTest(mx, my);
        if (hit) {
          // Widget drag
          ovDrag = hit;
          overlayCanvas.style.cursor =
            (hit.mode === 'edge0' || hit.mode === 'edge1') ? 'ew-resize' : 'move';
        } else {
          // Fall through to pan
          isPanning   = true;
          panStartMx  = e.clientX;
          panStartX0  = model.get('view_x0');
          panStartX1  = model.get('view_x1');
          overlayCanvas.style.cursor = 'grabbing';
        }
        e.preventDefault();
      });

      document.addEventListener('mousemove', (e) => {
        if (ovDrag) {
          const rect = overlayCanvas.getBoundingClientRect();
          const mx = e.clientX - rect.left, my = e.clientY - rect.top;

          const xArr  = JSON.parse(model.get('x_axis_json'));
          const xFrac = _canvasXToFrac(mx);
          const xUnit = xArr.length >= 2 ? _fracToX(xArr, xFrac) : xFrac;
          const yUnit = _pxToYUnit(my);

          let widgets;
          try { widgets = JSON.parse(model.get('overlay_widgets_json')); }
          catch (_) { return; }

          const s = ovDrag.snapW;
          const w = widgets[ovDrag.idx];

          if (w.type === 'vline') {
            w.x = xUnit;
          } else if (w.type === 'hline') {
            w.y = yUnit;
          } else if (w.type === 'range') {
            if (ovDrag.mode === 'edge0') {
              w.x0 = xUnit;
            } else if (ovDrag.mode === 'edge1') {
              w.x1 = xUnit;
            } else {
              const snapPx0 = _fracToCanvasX(xArr.length >= 2 ? _xToFrac(xArr, s.x0) : 0);
              const dxUnit  = xArr.length >= 2
                ? _fracToX(xArr, _canvasXToFrac(snapPx0 + (mx - ovDrag.startMX))) - s.x0
                : (mx - ovDrag.startMX) / (_plotRect().w || 1);
              w.x0 = s.x0 + dxUnit;
              w.x1 = s.x1 + dxUnit;
            }
          }

          model.set('overlay_widgets_json', JSON.stringify(widgets));
          model.save_changes();
          drawOverlay();
          e.preventDefault();
          return;
        }

        if (isPanning) {
          const r    = _plotRect();
          const dx   = (e.clientX - panStartMx) / (r.w || 1);
          const span = panStartX1 - panStartX0;
          let newX0  = panStartX0 - dx * span;
          let newX1  = panStartX1 - dx * span;
          if (newX0 < 0) { newX0 = 0; newX1 = span; }
          if (newX1 > 1) { newX1 = 1; newX0 = 1 - span; }
          model.set('view_x0', newX0);
          model.set('view_x1', newX1);
          model.save_changes();
          e.preventDefault();
        }
      });

      document.addEventListener('mouseup', () => {
        if (ovDrag)    { ovDrag    = null;  overlayCanvas.style.cursor = 'crosshair'; }
        if (isPanning) { isPanning = false; overlayCanvas.style.cursor = 'crosshair'; }
      });

      // ── coordinate helpers for markers ───────────────────────────────────
      // _offsetToCanvas / _xDataToPx / _yDataToPx are called with pre-parsed
      // xArr and yData passed in so JSON.parse is not called inside hot loops.

      function _offsetToCanvas(off, xArr, yData) {
        const xFrac = xArr.length >= 2
          ? _xToFrac(xArr, off[0])
          : (off[0] / ((xArr.length - 1) || 1));
        const px = _fracToCanvasX(xFrac);
        let py;
        if (off.length >= 2 && off[1] != null) {
          py = _valToCanvasY(off[1]);
        } else if (yData.length > 1) {
          const n   = yData.length;
          const idx = Math.max(0, Math.min(n - 1, Math.round(xFrac * (n - 1))));
          py = _valToCanvasY(yData[idx]);
        } else {
          py = _valToCanvasY(0);
        }
        return [px, py];
      }

      function _xDataToPx(xVal, xArr) {
        return _fracToCanvasX(xArr.length >= 2 ? _xToFrac(xArr, xVal) : 0);
      }

      function _yDataToPx(yVal) { return _valToCanvasY(yVal); }

      // ── drawMarkers ────────────────────────────────────────────────────────
      function drawMarkers() {
        const cw = parseInt(markersCanvas.style.width)  || model.get('viewer_width');
        const ch = parseInt(markersCanvas.style.height) || model.get('viewer_height');
        mkCtx.clearRect(0, 0, cw, ch);

        let sets;
        try { sets = JSON.parse(model.get('markers_json')); }
        catch (_) { return; }
        if (!Array.isArray(sets) || sets.length === 0) return;

        // Parse once — shared by all marker sets in this frame.
        const xArr  = JSON.parse(model.get('x_axis_json'));
        const yData = JSON.parse(model.get('data_json'));

        const r = _plotRect();

        // Clip all markers to the plot area.
        mkCtx.save();
        mkCtx.beginPath();
        mkCtx.rect(r.x, r.y, r.w, r.h);
        mkCtx.clip();

        for (const ms of sets) {
          const color     = ms.color     || '#ff0000';
          const linewidth = ms.linewidth != null ? ms.linewidth : 1.5;
          const type      = ms.type || 'points';
          const fillColor = ms.fill_color || null;
          const fillAlpha = ms.fill_alpha != null ? ms.fill_alpha : 0.3;
          mkCtx.save();
          mkCtx.strokeStyle = color;
          mkCtx.fillStyle   = color;
          mkCtx.lineWidth   = linewidth;

          // ── points ──────────────────────────────────────────────────────
          if (type === 'points') {
            const offsets = ms.offsets || [];
            const sizes   = ms.sizes   || [];
            for (let i = 0; i < offsets.length; i++) {
              const [px, py] = _offsetToCanvas(offsets[i], xArr, yData);
              const sz = sizes[i] != null ? sizes[i] : (sizes[0] != null ? sizes[0] : 5);
              mkCtx.beginPath();
              mkCtx.arc(px, py, Math.max(1, sz), 0, Math.PI * 2);
              if (fillColor) {
                mkCtx.save();
                mkCtx.globalAlpha = fillAlpha;
                mkCtx.fillStyle = fillColor;
                mkCtx.fill();
                mkCtx.restore();
              }
              mkCtx.stroke();
            }

          // ── vlines ──────────────────────────────────────────────────────
          } else if (type === 'vlines') {
            const offsets = ms.offsets || [];
            for (let i = 0; i < offsets.length; i++) {
              const xFrac = xArr.length >= 2 ? _xToFrac(xArr, offsets[i][0]) : 0;
              const px = _fracToCanvasX(xFrac);
              mkCtx.beginPath();
              mkCtx.moveTo(px, r.y);
              mkCtx.lineTo(px, r.y + r.h);
              mkCtx.stroke();
            }

          // ── hlines ──────────────────────────────────────────────────────
          } else if (type === 'hlines') {
            const offsets = ms.offsets || [];
            for (let i = 0; i < offsets.length; i++) {
              const py = _yDataToPx(offsets[i][0]);
              mkCtx.beginPath();
              mkCtx.moveTo(r.x, py);
              mkCtx.lineTo(r.x + r.w, py);
              mkCtx.stroke();
            }

          // ── lines (segments) ────────────────────────────────────────────
          } else if (type === 'lines') {
            const segments = ms.segments || [];
            for (const seg of segments) {
              const [px1, py1] = _offsetToCanvas(seg[0], xArr, yData);
              const [px2, py2] = _offsetToCanvas(seg[1], xArr, yData);
              mkCtx.beginPath();
              mkCtx.moveTo(px1, py1);
              mkCtx.lineTo(px2, py2);
              mkCtx.stroke();
            }

          // ── rectangles ──────────────────────────────────────────────────
          } else if (type === 'rectangles') {
            const offsets = ms.offsets || [];
            const widths  = ms.widths  || [];
            const heights = ms.heights || [];
            for (let i = 0; i < offsets.length; i++) {
              const [pcx, pcy] = _offsetToCanvas(offsets[i], xArr, yData);
              const wVal = widths[i]  != null ? widths[i]  : (widths[0]  != null ? widths[0]  : 20);
              const hVal = heights[i] != null ? heights[i] : (heights[0] != null ? heights[0] : 20);
              const xCentre = offsets[i][0];
              const px0 = _xDataToPx(xCentre - wVal / 2, xArr);
              const px1 = _xDataToPx(xCentre + wVal / 2, xArr);
              const pw  = Math.abs(px1 - px0);
              const yCentre = offsets[i].length >= 2 ? offsets[i][1] : 0;
              const py0 = _yDataToPx(yCentre - hVal / 2);
              const py1 = _yDataToPx(yCentre + hVal / 2);
              const ph  = Math.abs(py1 - py0);
              if (fillColor) {
                mkCtx.save();
                mkCtx.globalAlpha = fillAlpha;
                mkCtx.fillStyle = fillColor;
                mkCtx.fillRect(pcx - pw / 2, pcy - ph / 2, pw, ph);
                mkCtx.restore();
              }
              mkCtx.strokeRect(pcx - pw / 2, pcy - ph / 2, pw, ph);
            }

          // ── ellipses ────────────────────────────────────────────────────
          } else if (type === 'ellipses') {
            const offsets = ms.offsets  || [];
            const widths  = ms.widths   || [];
            const heights = ms.heights  || [];
            for (let i = 0; i < offsets.length; i++) {
              const [pcx, pcy] = _offsetToCanvas(offsets[i], xArr, yData);
              const wVal = widths[i]  != null ? widths[i]  : (widths[0]  != null ? widths[0]  : 20);
              const hVal = heights[i] != null ? heights[i] : (heights[0] != null ? heights[0] : 20);
              const xCentre = offsets[i][0];
              const rw  = Math.abs(_xDataToPx(xCentre + wVal / 2, xArr) - _xDataToPx(xCentre - wVal / 2, xArr)) / 2;
              const yCentre = offsets[i].length >= 2 ? offsets[i][1] : 0;
              const rh  = Math.abs(_yDataToPx(yCentre - hVal / 2) - _yDataToPx(yCentre + hVal / 2)) / 2;
              mkCtx.beginPath();
              mkCtx.ellipse(pcx, pcy, Math.max(1, rw), Math.max(1, rh), 0, 0, Math.PI * 2);
              if (fillColor) {
                mkCtx.save();
                mkCtx.globalAlpha = fillAlpha;
                mkCtx.fillStyle = fillColor;
                mkCtx.fill();
                mkCtx.restore();
              }
              mkCtx.stroke();
            }

          // ── polygons ────────────────────────────────────────────────────
          } else if (type === 'polygons') {
            const vertsList = ms.vertices_list || [];
            for (let i = 0; i < vertsList.length; i++) {
              const verts = vertsList[i];
              if (!verts || verts.length < 2) continue;
              const [px0, py0] = _offsetToCanvas(verts[0], xArr, yData);
              mkCtx.beginPath();
              mkCtx.moveTo(px0, py0);
              for (let k = 1; k < verts.length; k++) {
                const [px, py] = _offsetToCanvas(verts[k], xArr, yData);
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

          // ── texts ────────────────────────────────────────────────────────
          } else if (type === 'texts') {
            const offsets  = ms.offsets || [];
            const texts    = ms.texts   || [];
            const fontsize = ms.fontsize != null ? ms.fontsize : 12;
            mkCtx.font         = `${fontsize}px sans-serif`;
            mkCtx.textAlign    = 'left';
            mkCtx.textBaseline = 'top';
            for (let i = 0; i < offsets.length; i++) {
              const [px, py] = _offsetToCanvas(offsets[i], xArr, yData);
              mkCtx.fillText(texts[i] != null ? String(texts[i]) : '', px, py);
            }
          }

          mkCtx.restore();
        }

        mkCtx.restore(); // un-clip
      }

      // ── marker hit-test & tooltip ─────────────────────────────────────────
      const MARKER_HIT = 8;

      function _markerHitTest(mx, my) {
        let sets;
        try { sets = JSON.parse(model.get('markers_json')); }
        catch (_) { return null; }
        if (!Array.isArray(sets)) return null;

        const xArr  = JSON.parse(model.get('x_axis_json'));
        const yData = JSON.parse(model.get('data_json'));

        for (let si = sets.length - 1; si >= 0; si--) {
          const ms         = sets[si];
          const type       = ms.type || 'points';
          const collLabel  = ms.label  != null ? String(ms.label) : null;
          const perLabels  = Array.isArray(ms.labels) ? ms.labels : null;
          if (collLabel === null && perLabels === null) continue;

          if (type === 'points') {
            const offsets = ms.offsets || [], sizes = ms.sizes || [];
            for (let i = 0; i < offsets.length; i++) {
              const [px, py] = _offsetToCanvas(offsets[i], xArr, yData);
              const sz = sizes[i] != null ? sizes[i] : (sizes[0] != null ? sizes[0] : 5);
              if (Math.sqrt((mx - px) ** 2 + (my - py) ** 2) <= sz + MARKER_HIT)
                return { collectionLabel: collLabel,
                         markerLabel: perLabels ? String(perLabels[i] ?? '') : null };
            }
          } else if (type === 'vlines') {
            const offsets = ms.offsets || [];
            for (let i = 0; i < offsets.length; i++) {
              const px = _fracToCanvasX(xArr.length >= 2 ? _xToFrac(xArr, offsets[i][0]) : 0);
              const r2 = _plotRect();
              if (Math.abs(mx - px) <= MARKER_HIT && my >= r2.y && my <= r2.y + r2.h)
                return { collectionLabel: collLabel,
                         markerLabel: perLabels ? String(perLabels[i] ?? '') : null };
            }
          } else if (type === 'hlines') {
            const offsets = ms.offsets || [];
            for (let i = 0; i < offsets.length; i++) {
              const py = _yDataToPx(offsets[i][0]);
              const r2 = _plotRect();
              if (Math.abs(my - py) <= MARKER_HIT && mx >= r2.x && mx <= r2.x + r2.w)
                return { collectionLabel: collLabel,
                         markerLabel: perLabels ? String(perLabels[i] ?? '') : null };
            }
          }
        }
        return null;
      }

      function _showTooltip(text, clientX, clientY) {
        markerTooltip.textContent = text;
        markerTooltip.style.display = 'block';
        const tw = markerTooltip.offsetWidth  || 160;
        const th = markerTooltip.offsetHeight || 28;
        const vw = window.innerWidth, vh = window.innerHeight;
        let lx = clientX + 14, ly = clientY - th - 8;
        if (lx + tw > vw - 8) lx = clientX - tw - 14;
        if (ly < 8)            ly = clientY + 18;
        markerTooltip.style.left = lx + 'px';
        markerTooltip.style.top  = ly + 'px';
      }

      // ── keyboard ──────────────────────────────────────────────────────────
      // overlayCanvas is the topmost element — give it a tabIndex so it can
      // receive keyboard events, and focus it on mouseenter.
      overlayCanvas.tabIndex = 0;
      overlayCanvas.style.outline = 'none';

      overlayCanvas.addEventListener('keydown', (e) => {
        if (e.key.toLowerCase() === 'r') {
          model.set('view_x0', 0.0);
          model.set('view_x1', 1.0);
          model.save_changes();
          e.preventDefault();
        }
      });
      overlayCanvas.addEventListener('mouseenter', () => overlayCanvas.focus());

      // ── status overlay ────────────────────────────────────────────────────
      function _fmtPhys(v) {
        if (!isFinite(v)) return '?';
        const a = Math.abs(v);
        if (a === 0)    return '0';
        if (a >= 1e4)   return v.toExponential(2);
        if (a >= 100)   return v.toFixed(1);
        if (a >= 1)     return v.toFixed(3);
        if (a >= 1e-2)  return v.toFixed(5);
        return v.toExponential(2);
      }

      // Shared logic — called from both canvas and overlayCanvas mousemove.
      function _updateStatus(clientX, clientY, canvasEl) {
        const rect = canvasEl.getBoundingClientRect();
        const mx   = clientX - rect.left;
        const my   = clientY - rect.top;
        const r    = _plotRect();
        if (mx < r.x || mx > r.x + r.w || my < r.y || my > r.y + r.h) {
          statusBar.style.display    = 'none';
          markerTooltip.style.display = 'none';
          return;
        }
        const xArr  = JSON.parse(model.get('x_axis_json'));
        const frac  = _canvasXToFrac(mx);
        const phys  = xArr.length >= 2 ? _fracToX(xArr, frac) : frac;
        const units = model.get('units');
        const yData = JSON.parse(model.get('data_json'));
        let yVal = '–';
        if (yData.length > 1) {
          const idx = Math.max(0, Math.min(yData.length - 1, Math.round(frac * (yData.length - 1))));
          yVal = _fmtPhys(yData[idx]);
        }
        const yUnits = model.get('y_units');
        const xStr = (units && units !== 'px')
          ? `x: ${_fmtPhys(phys)} ${units}`
          : `x: ${_fmtPhys(phys)}`;
        const yStr = yUnits ? `y: ${yVal} ${yUnits}` : `y: ${yVal}`;
        statusBar.textContent   = `${xStr}   ${yStr}`;
        statusBar.style.display = 'block';

        // Marker tooltip
        const mhit = _markerHitTest(mx, my);
        if (mhit) {
          const parts = [];
          if (mhit.collectionLabel) parts.push(mhit.collectionLabel);
          if (mhit.markerLabel)     parts.push(mhit.markerLabel);
          if (parts.length > 0) {
            _showTooltip(parts.join('\n'), clientX, clientY);
            return;
          }
        }
        markerTooltip.style.display = 'none';
      }

      canvas.addEventListener('mousemove', (e) => {
        _updateStatus(e.clientX, e.clientY, canvas);
      });

      canvas.addEventListener('mouseleave', () => {
        statusBar.style.display    = 'none';
        markerTooltip.style.display = 'none';
      });

      // Also update status when hovering over the overlay canvas (widgets).
      overlayCanvas.addEventListener('mousemove', (e) => {
        _updateStatus(e.clientX, e.clientY, overlayCanvas);
        if (ovDrag) return;
        const rect = overlayCanvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const hit = _ovHitTest(mx, my);
        if (hit) {
          if (hit.mode === 'edge0' || hit.mode === 'edge1') {
            overlayCanvas.style.cursor = 'ew-resize';
          } else if (hit.wtype === 'hline') {
            overlayCanvas.style.cursor = 'ns-resize';
          } else {
            overlayCanvas.style.cursor = 'move';
          }
        } else {
          overlayCanvas.style.cursor = 'crosshair';
        }
      });

      overlayCanvas.addEventListener('mouseleave', () => {
        statusBar.style.display     = 'none';
        markerTooltip.style.display = 'none';
        if (!ovDrag) overlayCanvas.style.cursor = 'crosshair';
      });
      resizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX; startY = e.clientY;
        startW = parseInt(canvas.style.width);
        startH = parseInt(canvas.style.height);
        sizeLabel.style.display = 'block';
        e.preventDefault();
      });

      document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        const nw = Math.max(200, startW + (e.clientX - startX));
        const nh = Math.max(100, startH + (e.clientY - startY));
        _applySize(nw, nh);
        sizeLabel.textContent = `${Math.round(nw)} × ${Math.round(nh)}`;
        draw();
        drawOverlay();
        drawMarkers();
        e.preventDefault();
      });

      document.addEventListener('mouseup', (e) => {
        if (!isResizing) return;
        isResizing = false;
        sizeLabel.style.display = 'none';
        const nw = parseInt(canvas.style.width);
        const nh = parseInt(canvas.style.height);
        _suppressResize = true;
        model.set('viewer_width',  nw);
        model.set('viewer_height', nh);
        model.save_changes();
        _suppressResize = false;
      });

      // ── model listeners ───────────────────────────────────────────────────
      model.on('change:data_json',         draw);
      model.on('change:x_axis_json',       draw);
      model.on('change:view_x0',           () => { draw(); drawOverlay(); drawMarkers(); });
      model.on('change:view_x1',           () => { draw(); drawOverlay(); drawMarkers(); });
      model.on('change:data_min',          draw);
      model.on('change:data_max',          draw);
      model.on('change:line_color',        draw);
      model.on('change:line_linewidth',    draw);
      model.on('change:line_label',        draw);
      model.on('change:extra_lines_json',  draw);
      model.on('change:spans_json',        draw);
      model.on('change:units',             draw);
      model.on('change:y_units',           draw);
      model.on('change:markers_json',      drawMarkers);
      model.on('change:overlay_widgets_json', drawOverlay);
      model.on('change:viewer_width change:viewer_height', () => {
        if (_suppressResize) return;
        syncSize(); draw(); drawOverlay(); drawMarkers();
      });

      // ── initial render ─────────────────────────────────────────────────────
      draw();
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
        units: str = "px",
        y_units: str = "",
        color: str = "#4fc3f7",
        linewidth: float = 1.5,
        label: str = "",
    ) -> None:
        """Initialise the viewer with a 1-D data array.

        Parameters
        ----------
        data :
            1-D signal array of shape ``(N,)``.
        x_axis :
            Physical coordinates for each sample (length ``N``).
            Defaults to ``np.arange(N)``.
        units :
            X-axis unit label (default ``'px'``).
        y_units :
            Y-axis unit label (default ``''``).
        color :
            CSS colour of the primary line (default ``'#4fc3f7'``).
        linewidth :
            Stroke width of the primary line in canvas pixels (default ``1.5``).
        label :
            Legend label for the primary line (default ``''``).
        """
        super().__init__()

        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise ValueError(f"data must be 1-D, got shape {data.shape}")

        n = len(data)

        if x_axis is None:
            x_axis = np.arange(n, dtype=float)
        x_axis = np.asarray(x_axis, dtype=float)
        if len(x_axis) != n:
            raise ValueError(
                f"x_axis length ({len(x_axis)}) must match data length ({n})"
            )

        dmin = float(np.nanmin(data))
        dmax = float(np.nanmax(data))
        # Add a little padding on y so the line isn't flush with the border.
        pad = (dmax - dmin) * 0.05 if dmax > dmin else 0.5
        dmin -= pad
        dmax += pad

        with self.hold_trait_notifications():
            self.data_json   = json.dumps(data.tolist())
            self.x_axis_json = json.dumps(x_axis.tolist())
            self.units       = units
            self.y_units     = y_units
            self.data_min    = dmin
            self.data_max    = dmax
            self.view_x0     = 0.0
            self.view_x1     = 1.0
            self.line_color     = color
            self.line_linewidth = float(linewidth)
            self.line_label     = label

    # ------------------------------------------------------------------
    def update(
        self,
        data: np.ndarray,
        x_axis: np.ndarray | None = None,
        units: str | None = None,
        y_units: str | None = None,
    ) -> None:
        """Replace the displayed data.

        The current zoom / pan view is preserved.  Extra lines and spans are
        *not* cleared — call :meth:`clear_lines` or :meth:`clear_spans`
        explicitly if needed.

        Parameters
        ----------
        data :
            New 1-D array of shape ``(N,)``.
        x_axis :
            New x-axis coordinates (length ``N``).  Re-uses the previous
            axis if ``None`` and the length is unchanged.
        units :
            X-axis unit label.  Unchanged if ``None``.
        y_units :
            Y-axis unit label.  Unchanged if ``None``.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise ValueError(f"data must be 1-D, got shape {data.shape}")

        n = len(data)

        if x_axis is None:
            prev = np.array(json.loads(self.x_axis_json))
            x_axis = prev if len(prev) == n else np.arange(n, dtype=float)
        x_axis = np.asarray(x_axis, dtype=float)
        if len(x_axis) != n:
            raise ValueError(
                f"x_axis length ({len(x_axis)}) must match data length ({n})"
            )

        dmin = float(np.nanmin(data))
        dmax = float(np.nanmax(data))
        pad  = (dmax - dmin) * 0.05 if dmax > dmin else 0.5
        dmin -= pad
        dmax += pad

        with self.hold_trait_notifications():
            self.data_json   = json.dumps(data.tolist())
            self.x_axis_json = json.dumps(x_axis.tolist())
            self.data_min    = dmin
            self.data_max    = dmax
            if units is not None:
                self.units = units
            if y_units is not None:
                self.y_units = y_units

    # ------------------------------------------------------------------
    def add_line(
        self,
        data: np.ndarray,
        x_axis: np.ndarray | None = None,
        color: str = "#ffffff",
        linewidth: float = 1.5,
        label: str = "",
    ) -> str:
        """Overlay an additional line series on the plot.

        Parameters
        ----------
        data :
            1-D array of shape ``(N,)``.
        x_axis :
            X-axis coordinates.  Uses the primary x_axis if ``None``.
        color :
            CSS colour (default ``'#ffffff'``).
        linewidth :
            Stroke width in canvas pixels (default ``1.5``).
        label :
            Legend label (default ``''``).

        Returns
        -------
        str
            Unique line ID.  Pass to :meth:`remove_line` to delete it.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise ValueError(f"data must be 1-D, got shape {data.shape}")

        if x_axis is None:
            xa = json.loads(self.x_axis_json)
        else:
            xa = np.asarray(x_axis, dtype=float).tolist()

        lid = str(_uuid.uuid4())[:8]
        extras = json.loads(self.extra_lines_json)
        extras.append({
            "id": lid,
            "data": data.tolist(),
            "x_axis": xa,
            "color": color,
            "linewidth": float(linewidth),
            "label": label,
        })
        self.extra_lines_json = json.dumps(extras)
        return lid

    def remove_line(self, lid: str) -> None:
        """Remove an extra line by ID.

        Parameters
        ----------
        lid :
            ID returned by :meth:`add_line`.

        Raises
        ------
        KeyError
            If *lid* does not correspond to a known line.
        """
        extras = json.loads(self.extra_lines_json)
        new = [e for e in extras if e.get("id") != lid]
        if len(new) == len(extras):
            raise KeyError(f"No extra line with id {lid!r}")
        self.extra_lines_json = json.dumps(new)

    def clear_lines(self) -> None:
        """Remove all extra line series."""
        self.extra_lines_json = "[]"

    # ------------------------------------------------------------------
    def add_span(
        self,
        v0: float,
        v1: float,
        axis: str = "x",
        color: str | None = None,
    ) -> str:
        """Add a coloured span band to the plot.

        Parameters
        ----------
        v0, v1 :
            Start and end of the span in data coordinates.
        axis :
            ``'x'`` for a vertical band (between two x values) or ``'y'``
            for a horizontal band (between two y values).
        color :
            CSS colour / rgba string.  Defaults to a semi-transparent yellow
            in dark mode or amber in light mode (chosen automatically by the JS).

        Returns
        -------
        str
            Unique span ID.
        """
        sid = str(_uuid.uuid4())[:8]
        spans = json.loads(self.spans_json)
        entry: dict = {
            "id": sid,
            "axis": axis,
            "v0": float(v0),
            "v1": float(v1),
        }
        if color is not None:
            entry["color"] = color
        spans.append(entry)
        self.spans_json = json.dumps(spans)
        return sid

    def remove_span(self, sid: str) -> None:
        """Remove a span by ID.

        Parameters
        ----------
        sid :
            ID returned by :meth:`add_span`.

        Raises
        ------
        KeyError
            If *sid* does not correspond to a known span.
        """
        spans = json.loads(self.spans_json)
        new = [s for s in spans if s.get("id") != sid]
        if len(new) == len(spans):
            raise KeyError(f"No span with id {sid!r}")
        self.spans_json = json.dumps(new)

    def clear_spans(self) -> None:
        """Remove all span bands."""
        self.spans_json = "[]"

    # ==================================================================
    # Overlay widget API  (interactive, draggable)
    # ==================================================================

    @staticmethod
    def _new_widget_id() -> str:
        return str(_uuid.uuid4())[:8]

    # ------------------------------------------------------------------
    def add_vline_widget(
        self,
        x: float | None = None,
        color: str = "#00e5ff",
    ) -> str:
        """Add a draggable vertical-line widget.

        The line spans the full plot height at the given x position (in
        data/axis units).  Drag anywhere on the line or its handle to move it.

        Parameters
        ----------
        x :
            Initial x position in data-axis units.  Defaults to the mid-point
            of the current x axis.
        color :
            CSS colour string (default cyan ``'#00e5ff'``).

        Returns
        -------
        str
            Widget ID — pass to :meth:`get_widget`, :meth:`set_vline_x`, or
            :meth:`remove_widget`.

        Examples
        --------
        >>> wid = v.add_vline_widget(x=3.14)
        >>> v.get_widget(wid)
        {'id': ..., 'type': 'vline', 'x': 3.14, 'color': '#00e5ff'}
        """
        xarr = json.loads(self.x_axis_json)
        if x is None:
            x = (xarr[0] + xarr[-1]) / 2.0 if len(xarr) >= 2 else 0.0
        entry = {
            "id":    self._new_widget_id(),
            "type":  "vline",
            "x":     float(x),
            "color": color,
        }
        widgets = json.loads(self.overlay_widgets_json)
        widgets.append(entry)
        self.overlay_widgets_json = json.dumps(widgets)
        return entry["id"]

    def set_vline_x(self, wid: str, x: float) -> None:
        """Move an existing vline widget to a new x position.

        Parameters
        ----------
        wid :
            Widget ID returned by :meth:`add_vline_widget`.
        x :
            New x position in data-axis units.

        Raises
        ------
        KeyError
            If *wid* is not a known widget.
        """
        widgets = json.loads(self.overlay_widgets_json)
        for w in widgets:
            if w["id"] == wid:
                w["x"] = float(x)
                self.overlay_widgets_json = json.dumps(widgets)
                return
        raise KeyError(f"No widget with id {wid!r}")

    # ------------------------------------------------------------------
    def add_hline_widget(
        self,
        y: float | None = None,
        color: str = "#00e5ff",
    ) -> str:
        """Add a draggable horizontal-line widget.

        The line spans the full plot width at the given y value.  Drag the
        line or its handle to change the y value interactively.

        Parameters
        ----------
        y :
            Initial y value in data units.  Defaults to the mid-point of the
            current data range.
        color :
            CSS colour string (default cyan ``'#00e5ff'``).

        Returns
        -------
        str
            Widget ID.

        Examples
        --------
        >>> wid = v.add_hline_widget(y=0.5)
        >>> v.get_widget(wid)
        {'id': ..., 'type': 'hline', 'y': 0.5, 'color': '#00e5ff'}
        """
        if y is None:
            y = (self.data_min + self.data_max) / 2.0
        entry = {
            "id":    self._new_widget_id(),
            "type":  "hline",
            "y":     float(y),
            "color": color,
        }
        widgets = json.loads(self.overlay_widgets_json)
        widgets.append(entry)
        self.overlay_widgets_json = json.dumps(widgets)
        return entry["id"]

    def set_hline_y(self, wid: str, y: float) -> None:
        """Move an existing hline widget to a new y value.

        Parameters
        ----------
        wid :
            Widget ID returned by :meth:`add_hline_widget`.
        y :
            New y value in data units.

        Raises
        ------
        KeyError
            If *wid* is not a known widget.
        """
        widgets = json.loads(self.overlay_widgets_json)
        for w in widgets:
            if w["id"] == wid:
                w["y"] = float(y)
                self.overlay_widgets_json = json.dumps(widgets)
                return
        raise KeyError(f"No widget with id {wid!r}")

    # ------------------------------------------------------------------
    def add_range_widget(
        self,
        x0: float | None = None,
        x1: float | None = None,
        color: str = "#00e5ff",
    ) -> str:
        """Add a draggable range (span) widget.

        A semi-transparent filled band is drawn between *x0* and *x1*.
        Three handles are shown:

        * Left-edge handle — drag to move the left boundary only.
        * Right-edge handle — drag to move the right boundary only.
        * Centre handle (bottom) — drag to translate the whole range.

        Dragging anywhere inside the shaded band also translates the range.

        Parameters
        ----------
        x0 :
            Left edge in data-axis units.  Defaults to 25 % of the axis span.
        x1 :
            Right edge in data-axis units.  Defaults to 75 % of the axis span.
        color :
            CSS colour string (default cyan ``'#00e5ff'``).

        Returns
        -------
        str
            Widget ID.

        Examples
        --------
        >>> wid = v.add_range_widget(x0=2.0, x1=4.0)
        >>> v.get_widget(wid)
        {'id': ..., 'type': 'range', 'x0': 2.0, 'x1': 4.0, 'color': '#00e5ff'}
        """
        xarr = json.loads(self.x_axis_json)
        xmin = float(xarr[0])  if len(xarr) >= 2 else 0.0
        xmax = float(xarr[-1]) if len(xarr) >= 2 else 1.0
        span = xmax - xmin or 1.0
        if x0 is None:
            x0 = xmin + span * 0.25
        if x1 is None:
            x1 = xmin + span * 0.75
        entry = {
            "id":    self._new_widget_id(),
            "type":  "range",
            "x0":    float(x0),
            "x1":    float(x1),
            "color": color,
        }
        widgets = json.loads(self.overlay_widgets_json)
        widgets.append(entry)
        self.overlay_widgets_json = json.dumps(widgets)
        return entry["id"]

    def set_range(self, wid: str, x0: float, x1: float) -> None:
        """Update the x0/x1 boundaries of an existing range widget.

        Parameters
        ----------
        wid :
            Widget ID returned by :meth:`add_range_widget`.
        x0 :
            New left edge in data-axis units.
        x1 :
            New right edge in data-axis units.

        Raises
        ------
        KeyError
            If *wid* is not a known widget.
        """
        widgets = json.loads(self.overlay_widgets_json)
        for w in widgets:
            if w["id"] == wid:
                w["x0"] = float(x0)
                w["x1"] = float(x1)
                self.overlay_widgets_json = json.dumps(widgets)
                return
        raise KeyError(f"No widget with id {wid!r}")

    # ------------------------------------------------------------------
    def get_widget(self, wid: str) -> dict:
        """Return a copy of the current state of an overlay widget.

        Parameters
        ----------
        wid :
            Widget ID returned by any ``add_*_widget`` call.

        Returns
        -------
        dict
            Contains ``'id'``, ``'type'``, ``'color'``, and type-specific
            position fields (``'x'`` for vline, ``'y'`` for hline,
            ``'x0'``/``'x1'`` for range).

        Raises
        ------
        KeyError
            If *wid* does not correspond to a known widget.
        """
        for w in json.loads(self.overlay_widgets_json):
            if w["id"] == wid:
                return dict(w)
        raise KeyError(f"No widget with id {wid!r}")

    def list_widgets(self) -> list[dict]:
        """Return a summary list of all current overlay widgets.

        Returns
        -------
        list of dict
            Each entry contains ``'id'``, ``'type'``, ``'color'``, and
            the type-specific position key(s).
        """
        return [dict(w) for w in json.loads(self.overlay_widgets_json)]

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
        widgets = json.loads(self.overlay_widgets_json)
        new_widgets = [w for w in widgets if w["id"] != wid]
        if len(new_widgets) == len(widgets):
            raise KeyError(f"No widget with id {wid!r}")
        self.overlay_widgets_json = json.dumps(new_widgets)

    def clear_widgets(self) -> None:
        """Remove all overlay widgets."""
        self.overlay_widgets_json = "[]"

    # ==================================================================
    # Marker overlay API
    # ==================================================================

    @staticmethod
    def _new_marker_id() -> str:
        return str(_uuid.uuid4())[:8]

    def _write_marker(self, ms: dict, marker_id: str | None) -> str:
        """Insert or replace a marker set in ``markers_json``.

        Parameters
        ----------
        ms :
            Marker-set dict (without ``"id"``).
        marker_id :
            Existing ID to replace, or ``None`` to append a new set.

        Returns
        -------
        str  The final marker ID.
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

        A scalar is repeated *n* times; a 1-D array must have exactly *n*
        elements.

        Parameters
        ----------
        arr :
            Scalar or 1-D array-like.
        n :
            Required length.
        name :
            Parameter name for error messages.

        Returns
        -------
        list of float
        """
        a = np.asarray(arr, dtype=float)
        if a.ndim == 0:
            return np.full(n, float(a)).tolist()
        if a.ndim != 1 or len(a) != n:
            raise ValueError(
                f"'{name}' must be a scalar or 1-D array of length {n}"
            )
        return a.tolist()

    @staticmethod
    def _check_offsets_1d(offsets: object) -> np.ndarray:
        """Validate and normalise *offsets* for 1-D markers.

        Accepted input shapes and their interpretations:

        * ``(N,)`` — N scalar values, each becomes one ``[val]`` row → ``(N, 1)``
        * ``(N, 1)`` — N single-coordinate rows, returned as-is
        * ``(N, 2)`` — N ``[x, y]`` rows, returned as-is
        * ``(1, 2)`` or ``[[x, y]]`` — single ``[x, y]`` pair, returned as ``(1, 2)``

        A plain Python scalar is also accepted and becomes ``(1, 1)``.

        Parameters
        ----------
        offsets :
            Array-like of any of the shapes above.

        Returns
        -------
        np.ndarray of shape (N, 1) or (N, 2)
        """
        arr = np.asarray(offsets, dtype=float)
        # Scalar → [[val]]
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        # Plain 1-D array of N values → (N, 1).
        # This covers add_hlines([0.5, -0.5]) → two rows [[0.5],[−0.5]].
        if arr.ndim == 1:
            return arr[:, np.newaxis]
        if arr.ndim == 2 and arr.shape[1] in (1, 2):
            return arr
        raise ValueError(
            "offsets must be a 1-D array of values, or shape (N, 1) / (N, 2) — "
            "each row is [x] or [x, y]"
        )

    @staticmethod
    def _opt_labels(
        label: str | None,
        labels: list | None,
        n: int,
    ) -> dict:
        """Build the optional ``label``/``labels`` sub-dict.

        Parameters
        ----------
        label :
            Collection-level hover tooltip, or ``None``.
        labels :
            Per-marker hover tooltips (must have length *n*), or ``None``.
        n :
            Number of markers.

        Returns
        -------
        dict
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
            extra["labels"] = [str(lbl) for lbl in ls]
        return extra

    # ------------------------------------------------------------------
    # Points
    # ------------------------------------------------------------------
    def add_points(
        self,
        offsets,
        sizes=5,
        color: str = "#ff0000",
        linewidth: float = 1.5,
        fill_color: str | None = None,
        fill_alpha: float = 0.3,
        marker_id: str | None = None,
        label: str | None = None,
        labels: list | None = None,
    ) -> str:
        """Add or replace a set of point (circle) markers.

        Parameters
        ----------
        offsets : array-like
            ``[[x], ...]`` or ``[[x, y], ...]``.  If only *x* is given the
            y-position is sampled from the primary data at that x.
        sizes : array-like or scalar
            Radius in canvas pixels (default ``5``).
        color : str, optional
            Stroke colour (default ``'#ff0000'``).
        linewidth : float, optional
            Stroke width in canvas pixels.
        fill_color : str or None, optional
            Fill colour; ``None`` draws outline only.
        fill_alpha : float, optional
            Fill opacity in ``[0, 1]`` (default ``0.3``).
        marker_id : str, optional
            Replace an existing set if supplied.
        label : str, optional
            Collection hover tooltip.
        labels : list of str, optional
            Per-marker hover tooltips.

        Returns
        -------
        str  Marker ID.
        """
        arr = self._check_offsets_1d(offsets)
        n = len(arr)
        ms: dict = {
            "type": "points",
            "offsets": arr.tolist(),
            "sizes": self._broadcast_1d(sizes, n, "sizes"),
            "color": color,
            "linewidth": linewidth,
            **self._opt_labels(label, labels, n),
        }
        if fill_color is not None:
            ms["fill_color"] = fill_color
            ms["fill_alpha"] = float(fill_alpha)
        return self._write_marker(ms, marker_id)

    def set_points(self, offsets, sizes=5, color="#ff0000", linewidth=1.5,
                   fill_color=None, fill_alpha=0.3,
                   marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_points`."""
        return self.add_points(offsets, sizes, color=color, linewidth=linewidth,
                               fill_color=fill_color, fill_alpha=fill_alpha,
                               marker_id=marker_id, label=label, labels=labels)

    # ------------------------------------------------------------------
    # Vertical lines
    # ------------------------------------------------------------------
    def add_vlines(
        self,
        offsets,
        color: str = "#ff0000",
        linewidth: float = 1.5,
        marker_id: str | None = None,
        label: str | None = None,
        labels: list | None = None,
    ) -> str:
        """Add or replace a set of vertical line markers.

        Parameters
        ----------
        offsets : array-like
            ``[[x], ...]`` or a 1-D list of x values.  Each becomes a
            vertical line spanning the full height of the plot.
        color : str, optional
        linewidth : float, optional
        marker_id : str, optional
        label : str, optional
        labels : list of str, optional

        Returns
        -------
        str  Marker ID.
        """
        arr = self._check_offsets_1d(offsets)
        # Keep only the x column.
        arr = arr[:, :1]
        n = len(arr)
        ms: dict = {
            "type": "vlines",
            "offsets": arr.tolist(),
            "color": color,
            "linewidth": linewidth,
            **self._opt_labels(label, labels, n),
        }
        return self._write_marker(ms, marker_id)

    def set_vlines(self, offsets, color="#ff0000", linewidth=1.5,
                   marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_vlines`."""
        return self.add_vlines(offsets, color=color, linewidth=linewidth,
                               marker_id=marker_id, label=label, labels=labels)

    # ------------------------------------------------------------------
    # Horizontal lines
    # ------------------------------------------------------------------
    def add_hlines(
        self,
        offsets,
        color: str = "#ff0000",
        linewidth: float = 1.5,
        marker_id: str | None = None,
        label: str | None = None,
        labels: list | None = None,
    ) -> str:
        """Add or replace a set of horizontal line markers.

        Parameters
        ----------
        offsets : array-like
            ``[[y], ...]`` or a 1-D list of y values.  Each becomes a
            horizontal line spanning the full width of the plot.
        color : str, optional
        linewidth : float, optional
        marker_id : str, optional
        label : str, optional
        labels : list of str, optional

        Returns
        -------
        str  Marker ID.
        """
        arr = self._check_offsets_1d(offsets)
        arr = arr[:, :1]
        n = len(arr)
        ms: dict = {
            "type": "hlines",
            "offsets": arr.tolist(),
            "color": color,
            "linewidth": linewidth,
            **self._opt_labels(label, labels, n),
        }
        return self._write_marker(ms, marker_id)

    def set_hlines(self, offsets, color="#ff0000", linewidth=1.5,
                   marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_hlines`."""
        return self.add_hlines(offsets, color=color, linewidth=linewidth,
                               marker_id=marker_id, label=label, labels=labels)

    # ------------------------------------------------------------------
    # Line segments
    # ------------------------------------------------------------------
    def add_lines(
        self,
        segments,
        color: str = "#ff0000",
        linewidth: float = 1.5,
        marker_id: str | None = None,
        label: str | None = None,
        labels: list | None = None,
    ) -> str:
        """Add or replace a set of line-segment markers.

        Parameters
        ----------
        segments : array-like (N, 2, 2)
            ``[[[x1, y1], [x2, y2]], ...]`` in data-unit space.
        color : str, optional
        linewidth : float, optional
        marker_id : str, optional
        label : str, optional
        labels : list of str, optional  One per segment.

        Returns
        -------
        str  Marker ID.
        """
        segs = np.asarray(segments, dtype=float)
        if segs.ndim == 2 and segs.shape == (2, 2):
            segs = segs[np.newaxis]
        if segs.ndim != 3 or segs.shape[1:] != (2, 2):
            raise ValueError("segments must be shape (N, 2, 2)")
        n = len(segs)
        ms: dict = {
            "type": "lines",
            "segments": segs.tolist(),
            "color": color,
            "linewidth": linewidth,
            **self._opt_labels(label, labels, n),
        }
        return self._write_marker(ms, marker_id)

    def set_lines(self, segments, color="#ff0000", linewidth=1.5,
                  marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_lines`."""
        return self.add_lines(segments, color=color, linewidth=linewidth,
                              marker_id=marker_id, label=label, labels=labels)

    # ------------------------------------------------------------------
    # Rectangles
    # ------------------------------------------------------------------
    def add_rectangles(
        self,
        offsets,
        widths,
        heights,
        color: str = "#ff0000",
        linewidth: float = 1.5,
        fill_color: str | None = None,
        fill_alpha: float = 0.3,
        marker_id: str | None = None,
        label: str | None = None,
        labels: list | None = None,
    ) -> str:
        """Add or replace a set of rectangle markers.

        Parameters
        ----------
        offsets : array-like
            ``[[x, y], ...]`` centre positions in data-unit space.
        widths : array-like or scalar
            Full width in x data units.
        heights : array-like or scalar
            Full height in y data units.
        color : str, optional
        linewidth : float, optional
        fill_color : str or None, optional
        fill_alpha : float, optional
        marker_id : str, optional
        label : str, optional
        labels : list of str, optional

        Returns
        -------
        str  Marker ID.
        """
        arr = self._check_offsets_1d(offsets)
        n = len(arr)
        ms: dict = {
            "type": "rectangles",
            "offsets": arr.tolist(),
            "widths":  self._broadcast_1d(widths,  n, "widths"),
            "heights": self._broadcast_1d(heights, n, "heights"),
            "color": color,
            "linewidth": linewidth,
            **self._opt_labels(label, labels, n),
        }
        if fill_color is not None:
            ms["fill_color"] = fill_color
            ms["fill_alpha"] = float(fill_alpha)
        return self._write_marker(ms, marker_id)

    def set_rectangles(self, offsets, widths, heights, color="#ff0000",
                       linewidth=1.5, fill_color=None, fill_alpha=0.3,
                       marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_rectangles`."""
        return self.add_rectangles(offsets, widths, heights, color=color,
                                   linewidth=linewidth, fill_color=fill_color,
                                   fill_alpha=fill_alpha, marker_id=marker_id,
                                   label=label, labels=labels)

    # ------------------------------------------------------------------
    # Ellipses
    # ------------------------------------------------------------------
    def add_ellipses(
        self,
        offsets,
        widths,
        heights,
        color: str = "#ff0000",
        linewidth: float = 1.5,
        fill_color: str | None = None,
        fill_alpha: float = 0.3,
        marker_id: str | None = None,
        label: str | None = None,
        labels: list | None = None,
    ) -> str:
        """Add or replace a set of ellipse markers.

        Parameters
        ----------
        offsets : array-like
            ``[[x, y], ...]`` centres in data-unit space.
        widths : array-like or scalar
            Full width in x data units.
        heights : array-like or scalar
            Full height in y data units.
        color : str, optional
        linewidth : float, optional
        fill_color : str or None, optional
        fill_alpha : float, optional
        marker_id : str, optional
        label : str, optional
        labels : list of str, optional

        Returns
        -------
        str  Marker ID.
        """
        arr = self._check_offsets_1d(offsets)
        n = len(arr)
        ms: dict = {
            "type": "ellipses",
            "offsets": arr.tolist(),
            "widths":  self._broadcast_1d(widths,  n, "widths"),
            "heights": self._broadcast_1d(heights, n, "heights"),
            "color": color,
            "linewidth": linewidth,
            **self._opt_labels(label, labels, n),
        }
        if fill_color is not None:
            ms["fill_color"] = fill_color
            ms["fill_alpha"] = float(fill_alpha)
        return self._write_marker(ms, marker_id)

    def set_ellipses(self, offsets, widths, heights, color="#ff0000",
                     linewidth=1.5, fill_color=None, fill_alpha=0.3,
                     marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_ellipses`."""
        return self.add_ellipses(offsets, widths, heights, color=color,
                                 linewidth=linewidth, fill_color=fill_color,
                                 fill_alpha=fill_alpha, marker_id=marker_id,
                                 label=label, labels=labels)

    # ------------------------------------------------------------------
    # Polygons
    # ------------------------------------------------------------------
    def add_polygons(
        self,
        vertices_list,
        color: str = "#ff0000",
        linewidth: float = 1.5,
        fill_color: str | None = None,
        fill_alpha: float = 0.3,
        marker_id: str | None = None,
        label: str | None = None,
        labels: list | None = None,
    ) -> str:
        """Add or replace a set of polygon markers.

        Parameters
        ----------
        vertices_list : list of array-like
            Each element is one polygon: ``[[x, y], ...]`` in data-unit
            space, minimum 3 vertices.
        color : str, optional
        linewidth : float, optional
        fill_color : str or None, optional
        fill_alpha : float, optional
        marker_id : str, optional
        label : str, optional
        labels : list of str, optional  One per polygon.

        Returns
        -------
        str  Marker ID.
        """
        vlist = []
        for poly in vertices_list:
            arr = np.asarray(poly, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("each polygon must be shape (N, 2)")
            if len(arr) < 3:
                raise ValueError("each polygon needs at least 3 vertices")
            vlist.append(arr.tolist())
        n = len(vlist)
        ms: dict = {
            "type": "polygons",
            "vertices_list": vlist,
            "color": color,
            "linewidth": linewidth,
            **self._opt_labels(label, labels, n),
        }
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
    def add_texts(
        self,
        offsets,
        texts: list,
        color: str = "#ff0000",
        fontsize: int = 12,
        marker_id: str | None = None,
        label: str | None = None,
        labels: list | None = None,
    ) -> str:
        """Add or replace a set of text markers.

        Parameters
        ----------
        offsets : array-like
            ``[[x], ...]`` or ``[[x, y], ...]`` anchor positions.
        texts : list of str
            Displayed text for each offset.
        color : str, optional
        fontsize : int, optional
        marker_id : str, optional
        label : str, optional
            Collection hover tooltip (distinct from the displayed texts).
        labels : list of str, optional
            Per-marker hover tooltip; defaults to *texts* if omitted.

        Returns
        -------
        str  Marker ID.
        """
        arr = self._check_offsets_1d(offsets)
        texts = list(texts)
        n = len(arr)
        if len(texts) != n:
            raise ValueError("len(texts) must equal len(offsets)")
        # Default per-marker hover label to the displayed text.
        if labels is None:
            labels = texts
        ms: dict = {
            "type": "texts",
            "offsets": arr.tolist(),
            "texts": texts,
            "color": color,
            "fontsize": fontsize,
            **self._opt_labels(label, labels, n),
        }
        return self._write_marker(ms, marker_id)

    def set_texts(self, offsets, texts, color="#ff0000", fontsize=12,
                  marker_id=None, label=None, labels=None) -> str:
        """Alias for :meth:`add_texts`."""
        return self.add_texts(offsets, texts, color=color, fontsize=fontsize,
                              marker_id=marker_id, label=label, labels=labels)

    # ------------------------------------------------------------------
    # Inspection / removal
    # ------------------------------------------------------------------
    def get_marker(self, marker_id: str) -> dict:
        """Return a copy of the marker-set dict for *marker_id*.

        Parameters
        ----------
        marker_id :
            ID returned by any ``add_*`` / ``set_*`` marker call.

        Returns
        -------
        dict

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
            ID returned by any ``add_*`` / ``set_*`` marker call.

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
            Each entry contains ``'id'``, ``'type'``, ``'color'``, ``'n'``,
            and optionally ``'label'`` / ``'labels'``.
        """
        out: list[dict] = []
        for ms in json.loads(self.markers_json):
            t = ms.get("type", "?")
            if t == "lines":
                n = len(ms.get("segments", []))
            elif t == "polygons":
                n = len(ms.get("vertices_list", []))
            else:
                n = len(ms.get("offsets", []))
            entry: dict = {
                "id":    ms.get("id"),
                "type":  t,
                "color": ms.get("color"),
                "n":     n,
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

    # ------------------------------------------------------------------
    def set_view(
        self,
        x0: float | None = None,
        x1: float | None = None,
    ) -> None:
        """Set the visible x range in physical axis units.

        Parameters
        ----------
        x0 :
            Left edge of the view in physical x units.  Defaults to the
            start of the x axis.
        x1 :
            Right edge of the view in physical x units.  Defaults to the
            end of the x axis.

        Examples
        --------
        >>> v.set_view(1.0, 5.0)   # zoom to x ∈ [1, 5]
        >>> v.set_view()           # reset to full view
        """
        xarr = np.array(json.loads(self.x_axis_json))
        if len(xarr) < 2:
            return
        xmin, xmax = float(xarr[0]), float(xarr[-1])
        span = xmax - xmin or 1.0

        f0 = 0.0 if x0 is None else max(0.0, min(1.0, (float(x0) - xmin) / span))
        f1 = 1.0 if x1 is None else max(0.0, min(1.0, (float(x1) - xmin) / span))
        if f0 >= f1:
            raise ValueError("x0 must be less than x1")
        with self.hold_trait_notifications():
            self.view_x0 = f0
            self.view_x1 = f1

    def reset_view(self) -> None:
        """Reset zoom and pan to show the full x range."""
        with self.hold_trait_notifications():
            self.view_x0 = 0.0
            self.view_x1 = 1.0

    # ------------------------------------------------------------------
    def _repr_mimebundle_(self, **kwargs: object) -> dict:
        """Return the anywidget MIME bundle plus a ``image/png`` fallback."""
        bundle: dict = super()._repr_mimebundle_(**kwargs) or {}
        try:
            bundle["image/png"] = base64.b64encode(
                self._to_png_bytes()
            ).decode("ascii")
        except Exception:
            pass
        return bundle

    def _to_png_bytes(self) -> bytes:
        """Render a simple matplotlib line-plot as PNG for static previews."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data  = np.array(json.loads(self.data_json))
        xarr  = np.array(json.loads(self.x_axis_json))
        if len(xarr) != len(data):
            xarr = np.arange(len(data), dtype=float)

        fig, ax = plt.subplots(figsize=(4.8, 2.56), dpi=100)
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#181825")
        ax.tick_params(colors="0.6")
        for spine in ax.spines.values():
            spine.set_edgecolor("0.35")
        ax.plot(xarr, data, color=self.line_color,
                linewidth=self.line_linewidth,
                label=self.line_label or None)

        extras = json.loads(self.extra_lines_json)
        for ex in extras:
            exa = np.array(ex.get("x_axis") or xarr)
            ax.plot(exa, ex["data"],
                    color=ex.get("color", "#ffffff"),
                    linewidth=ex.get("linewidth", 1.5),
                    label=ex.get("label") or None)

        if self.units and self.units != "px":
            ax.set_xlabel(self.units, color="0.6")
        if self.y_units:
            ax.set_ylabel(self.y_units, color="0.6")
        if self.line_label or any(e.get("label") for e in extras):
            ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        return buf.getvalue()

