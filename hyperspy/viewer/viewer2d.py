"""
viewer2d.py

A standalone 2D image viewer widget that accepts a numpy array directly.
Supports:
 - x_axis / y_axis arrays for physical coordinates
 - Scale bar when x and y have equal (uniform) scales
 - X/Y axis rulers when scales differ
 - Zoom, pan, histogram, resize
"""

import base64
import io
import json

import anywidget
import numpy as np
import traitlets


def _is_uniform(axis: np.ndarray, rtol: float = 1e-3) -> bool:
    """Return True if the axis has a constant step size."""
    if len(axis) < 2:
        return True
    steps = np.diff(axis)
    return bool(np.allclose(steps, steps[0], rtol=rtol))


def _equal_scale(x_axis: np.ndarray, y_axis: np.ndarray, rtol: float = 1e-3) -> bool:
    """Return True when both axes are uniform AND have the same pixel size."""
    if not (_is_uniform(x_axis) and _is_uniform(y_axis)):
        return False
    dx = (x_axis[-1] - x_axis[0]) / (len(x_axis) - 1) if len(x_axis) > 1 else 1.0
    dy = (y_axis[-1] - y_axis[0]) / (len(y_axis) - 1) if len(y_axis) > 1 else 1.0
    return bool(np.isclose(abs(dx), abs(dy), rtol=rtol))


class Viewer2D(anywidget.AnyWidget):
    """2-D image viewer that works directly with numpy arrays."""

    # ------------------------------------------------------------------ traits
    image_bytes = traitlets.Bytes(b"").tag(sync=True)
    image_width = traitlets.Int(256).tag(sync=True)
    image_height = traitlets.Int(256).tag(sync=True)

    viewer_width = traitlets.Int(256).tag(sync=True)
    viewer_height = traitlets.Int(256).tag(sync=True)

    x_axis_json = traitlets.Unicode("[]").tag(sync=True)
    y_axis_json = traitlets.Unicode("[]").tag(sync=True)
    units = traitlets.Unicode("px").tag(sync=True)

    use_scalebar = traitlets.Bool(True).tag(sync=True)
    scale_x = traitlets.Float(1.0).tag(sync=True)
    scale_y = traitlets.Float(1.0).tag(sync=True)

    histogram_data = traitlets.Unicode('{"bins":[],"counts":[]}').tag(sync=True)
    hist_min = traitlets.Float(0.0).tag(sync=True)
    hist_max = traitlets.Float(255.0).tag(sync=True)
    histogram_visible = traitlets.Bool(True).tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    show_colorbar = traitlets.Bool(True).tag(sync=True)
    colorbar_width = traitlets.Int(20).tag(sync=True)
    histogram_width = traitlets.Int(120).tag(sync=True)
    gap = traitlets.Int(10).tag(sync=True)

    zoom = traitlets.Float(1.0).tag(sync=True)
    center_x = traitlets.Float(0.5).tag(sync=True)
    center_y = traitlets.Float(0.5).tag(sync=True)

    # Overlay widgets – JSON list of shape dicts synced to JS
    overlay_widgets = traitlets.Unicode("[]").tag(sync=True)

    # Marker overlay – mirrors Circles(Markers): list of marker-set dicts
    # Each dict: { offsets:[[x,y],...], sizes:[r,...], color, linewidth }
    markers_json = traitlets.Unicode("[]").tag(sync=True)

    # ------------------------------------------------------------------ JS
    _esm = r"""
    function render({ model, el }) {
      const dpr       = window.devicePixelRatio || 1;
      const AXIS_SIZE = 40;

      // ── DOM ────────────────────────────────────────────────────────────────
      const outerContainer = document.createElement('div');
      outerContainer.style.cssText = 'position:relative;display:inline-block;';

      const container = document.createElement('div');
      container.style.cssText =
        `display:flex;flex-direction:row;gap:${model.get('gap')}px;` +
        'background:#f5f5f5;padding:10px;border-radius:4px;position:relative;';

      const histWidth = model.get('histogram_width');

      // Canvas wrapper
      const canvasWrapper = document.createElement('div');
      canvasWrapper.style.cssText = 'position:relative;display:inline-block;';

      // Image canvas
      const imageCanvas = document.createElement('canvas');
      imageCanvas.tabIndex = 1;
      imageCanvas.style.cssText = 'outline:none;cursor:default;background:white;border:1px solid #ccc;border-radius:2px;';
      const imgCtx = imageCanvas.getContext('2d');
      imageCanvas.addEventListener('focus', () => { imageCanvas.style.boxShadow = '0 0 0 2px rgba(76,175,80,0.5)'; });
      imageCanvas.addEventListener('blur',  () => { imageCanvas.style.boxShadow = 'none'; });

      // Axis canvases
      const xAxisCanvas = document.createElement('canvas');
      xAxisCanvas.style.cssText = 'display:block;background:#f5f5f5;';
      const xCtx = xAxisCanvas.getContext('2d');

      const yAxisCanvas = document.createElement('canvas');
      yAxisCanvas.style.cssText = 'display:block;background:#f5f5f5;';
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

      // Zoom display
      const zoomDisplay = document.createElement('div');
      zoomDisplay.style.cssText =
        'position:absolute;top:10px;left:10px;padding:8px 12px;font-size:13px;' +
        'font-family:monospace;font-weight:bold;background:rgba(0,0,0,0.75);color:white;' +
        'border-radius:4px;pointer-events:none;display:none;';
      canvasWrapper.appendChild(zoomDisplay);

      // Histogram canvas
      const histCanvas = document.createElement('canvas');
      histCanvas.style.cssText =
        `background:white;border:1px solid #ccc;border-radius:2px;` +
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

      // ── syncCanvasSizes ────────────────────────────────────────────────────
      function syncCanvasSizes() {
        const useScalebar = model.get('use_scalebar');
        const w = model.get('viewer_width');
        const h = model.get('viewer_height');

        imageCanvas.width  = w * dpr;  imageCanvas.height = h * dpr;
        imageCanvas.style.width  = w + 'px';  imageCanvas.style.height = h + 'px';
        imgCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        imgCtx.imageSmoothingEnabled = false;

        overlayCanvas.width  = w * dpr;  overlayCanvas.height = h * dpr;
        overlayCanvas.style.width  = w + 'px';  overlayCanvas.style.height = h + 'px';
        ovCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

        markersCanvas.width  = w * dpr;  markersCanvas.height = h * dpr;
        markersCanvas.style.width  = w + 'px';  markersCanvas.style.height = h + 'px';
        mkCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

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

      // ── scale bar ──────────────────────────────────────────────────────────
      function drawScaleBar() {
        const w         = model.get('viewer_width');
        const imgW      = model.get('image_width');
        const scaleX    = model.get('scale_x');
        const units     = model.get('units');
        const zoom      = model.get('zoom');
        const usePixels = !scaleX || scaleX <= 0 || units === 'px';
        const targetPx  = w / 4;
        if (usePixels) {
          const ppsp = imgW / w;
          const nice = findNice(targetPx * ppsp / zoom);
          scaleBarLine.style.width  = (nice / ppsp * zoom) + 'px';
          scaleBarLabel.textContent = fmtVal(nice) + ' px';
        } else {
          const ppsp = imgW / w;
          const nice = findNice(targetPx * ppsp / zoom * scaleX);
          scaleBarLine.style.width  = (nice / scaleX / ppsp * zoom) + 'px';
          scaleBarLabel.textContent = fmtVal(nice) + ' ' + units;
        }
      }

      // ── axis rulers ────────────────────────────────────────────────────────
      function drawAxes() {
        const xArr  = JSON.parse(model.get('x_axis_json'));
        const yArr  = JSON.parse(model.get('y_axis_json'));
        const units = model.get('units');
        const w     = model.get('viewer_width');
        const h     = model.get('viewer_height');

        xCtx.clearRect(0, 0, w, AXIS_SIZE);
        xCtx.fillStyle = '#f5f5f5';
        xCtx.fillRect(0, 0, w, AXIS_SIZE);
        if (xArr.length >= 2) {
          const xMin = xArr[0], xMax = xArr[xArr.length - 1], range = xMax - xMin || 1;
          const step = findNice(range / Math.max(3, Math.floor(w / 60)));
          xCtx.strokeStyle = '#555'; xCtx.lineWidth = 1;
          xCtx.fillStyle = '#333'; xCtx.font = '10px sans-serif'; xCtx.textAlign = 'center';
          xCtx.beginPath(); xCtx.moveTo(0, 0); xCtx.lineTo(w, 0); xCtx.stroke();
          for (let v = Math.ceil(xMin / step) * step; v <= xMax + step * 0.01; v += step) {
            const px = (v - xMin) / range * w;
            xCtx.beginPath(); xCtx.moveTo(px, 0); xCtx.lineTo(px, 6); xCtx.stroke();
            xCtx.fillText(fmtVal(v), px, 18);
          }
          xCtx.textAlign = 'right'; xCtx.fillStyle = '#777';
          xCtx.fillText(units, w - 2, AXIS_SIZE - 4);
        }

        yCtx.clearRect(0, 0, AXIS_SIZE, h);
        yCtx.fillStyle = '#f5f5f5';
        yCtx.fillRect(0, 0, AXIS_SIZE, h);
        if (yArr.length >= 2) {
          const yMin = yArr[0], yMax = yArr[yArr.length - 1], range = yMax - yMin || 1;
          const step = findNice(range / Math.max(3, Math.floor(h / 60)));
          yCtx.strokeStyle = '#555'; yCtx.lineWidth = 1;
          yCtx.fillStyle = '#333'; yCtx.font = '10px sans-serif'; yCtx.textAlign = 'right';
          yCtx.beginPath(); yCtx.moveTo(AXIS_SIZE, 0); yCtx.lineTo(AXIS_SIZE, h); yCtx.stroke();
          for (let v = Math.ceil(yMin / step) * step; v <= yMax + step * 0.01; v += step) {
            const py = (v - yMin) / range * h;
            yCtx.beginPath(); yCtx.moveTo(AXIS_SIZE, py); yCtx.lineTo(AXIS_SIZE - 6, py); yCtx.stroke();
            yCtx.fillText(fmtVal(v), AXIS_SIZE - 8, py + 4);
          }
        }
      }

      // ── draw image ─────────────────────────────────────────────────────────
      function drawImage() {
        const raw = model.get('image_bytes');
        let bytes;
        if (raw instanceof Uint8Array)                    bytes = raw;
        else if (raw instanceof ArrayBuffer)              bytes = new Uint8Array(raw);
        else if (raw && raw.buffer instanceof ArrayBuffer) bytes = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
        else                                              bytes = new Uint8Array(0);

        const iw = model.get('image_width'),  ih = model.get('image_height');
        const cw = parseInt(imageCanvas.style.width)  || model.get('viewer_width');
        const ch = parseInt(imageCanvas.style.height) || model.get('viewer_height');

        imgCtx.clearRect(0, 0, cw, ch);
        if (bytes.length === 0) return;

        imgCtx.imageSmoothingEnabled = false;

        const imageData = imgCtx.createImageData(iw, ih);
        for (let i = 0; i < bytes.length; i++) {
          const g = bytes[i];
          imageData.data[i * 4]     = g;
          imageData.data[i * 4 + 1] = g;
          imageData.data[i * 4 + 2] = g;
          imageData.data[i * 4 + 3] = 255;
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
          imgCtx.fillStyle = '#ffffff';
          imgCtx.fillRect(0, 0, cw, ch);
          imgCtx.drawImage(tmp, 0, 0, iw, ih, (cw - dstW) / 2, (ch - dstH) / 2, dstW, dstH);
        }

        if (model.get('use_scalebar')) drawScaleBar(); else drawAxes();
        drawOverlay();
        drawMarkers();
      }

      // ── histogram ──────────────────────────────────────────────────────────
      function drawHistogram() {
        if (!model.get('histogram_visible')) return;
        const histData = JSON.parse(model.get('histogram_data'));
        const counts = histData.counts;
        const w = parseInt(histCanvas.style.width)  || histWidth;
        const h = parseInt(histCanvas.style.height) || model.get('viewer_height');
        const useLog = model.get('log_scale');
        const showCB = model.get('show_colorbar');
        const cbW    = model.get('colorbar_width');

        histCtx.clearRect(0, 0, w, h);
        if (!counts || counts.length === 0) return;

        let proc = counts.slice();
        if (useLog) proc = counts.map(c => c > 0 ? Math.log10(c + 1) : 0);
        const maxC   = Math.max(...proc, 1);
        const chartX = showCB ? cbW + 5 : 0;
        const chartW = w - chartX;
        const barH   = h / counts.length;

        histCtx.fillStyle = '#4CAF50'; histCtx.strokeStyle = '#2E7D32'; histCtx.lineWidth = 0.5;
        for (let i = 0; i < counts.length; i++) {
          const idx = counts.length - 1 - i;
          const bw  = (proc[idx] / maxC) * chartW;
          const y   = i * barH;
          if (bw > 0) {
            histCtx.fillRect(chartX, y, bw, barH);
            if (barH > 2) histCtx.strokeRect(chartX, y, bw, barH);
          }
        }

        if (showCB) {
          const g = histCtx.createLinearGradient(0, 0, 0, h);
          for (let i = 0; i <= 10; i++) {
            const v = Math.round((1 - i / 10) * 255);
            g.addColorStop(i / 10, `rgb(${v},${v},${v})`);
          }
          histCtx.fillStyle = g; histCtx.fillRect(0, 0, cbW, h);
          histCtx.strokeStyle = '#666'; histCtx.lineWidth = 1; histCtx.strokeRect(0, 0, cbW, h);
        }

        histCtx.fillStyle = '#666'; histCtx.font = '10px monospace'; histCtx.textAlign = 'left';
        histCtx.fillText(model.get('hist_max').toFixed(0), chartX + 2, 12);
        histCtx.fillText(model.get('hist_min').toFixed(0), chartX + 2, h - 3);
      }

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
            _drawHandle(ccx + cr, ccy, w.color);
          } else if (w.type === 'rectangle') {
            const [rx, ry] = _imgToCanvas(w.x, w.y);
            const rw = w.w * scale, rh = w.h * scale;
            ovCtx.strokeRect(rx, ry, rw, rh);
            _drawHandle(rx,      ry,      w.color);
            _drawHandle(rx + rw, ry,      w.color);
            _drawHandle(rx,      ry + rh, w.color);
            _drawHandle(rx + rw, ry + rh, w.color);
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
        const [rx, ry] = _imgToCanvas(w.x, w.y);
        const rw = w.w * scale, rh = w.h * scale;
        return [{ x: rx, y: ry }, { x: rx + rw, y: ry },
                { x: rx, y: ry + rh }, { x: rx + rw, y: ry + rh }];
      }

      function _hitTest(ex, ey) {
        const widgets = JSON.parse(model.get('overlay_widgets'));
        const rect    = imageCanvas.getBoundingClientRect();
        const mx = ex - rect.left, my = ey - rect.top;
        // Handles first
        for (let i = widgets.length - 1; i >= 0; i--) {
          const hs = _handles(widgets[i]);
          for (let hi = 0; hi < hs.length; hi++) {
            const dx = mx - hs[hi].x, dy = my - hs[hi].y;
            if (Math.sqrt(dx * dx + dy * dy) <= HANDLE_R)
              return { idx: i, mode: 'resize', hi };
          }
        }
        // Body
        const scale = _imgScale();
        for (let i = widgets.length - 1; i >= 0; i--) {
          const w = widgets[i];
          if (w.type === 'circle') {
            const [ccx, ccy] = _imgToCanvas(w.cx, w.cy);
            const dx = mx - ccx, dy = my - ccy;
            if (Math.sqrt(dx * dx + dy * dy) <= w.r * scale + 4)
              return { idx: i, mode: 'move', hi: -1 };
          } else {
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
          } else {
            widgets[ovDrag.idx].x = s.x + dx;
            widgets[ovDrag.idx].y = s.y + dy;
          }
        } else {
          if (s.type === 'circle') {
            widgets[ovDrag.idx].r = Math.max(2,
              Math.sqrt((ix - s.cx) ** 2 + (iy - s.cy) ** 2));
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

      // hover cursor
      imageCanvas.addEventListener('mousemove', (e) => {
        if (ovDrag || isPanning || isResizing) return;
        const hit = _hitTest(e.clientX, e.clientY);
        imageCanvas.style.cursor = hit
          ? (hit.mode === 'resize' ? 'nwse-resize' : 'move')
          : 'default';
      });

      // ── resize ─────────────────────────────────────────────────────────────
      resizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX; startY = e.clientY;
        startWidth  = parseInt(imageCanvas.style.width);
        startHeight = parseInt(imageCanvas.style.height);
        sizeLabel.style.display = 'block';
        e.preventDefault();
      });

      document.addEventListener('mousemove', (e) => {
        if (isResizing) {
          const dX = e.clientX - startX, dY = e.clientY - startY;
          const imgW = model.get('image_width'), imgH = model.get('image_height');
          let nw, nh;
          if (imgW > 0 && imgH > 0) {
            const ar = imgW / imgH, avg = (dX + dY) / 2;
            nw = Math.max(128, startWidth + avg);
            nh = Math.max(128, nw / ar);
            if (nh < 128) { nh = 128; nw = nh * ar; }
          } else {
            nw = Math.max(128, startWidth  + dX);
            nh = Math.max(128, startHeight + dY);
          }
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
            xAxisCanvas.style.width = nw + 'px'; xAxisCanvas.width = nw * dpr;
            yAxisCanvas.style.height = nh + 'px'; yAxisCanvas.height = nh * dpr;
            xCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
            yCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
          }
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
          model.set('viewer_width',  Math.round(parseInt(imageCanvas.style.width)));
          model.set('viewer_height', Math.round(parseInt(imageCanvas.style.height)));
          model.save_changes();
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
        zoomDisplay.textContent = newZ.toFixed(2) + 'x';
        zoomDisplay.style.display = newZ !== 1.0 ? 'block' : 'none';
      });

      // ── keyboard shortcuts ─────────────────────────────────────────────────
      imageCanvas.addEventListener('keydown', (e) => {
        if (e.key === 'r' || e.key === 'R') {
          model.set('zoom', 1.0); model.set('center_x', 0.5); model.set('center_y', 0.5);
          model.save_changes(); zoomDisplay.style.display = 'none'; e.preventDefault();
        } else if (e.key === 'h' || e.key === 'H') {
          model.set('histogram_visible', !model.get('histogram_visible'));
          model.save_changes(); e.preventDefault();
        }
      });

      // ── model listeners ────────────────────────────────────────────────────
      model.on('change:image_bytes',       drawImage);
      model.on('change:histogram_data',    drawHistogram);
      model.on('change:log_scale',         drawHistogram);
      model.on('change:show_colorbar',     drawHistogram);
      model.on('change:x_axis_json',       () => { if (!model.get('use_scalebar')) drawAxes(); });
      model.on('change:y_axis_json',       () => { if (!model.get('use_scalebar')) drawAxes(); });
      model.on('change:use_scalebar',      () => { syncCanvasSizes(); drawImage(); drawHistogram(); });
      model.on('change:histogram_visible', () => {
        histCanvas.style.display = model.get('histogram_visible') ? 'block' : 'none';
        container.style.gap = model.get('histogram_visible') ? model.get('gap') + 'px' : '0px';
        drawHistogram();
      });
      model.on('change:viewer_width',  () => { syncCanvasSizes(); drawImage(); drawScaleBar(); });
      model.on('change:viewer_height', () => { syncCanvasSizes(); drawImage(); drawHistogram(); });
      model.on('change:zoom', () => {
        const z = model.get('zoom');
        zoomDisplay.textContent = z.toFixed(2) + 'x';
        zoomDisplay.style.display = z !== 1.0 ? 'block' : 'none';
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
    def __init__(self, data: np.ndarray, x_axis=None, y_axis=None, units: str = "px"):
        """Create a Viewer2D widget.

        Parameters
        ----------
        data : np.ndarray
            2-D grayscale image (H × W), or 3-D (H × W × 3/4).
        x_axis : array-like, optional
            Physical coordinates for each column.
        y_axis : array-like, optional
            Physical coordinates for each row.
        units : str, optional
            Physical unit label (default ``'px'``).
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

    # ------------------------------------------------------------------
    def _to_png_bytes(self) -> bytes:
        """Render current image as PNG (PyCharm static fallback)."""
        from PIL import Image as _PILImage

        arr = np.frombuffer(self.image_bytes, dtype=np.uint8).reshape(
            self.image_height, self.image_width
        )
        img = _PILImage.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _repr_mimebundle_(self, **kwargs):
        """Return anywidget bundle + PNG fallback for PyCharm."""
        bundle = super()._repr_mimebundle_(**kwargs)
        if bundle is None:
            bundle = {}
        try:
            bundle["image/png"] = base64.b64encode(self._to_png_bytes()).decode("ascii")
        except Exception:
            pass
        return bundle

    # ------------------------------------------------------------------
    def update(
        self, data: np.ndarray, x_axis=None, y_axis=None, units: str | None = None
    ):
        """Update the viewer with new data."""
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

        with self.hold_trait_notifications():
            self.image_bytes = img_u8.tobytes()
            self.image_width = w
            self.image_height = h
            self.use_scalebar = equal
            self.scale_x = float(abs(dx))
            self.scale_y = float(abs(dx))
            self.x_axis_json = json.dumps(x_axis.tolist())
            self.y_axis_json = json.dumps(y_axis.tolist())
            self.histogram_data = histogram_json
            self.hist_min = float(vmin)
            self.hist_max = float(vmax)
            if units is not None:
                self.units = units

    # ------------------------------------------------------------------
    def add_widget(self, kind: str, color: str = "#00e5ff", **kwargs) -> str:
        """Add a moveable, resizable overlay widget to the viewer.

        Parameters
        ----------
        kind : ``'circle'`` or ``'rectangle'``
        color : str, optional
            CSS colour (default cyan ``'#00e5ff'``).
        **kwargs
            circle     – ``cx``, ``cy`` (image px centre), ``r`` (radius).
            rectangle  – ``x``, ``y`` (top-left, image px), ``w``, ``h``.

        Returns
        -------
        str  Widget ID (pass to :meth:`remove_widget` / :meth:`get_widget`).
        """
        import uuid

        kind = kind.lower()
        if kind not in ("circle", "rectangle"):
            raise ValueError(f"kind must be 'circle' or 'rectangle', got {kind!r}")
        iw, ih = self.image_width, self.image_height
        wid = str(uuid.uuid4())[:8]
        if kind == "circle":
            entry = {
                "id": wid,
                "type": "circle",
                "cx": float(kwargs.get("cx", iw / 2)),
                "cy": float(kwargs.get("cy", ih / 2)),
                "r": float(kwargs.get("r", iw * 0.1)),
                "color": color,
            }
        else:
            entry = {
                "id": wid,
                "type": "rectangle",
                "x": float(kwargs.get("x", iw * 0.25)),
                "y": float(kwargs.get("y", ih * 0.25)),
                "w": float(kwargs.get("w", iw * 0.5)),
                "h": float(kwargs.get("h", ih * 0.5)),
                "color": color,
            }
        widgets = json.loads(self.overlay_widgets)
        widgets.append(entry)
        self.overlay_widgets = json.dumps(widgets)
        return wid

    def remove_widget(self, wid: str) -> None:
        """Remove overlay widget by ID."""
        widgets = [w for w in json.loads(self.overlay_widgets) if w["id"] != wid]
        self.overlay_widgets = json.dumps(widgets)

    def clear_widgets(self) -> None:
        """Remove all overlay widgets."""
        self.overlay_widgets = "[]"

    def get_widget(self, wid: str) -> dict:
        """Return current state of a widget as a dict (position/size in image px).

        Raises ``KeyError`` if *wid* does not exist.
        """
        for w in json.loads(self.overlay_widgets):
            if w["id"] == wid:
                return dict(w)
        raise KeyError(f"No overlay widget with id {wid!r}")

    # ================================================================== markers
    # Low-level helper
    # -----------------------------------------------------------------
    def _push_markers(self, ms: dict, replace: bool) -> None:
        """Append or replace the markers_json list."""
        if replace:
            self.markers_json = json.dumps([ms])
        else:
            existing = json.loads(self.markers_json)
            existing.append(ms)
            self.markers_json = json.dumps(existing)

    @staticmethod
    def _broadcast_1d(arr, n: int, name: str) -> list:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 0:
            return np.full(n, float(arr)).tolist()
        if arr.ndim != 1 or len(arr) != n:
            raise ValueError(f"'{name}' must be a scalar or 1-D array of length {n}")
        return arr.tolist()

    @staticmethod
    def _check_offsets(offsets) -> np.ndarray:
        offsets = np.asarray(offsets, dtype=float)
        if offsets.ndim == 1 and offsets.shape[0] == 2:
            offsets = offsets[np.newaxis, :]
        if offsets.ndim != 2 or offsets.shape[1] != 2:
            raise ValueError("offsets must be shape (N, 2)")
        return offsets

    # -----------------------------------------------------------------
    def set_circles(self, offsets, sizes, color="#ff0000", linewidth=1.5) -> None:
        """Set circle markers (replaces all existing markers).

        Mirrors :class:`~hyperspy.drawing._markers.circles.Circles`.

        Parameters
        ----------
        offsets : array-like (N, 2)
            ``[[x, y], ...]`` centre positions in image-pixel space.
        sizes : array-like or scalar
            Radius of each circle in image-pixel units.
        color : str, optional  CSS colour (default ``'#ff0000'``).
        linewidth : float, optional  Stroke width in canvas pixels.
        """
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms = {"type": "circles", "offsets": offsets.tolist(),
              "sizes": self._broadcast_1d(sizes, n, "sizes"),
              "color": color, "linewidth": linewidth}
        self._push_markers(ms, replace=True)

    def add_circles(self, offsets, sizes, color="#ff0000", linewidth=1.5) -> None:
        """Add circle markers on top of existing markers.  Same parameters as
        :meth:`set_circles`."""
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms = {"type": "circles", "offsets": offsets.tolist(),
              "sizes": self._broadcast_1d(sizes, n, "sizes"),
              "color": color, "linewidth": linewidth}
        self._push_markers(ms, replace=False)

    # -----------------------------------------------------------------
    def set_arrows(self, offsets, U, V, color="#ff0000", linewidth=1.5) -> None:
        """Set arrow markers (replaces all existing markers).

        Mirrors :class:`~hyperspy.drawing._markers.arrows.Arrows`.

        Parameters
        ----------
        offsets : array-like (N, 2)   Arrow tail positions ``[[x, y], ...]``.
        U : array-like or scalar      Horizontal component (image-pixel units).
        V : array-like or scalar      Vertical component (image-pixel units).
        color : str, optional
        linewidth : float, optional
        """
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms = {"type": "arrows", "offsets": offsets.tolist(),
              "U": self._broadcast_1d(U, n, "U"),
              "V": self._broadcast_1d(V, n, "V"),
              "color": color, "linewidth": linewidth}
        self._push_markers(ms, replace=True)

    def add_arrows(self, offsets, U, V, color="#ff0000", linewidth=1.5) -> None:
        """Add arrow markers on top of existing markers.  Same parameters as
        :meth:`set_arrows`."""
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms = {"type": "arrows", "offsets": offsets.tolist(),
              "U": self._broadcast_1d(U, n, "U"),
              "V": self._broadcast_1d(V, n, "V"),
              "color": color, "linewidth": linewidth}
        self._push_markers(ms, replace=False)

    # -----------------------------------------------------------------
    def set_ellipses(self, offsets, widths, heights, angles=0,
                     color="#ff0000", linewidth=1.5) -> None:
        """Set ellipse markers (replaces all existing markers).

        Mirrors :class:`~hyperspy.drawing._markers.ellipses.Ellipses`.

        Parameters
        ----------
        offsets : array-like (N, 2)   Centre positions ``[[x, y], ...]``.
        widths : array-like or scalar  Full width in image-pixel units.
        heights : array-like or scalar Full height in image-pixel units.
        angles : array-like or scalar  Rotation angle in degrees (default 0).
        color : str, optional
        linewidth : float, optional
        """
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms = {"type": "ellipses", "offsets": offsets.tolist(),
              "widths":  self._broadcast_1d(widths,  n, "widths"),
              "heights": self._broadcast_1d(heights, n, "heights"),
              "angles":  self._broadcast_1d(angles,  n, "angles"),
              "color": color, "linewidth": linewidth}
        self._push_markers(ms, replace=True)

    def add_ellipses(self, offsets, widths, heights, angles=0,
                     color="#ff0000", linewidth=1.5) -> None:
        """Add ellipse markers on top of existing markers.  Same parameters as
        :meth:`set_ellipses`."""
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms = {"type": "ellipses", "offsets": offsets.tolist(),
              "widths":  self._broadcast_1d(widths,  n, "widths"),
              "heights": self._broadcast_1d(heights, n, "heights"),
              "angles":  self._broadcast_1d(angles,  n, "angles"),
              "color": color, "linewidth": linewidth}
        self._push_markers(ms, replace=False)

    # -----------------------------------------------------------------
    def set_lines(self, segments, color="#ff0000", linewidth=1.5) -> None:
        """Set line-segment markers (replaces all existing markers).

        Mirrors :class:`~hyperspy.drawing._markers.lines.Lines`.

        Parameters
        ----------
        segments : array-like (N, 2, 2)
            ``[[[x1,y1],[x2,y2]], ...]`` in image-pixel space.
        color : str, optional
        linewidth : float, optional
        """
        segments = np.asarray(segments, dtype=float)
        if segments.ndim == 2 and segments.shape == (2, 2):
            segments = segments[np.newaxis]           # single segment
        if segments.ndim != 3 or segments.shape[1:] != (2, 2):
            raise ValueError("segments must be shape (N, 2, 2)")
        ms = {"type": "lines", "segments": segments.tolist(),
              "color": color, "linewidth": linewidth}
        self._push_markers(ms, replace=True)

    def add_lines(self, segments, color="#ff0000", linewidth=1.5) -> None:
        """Add line-segment markers on top of existing markers.  Same
        parameters as :meth:`set_lines`."""
        segments = np.asarray(segments, dtype=float)
        if segments.ndim == 2 and segments.shape == (2, 2):
            segments = segments[np.newaxis]
        if segments.ndim != 3 or segments.shape[1:] != (2, 2):
            raise ValueError("segments must be shape (N, 2, 2)")
        ms = {"type": "lines", "segments": segments.tolist(),
              "color": color, "linewidth": linewidth}
        self._push_markers(ms, replace=False)

    # -----------------------------------------------------------------
    def set_rectangles(self, offsets, widths, heights, angles=0,
                       color="#ff0000", linewidth=1.5) -> None:
        """Set rectangle markers (replaces all existing markers).

        Parameters
        ----------
        offsets : array-like (N, 2)    Centre positions ``[[x, y], ...]``.
        widths : array-like or scalar  Width in image-pixel units.
        heights : array-like or scalar Height in image-pixel units.
        angles : array-like or scalar  Rotation in degrees (default 0).
        color : str, optional
        linewidth : float, optional
        """
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms = {"type": "rectangles", "offsets": offsets.tolist(),
              "widths":  self._broadcast_1d(widths,  n, "widths"),
              "heights": self._broadcast_1d(heights, n, "heights"),
              "angles":  self._broadcast_1d(angles,  n, "angles"),
              "color": color, "linewidth": linewidth}
        self._push_markers(ms, replace=True)

    def add_rectangles(self, offsets, widths, heights, angles=0,
                       color="#ff0000", linewidth=1.5) -> None:
        """Add rectangle markers on top of existing markers.  Same parameters
        as :meth:`set_rectangles`."""
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms = {"type": "rectangles", "offsets": offsets.tolist(),
              "widths":  self._broadcast_1d(widths,  n, "widths"),
              "heights": self._broadcast_1d(heights, n, "heights"),
              "angles":  self._broadcast_1d(angles,  n, "angles"),
              "color": color, "linewidth": linewidth}
        self._push_markers(ms, replace=False)

    # -----------------------------------------------------------------
    def set_squares(self, offsets, widths, angles=0,
                    color="#ff0000", linewidth=1.5) -> None:
        """Set square markers (replaces all existing markers).

        Mirrors :class:`~hyperspy.drawing._markers.squares.Squares`.

        Parameters
        ----------
        offsets : array-like (N, 2)    Centre positions ``[[x, y], ...]``.
        widths : array-like or scalar  Side length in image-pixel units.
        angles : array-like or scalar  Rotation in degrees (default 0).
        color : str, optional
        linewidth : float, optional
        """
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms = {"type": "squares", "offsets": offsets.tolist(),
              "widths": self._broadcast_1d(widths, n, "widths"),
              "angles": self._broadcast_1d(angles, n, "angles"),
              "color": color, "linewidth": linewidth}
        self._push_markers(ms, replace=True)

    def add_squares(self, offsets, widths, angles=0,
                    color="#ff0000", linewidth=1.5) -> None:
        """Add square markers on top of existing markers.  Same parameters as
        :meth:`set_squares`."""
        offsets = self._check_offsets(offsets)
        n = len(offsets)
        ms = {"type": "squares", "offsets": offsets.tolist(),
              "widths": self._broadcast_1d(widths, n, "widths"),
              "angles": self._broadcast_1d(angles, n, "angles"),
              "color": color, "linewidth": linewidth}
        self._push_markers(ms, replace=False)

    # -----------------------------------------------------------------
    def set_texts(self, offsets, texts, color="#ff0000", fontsize=12) -> None:
        """Set text markers (replaces all existing markers).

        Mirrors :class:`~hyperspy.drawing._markers.texts.Texts`.

        Parameters
        ----------
        offsets : array-like (N, 2)   Anchor positions ``[[x, y], ...]``.
        texts : list of str           One label per position.
        color : str, optional
        fontsize : int, optional      Font size in canvas pixels (default 12).
        """
        offsets = self._check_offsets(offsets)
        texts = list(texts)
        if len(texts) != len(offsets):
            raise ValueError("len(texts) must equal len(offsets)")
        ms = {"type": "texts", "offsets": offsets.tolist(),
              "texts": texts, "color": color, "fontsize": fontsize}
        self._push_markers(ms, replace=True)

    def add_texts(self, offsets, texts, color="#ff0000", fontsize=12) -> None:
        """Add text markers on top of existing markers.  Same parameters as
        :meth:`set_texts`."""
        offsets = self._check_offsets(offsets)
        texts = list(texts)
        if len(texts) != len(offsets):
            raise ValueError("len(texts) must equal len(offsets)")
        ms = {"type": "texts", "offsets": offsets.tolist(),
              "texts": texts, "color": color, "fontsize": fontsize}
        self._push_markers(ms, replace=False)

    # -----------------------------------------------------------------
    # Legacy aliases kept for back-compat with earlier set_markers call
    def set_markers(self, offsets, sizes, color="#ff0000", linewidth=1.5) -> None:
        """Alias for :meth:`set_circles` (kept for backwards compatibility)."""
        self.set_circles(offsets, sizes, color=color, linewidth=linewidth)

    def add_markers(self, offsets, sizes, color="#ff0000", linewidth=1.5) -> None:
        """Alias for :meth:`add_circles` (kept for backwards compatibility)."""
        self.add_circles(offsets, sizes, color=color, linewidth=linewidth)

    def clear_markers(self) -> None:
        """Remove all marker overlays."""
        self.markers_json = "[]"

