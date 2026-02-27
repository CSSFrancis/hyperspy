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
    image_bytes   = traitlets.Bytes(b"").tag(sync=True)
    image_width   = traitlets.Int(256).tag(sync=True)
    image_height  = traitlets.Int(256).tag(sync=True)

    viewer_width  = traitlets.Int(256).tag(sync=True)
    viewer_height = traitlets.Int(256).tag(sync=True)

    x_axis_json   = traitlets.Unicode("[]").tag(sync=True)
    y_axis_json   = traitlets.Unicode("[]").tag(sync=True)
    units         = traitlets.Unicode("px").tag(sync=True)

    use_scalebar  = traitlets.Bool(True).tag(sync=True)
    scale_x       = traitlets.Float(1.0).tag(sync=True)
    scale_y       = traitlets.Float(1.0).tag(sync=True)

    histogram_data    = traitlets.Unicode('{"bins":[],"counts":[]}').tag(sync=True)
    hist_min          = traitlets.Float(0.0).tag(sync=True)
    hist_max          = traitlets.Float(255.0).tag(sync=True)
    histogram_visible = traitlets.Bool(True).tag(sync=True)
    log_scale         = traitlets.Bool(False).tag(sync=True)
    show_colorbar     = traitlets.Bool(True).tag(sync=True)
    colorbar_width    = traitlets.Int(20).tag(sync=True)
    histogram_width   = traitlets.Int(120).tag(sync=True)
    gap               = traitlets.Int(10).tag(sync=True)

    zoom     = traitlets.Float(1.0).tag(sync=True)
    center_x = traitlets.Float(0.5).tag(sync=True)
    center_y = traitlets.Float(0.5).tag(sync=True)

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

      // ── pan ─────────────────────────────────────────────────────────────────
      imageCanvas.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        isPanning = true;
        panStartX = e.clientX; panStartY = e.clientY;
        panStartCenterX = model.get('center_x');
        panStartCenterY = model.get('center_y');
        imageCanvas.style.cursor = 'grabbing';
        imageCanvas.focus();
        e.preventDefault();
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

      // ── initial render ─────────────────────────────────────────────────────
      drawImage();
      drawHistogram();
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
        histogram_json = json.dumps({
            "bins":   bin_centers.tolist(),
            "counts": counts.tolist(),
        })

        aspect = w / h if h > 0 else 1.0
        if aspect >= 1.0:
            vw, vh = 256, max(64, int(256 / aspect))
        else:
            vh, vw = 256, max(64, int(256 * aspect))

        with self.hold_trait_notifications():
            self.image_bytes    = img_u8.tobytes()
            self.image_width    = w
            self.image_height   = h
            self.viewer_width   = vw
            self.viewer_height  = vh
            self.units          = units
            self.use_scalebar   = equal
            self.scale_x        = float(abs(dx))
            self.scale_y        = float(abs(dx))
            self.x_axis_json    = json.dumps(x_axis.tolist())
            self.y_axis_json    = json.dumps(y_axis.tolist())
            self.histogram_data = histogram_json
            self.hist_min       = float(vmin)
            self.hist_max       = float(vmax)

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
    def update(self, data: np.ndarray, x_axis=None, y_axis=None,
               units: str | None = None):
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
        histogram_json = json.dumps({
            "bins":   bin_centers.tolist(),
            "counts": counts.tolist(),
        })

        with self.hold_trait_notifications():
            self.image_bytes    = img_u8.tobytes()
            self.image_width    = w
            self.image_height   = h
            self.use_scalebar   = equal
            self.scale_x        = float(abs(dx))
            self.scale_y        = float(abs(dx))
            self.x_axis_json    = json.dumps(x_axis.tolist())
            self.y_axis_json    = json.dumps(y_axis.tolist())
            self.histogram_data = histogram_json
            self.hist_min       = float(vmin)
            self.hist_max       = float(vmax)
            if units is not None:
                self.units = units

