import os
import io
import time
import json
import csv
import signal
import threading
import queue
from datetime import datetime

from flask import Flask, Response, jsonify, request, render_template_string, redirect, url_for

import numpy as np
from PIL import Image

import pypuclib
from pypuclib import CameraFactory

# =========================
# Fixed capture settings
# =========================
TARGET_FPS = 500
TARGET_W   = 1246
TARGET_H   = 1080

# Preview frame rate (browser stream). Keep this modest.
PREVIEW_FPS = 30

# Save directory (no file dialog in web UI)
SAVE_DIR = os.path.abspath("./captures")

# File modes
FILE_CSV = 0
FILE_BINARY = 1

# =========================
# Recording writer thread
# =========================
class RecordingWriter(threading.Thread):
    """
    Writes frames in background without blocking the camera callback.
    - CSV logs sequence numbers and gaps
    - BINARY writes a single .npy stream (np.save per frame) + one .json metadata
    """
    def __init__(self, basepath: str, filetype: int, cam, max_queue=4096, flush_interval=1.0):
        super().__init__(daemon=True)
        self.basepath = basepath
        self.filetype = filetype
        self.cam = cam
        self.q = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._last_seq = None
        self._last_flush = time.time()
        self.flush_interval = flush_interval
        os.makedirs(os.path.dirname(self.basepath), exist_ok=True)
        self._open_files()

    def _open_files(self):
        if self.filetype == FILE_CSV:
            self.csvf = open(self.basepath + ".csv", "w", newline="")
            self.csvw = csv.writer(self.csvf, lineterminator="\n")
            self.csvw.writerow(["SequenceNo", "diff", "drop"])
        elif self.filetype == FILE_BINARY:
            meta = {
                "framerate": self.cam.framerate(),
                "shutter": self.cam.shutter(),
                "width": self.cam.resolution().width,
                "height": self.cam.resolution().height,
                "quantization": self.cam.decoder().quantization()
            }
            with open(self.basepath + ".json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            self.bin = open(self.basepath + ".npy", "ab")
        else:
            raise ValueError("Unknown filetype")

    def enqueue(self, seq, arr):
        # prefer newest; if full, drop one
        try:
            self.q.put_nowait((seq, arr))
        except queue.Full:
            try:
                _ = self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait((seq, arr))
            except queue.Full:
                pass  # drop

    def run(self):
        try:
            while not self._stop.is_set() or not self.q.empty():
                try:
                    seq, arr = self.q.get(timeout=0.2)
                except queue.Empty:
                    self._maybe_flush()
                    continue

                if self.filetype == FILE_CSV:
                    if self._last_seq is None:
                        diff, drop = 0, ""
                    else:
                        diff = seq - self._last_seq
                        drop = "*" if diff > 1 else ""
                    self.csvw.writerow([seq, diff, drop])
                    self._last_seq = seq
                else:
                    np.save(self.bin, arr, allow_pickle=False)

                self._maybe_flush()
        finally:
            self._close_files()

    def _maybe_flush(self):
        now = time.time()
        if now - self._last_flush >= self.flush_interval:
            try:
                if self.filetype == FILE_CSV:
                    self.csvf.flush(); os.fsync(self.csvf.fileno())
                else:
                    self.bin.flush(); os.fsync(self.bin.fileno())
            except Exception:
                pass
            self._last_flush = now

    def _close_files(self):
        try:
            if self.filetype == FILE_CSV:
                self.csvf.flush(); self.csvf.close()
            else:
                self.bin.flush(); self.bin.close()
        except Exception:
            pass

    def stop(self):
        self._stop.set()


# =========================
# Camera manager
# =========================
class CameraManager:
    """
    Handles camera, DMA acquisition, latest-frame sharing, preview encode, and recording.
    """
    def __init__(self):
        self.cam = CameraFactory().create()
        self.decoder = self.cam.decoder()

        # Apply settings
        try:
            self.cam.setResolution(TARGET_W, TARGET_H)
        except Exception as e:
            print(f"[WARN] setResolution({TARGET_W},{TARGET_H}) failed: {e}")
        try:
            self.cam.setFramerateShutter(TARGET_FPS, TARGET_FPS)
        except Exception as e:
            print(f"[WARN] setFramerateShutter({TARGET_FPS}) failed: {e}")

        # latest XferData (for preview decode)
        self._latest_lock = threading.Lock()
        self._latest_xfer = None      # last XferData
        self._latest_seq = None

        # Preview JPEG bytes cached by preview thread
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg = None
        self._latest_jpeg_seq = None

        # Recording
        self.rec_writer = None
        self.rec_active = False
        self.rec_mode = FILE_BINARY

        # Threads
        self._running = True
        self.preview_thread = threading.Thread(target=self._preview_worker, daemon=True)

        # Start DMA and preview thread
        self.cam.beginXfer(self._xfer_callback)
        self.preview_thread.start()

    # Callback from SDK DMA thread
    def _xfer_callback(self, xferData):
        try:
            seq = xferData.sequenceNo()
            # Save latest reference for preview decode thread
            with self._latest_lock:
                self._latest_xfer = xferData
                self._latest_seq = seq
            # Recording (enqueue raw encoded data)
            if self.rec_active and self.rec_writer is not None:
                if self.rec_mode == FILE_BINARY:
                    self.rec_writer.enqueue(seq, xferData.data())
                else:
                    self.rec_writer.enqueue(seq, None)
        except Exception:
            # Do not break acquisition thread
            pass

    def _preview_worker(self):
        # Produce JPEG at PREVIEW_FPS from latest xfer
        dt = 1.0 / max(1, PREVIEW_FPS)
        while self._running:
            start = time.time()
            try:
                with self._latest_lock:
                    xfer = self._latest_xfer
                    seq  = self._latest_seq
                if xfer is not None:
                    # Decode on this thread
                    arr = self.decoder.decode(xfer)  # numpy
                    # Ensure fixed size 1:1
                    if arr.shape[1] != TARGET_W or arr.shape[0] != TARGET_H:
                        arr = np.asarray(Image.fromarray(arr).resize((TARGET_W, TARGET_H)))
                    # Convert to JPEG bytes
                    pil = Image.fromarray(arr)
                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG", quality=80)
                    jpeg_bytes = buf.getvalue()
                    with self._jpeg_lock:
                        self._latest_jpeg = jpeg_bytes
                        self._latest_jpeg_seq = seq
            except Exception as e:
                # Keep going
                pass
            # throttle to target preview fps
            elapsed = time.time() - start
            to_sleep = dt - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    def start_recording(self, mode: int):
        if self.rec_active:
            return False, "Already recording"
        base = self._make_basepath()
        self.rec_mode = mode
        self.rec_writer = RecordingWriter(base, mode, self.cam)
        self.rec_writer.start()
        self.rec_active = True
        return True, os.path.basename(base)

    def stop_recording(self):
        if not self.rec_active:
            return False, "Not recording"
        try:
            self.rec_writer.stop()
            self.rec_writer.join(timeout=5.0)
        except Exception:
            pass
        self.rec_writer = None
        self.rec_active = False
        return True, "Stopped"

    def get_latest_jpeg(self):
        with self._jpeg_lock:
            return self._latest_jpeg, self._latest_jpeg_seq

    def get_status(self):
        return {
            "fps": TARGET_FPS,
            "resolution": [TARGET_W, TARGET_H],
            "recording": self.rec_active,
            "mode": "binary" if self.rec_mode == FILE_BINARY else "csv"
        }

    def _make_basepath(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"{ts}_{TARGET_FPS}fps_{TARGET_W}x{TARGET_H}_capture"
        return os.path.join(SAVE_DIR, name)

    def close(self):
        # Stop preview thread
        self._running = False
        try:
            self.preview_thread.join(timeout=2.0)
        except Exception:
            pass
        # Stop recording if any
        if self.rec_active:
            try:
                self.stop_recording()
            except Exception:
                pass
        # Stop DMA and close camera
        try:
            if self.cam.isXferring():
                self.cam.endXfer()
        except Exception:
            pass
        try:
            self.cam.close()
        except Exception:
            pass


# =========================
# Flask app
# =========================
app = Flask(__name__)
cam_mgr = CameraManager()

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Photron Live — 500 fps @ 1246×1080</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;background:#0f1115;color:#e7eaf0;margin:0}
    header{display:flex;justify-content:space-between;align-items:center;padding:14px 18px;background:#171923;border-bottom:1px solid #2d3142}
    .pill{opacity:.9}
    main{display:flex;gap:16px;padding:16px}
    .panel{background:#171923;border:1px solid #2d3142;border-radius:12px;padding:14px}
    .controls{min-width:280px}
    button{background:#2b6cb0;color:#fff;border:none;border-radius:10px;padding:10px 14px;font-weight:600;cursor:pointer}
    button.secondary{background:#2d3748}
    button.danger{background:#c53030}
    .row{display:flex;gap:10px;align-items:center;margin:10px 0}
    .badge{display:inline-block;background:#2d3748;color:#e7eaf0;border-radius:999px;padding:6px 10px;font-size:12px}
    img{border-radius:12px;border:1px solid #2d3142;max-width:100%}
    .status{font-size:14px;color:#cbd5e0}
    .radio{display:flex;gap:10px;margin:8px 0}
    .radio label{display:flex;gap:6px;align-items:center;cursor:pointer}
    footer{padding:10px 18px;color:#9aa4b2;border-top:1px solid #2d3142}
  </style>
</head>
<body>
  <header>
    <div><strong>Photron Live</strong> <span class="pill">— 500 fps @ 1246×1080</span></div>
    <div id="status" class="status">Loading status…</div>
  </header>
  <main>
    <div class="panel">
      <img id="stream" src="{{ url_for('stream') }}" alt="Live Stream">
    </div>
    <div class="panel controls">
      <div class="row">
        <span class="badge">Save mode</span>
      </div>
      <div class="radio">
        <label><input type="radio" name="mode" value="binary" checked> Binary</label>
        <label><input type="radio" name="mode" value="csv"> CSV (seq log)</label>
      </div>
      <div class="row">
        <button id="startBtn">● Start Recording</button>
        <button id="stopBtn" class="secondary">■ Stop Recording</button>
      </div>
      <div class="row">
        <button id="exitBtn" class="danger">Close Camera & Exit</button>
      </div>
      <div class="row">
        <div id="msg" class="status"></div>
      </div>
      <div class="row">
        <span class="badge">Saves to: {{ save_dir }}</span>
      </div>
    </div>
  </main>
  <footer>
    Live stream is throttled to ~{{ preview_fps }} fps for the browser. Acquisition remains at 500 fps.
  </footer>

<script>
async function updateStatus(){
  try{
    const r = await fetch('/status'); const j = await r.json();
    document.getElementById('status').textContent = `Recording: ${j.recording ? 'ON' : 'OFF'} | Mode: ${j.mode} | ${j.resolution[0]}x${j.resolution[1]} @ ${j.fps} fps`;
  }catch(e){}
}
setInterval(updateStatus, 1000); updateStatus();

function getMode(){
  return document.querySelector('input[name="mode"]:checked').value === 'binary' ? 0 : 1;
}

document.getElementById('startBtn').onclick = async ()=>{
  const mode = getMode();
  const r = await fetch('/start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({mode})});
  const j = await r.json();
  document.getElementById('msg').textContent = j.message || '';
  updateStatus();
}

document.getElementById('stopBtn').onclick = async ()=>{
  const r = await fetch('/stop', {method:'POST'});
  const j = await r.json();
  document.getElementById('msg').textContent = j.message || '';
  updateStatus();
}

document.getElementById('exitBtn').onclick = async ()=>{
  await fetch('/exit', {method:'POST'});
  // give server a moment to shut down
  setTimeout(()=>{ window.close(); }, 500);
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML, save_dir=SAVE_DIR, preview_fps=PREVIEW_FPS)

@app.route("/status")
def status():
    return jsonify(cam_mgr.get_status())

@app.route("/start", methods=["POST"])
def start_recording():
    data = request.get_json(silent=True) or {}
    mode = int(data.get("mode", FILE_BINARY))
    ok, msg = cam_mgr.start_recording(mode)
    return jsonify({"ok": ok, "message": msg})

@app.route("/stop", methods=["POST"])
def stop_recording():
    ok, msg = cam_mgr.stop_recording()
    return jsonify({"ok": ok, "message": msg})

@app.route("/exit", methods=["POST"])
def exit_app():
    # Close camera and shutdown server
    def _shutdown():
        time.sleep(0.1)
        os._exit(0)
    try:
        cam_mgr.close()
    finally:
        threading.Thread(target=_shutdown, daemon=True).start()
    return jsonify({"ok": True, "message": "Exiting..."})

@app.route("/stream")
def stream():
    """
    MJPEG stream using the latest pre-encoded JPEG produced by preview thread.
    """
    def gen():
        boundary = b"--frame"
        while True:
            frame, seq = cam_mgr.get_latest_jpeg()
            if frame is None:
                # send a black frame placeholder to keep the connection alive
                img = Image.new("L", (TARGET_W, TARGET_H), color=0)
                buf = io.BytesIO(); img.save(buf, format="JPEG", quality=80)
                frame = buf.getvalue()
            yield (b"%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % (boundary, len(frame))) + frame + b"\r\n"
            time.sleep(1.0 / PREVIEW_FPS)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

def handle_sigterm(signum, frame):
    try:
        cam_mgr.close()
    except Exception:
        pass
    os._exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_sigterm)
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Run Flask dev server on localhost:5000
    app.run(host="127.0.0.1", port=5000, threaded=True)
