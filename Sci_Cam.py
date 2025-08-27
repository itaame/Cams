import os
import io
import time
import json
import signal
import threading
import queue
from datetime import datetime

from flask import Flask, Response, jsonify, request, render_template_string

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

# Browser preview fps (acquisition remains 500 fps)
PREVIEW_FPS = 30

# Save directory
SAVE_DIR = os.path.abspath("./captures")

# Binary-only
FILE_BINARY = 1


# =========================
# Recording writer thread
# =========================
class RecordingWriter(threading.Thread):
    """
    Writes frames in background without blocking the camera callback.
    Binary mode: one .json metadata + .npy stream (np.save per frame).
    """
    def __init__(self, basepath: str, cam, max_queue=4096, flush_interval=1.0):
        super().__init__(daemon=True)
        self.basepath = basepath
        self.cam = cam
        self.q = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._last_flush = time.time()
        self.flush_interval = flush_interval
        os.makedirs(os.path.dirname(self.basepath), exist_ok=True)
        self._open_files()

    def _open_files(self):
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

    def enqueue(self, seq, arr):
        # Prefer newest; if full, drop one
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
                pass

    def run(self):
        try:
            while not self._stop.is_set() or not self.q.empty():
                try:
                    seq, arr = self.q.get(timeout=0.2)
                except queue.Empty:
                    self._maybe_flush()
                    continue
                np.save(self.bin, arr, allow_pickle=False)
                self._maybe_flush()
        finally:
            self._close_files()

    def _maybe_flush(self):
        now = time.time()
        if now - self._last_flush >= self.flush_interval:
            try:
                self.bin.flush()
                os.fsync(self.bin.fileno())
            except Exception:
                pass
            self._last_flush = now

    def _close_files(self):
        try:
            self.bin.flush()
            self.bin.close()
        except Exception:
            pass

    def stop(self):
        self._stop.set()


# =========================
# Camera manager
# =========================
class CameraManager:
    """
    Handles camera, DMA acquisition, latest-frame sharing, preview JPEG, and recording.
    """
    def __init__(self):
        self.cam = CameraFactory().create()
        self.decoder = self.cam.decoder()

        # Apply fixed settings
        try:
            self.cam.setResolution(TARGET_W, TARGET_H)
        except Exception as e:
            print(f"[WARN] setResolution({TARGET_W},{TARGET_H}) failed: {e}")
        try:
            self.cam.setFramerateShutter(TARGET_FPS, TARGET_FPS)
        except Exception as e:
            print(f"[WARN] setFramerateShutter({TARGET_FPS}) failed: {e}")

        # Latest xfer for preview
        self._latest_lock = threading.Lock()
        self._latest_xfer = None
        self._latest_seq = None

        # Cached preview JPEG
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg = None

        # Recording
        self.rec_writer = None
        self.rec_active = False

        # Threads control
        self._running = True
        self.preview_thread = threading.Thread(target=self._preview_worker, daemon=True)

        # Start DMA + preview thread
        self.cam.beginXfer(self._xfer_callback)
        self.preview_thread.start()

    def _xfer_callback(self, xferData):
        try:
            seq = xferData.sequenceNo()
            with self._latest_lock:
                self._latest_xfer = xferData
                self._latest_seq  = seq
            if self.rec_active and self.rec_writer is not None:
                self.rec_writer.enqueue(seq, xferData.data())
        except Exception:
            pass

    def _preview_worker(self):
        dt = 1.0 / max(1, PREVIEW_FPS)
        while self._running:
            t0 = time.time()
            try:
                with self._latest_lock:
                    xfer = self._latest_xfer
                if xfer is not None:
                    arr = self.decoder.decode(xfer)
                    # Ensure fixed size
                    if arr.shape[1] != TARGET_W or arr.shape[0] != TARGET_H:
                        arr = np.asarray(Image.fromarray(arr).resize((TARGET_W, TARGET_H)))
                    pil = Image.fromarray(arr)
                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG", quality=80)
                    with self._jpeg_lock:
                        self._latest_jpeg = buf.getvalue()
            except Exception:
                pass
            # throttle
            sleep_t = dt - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def toggle_recording(self):
        if not self.rec_active:
            base = self._make_basepath()
            self.rec_writer = RecordingWriter(base, self.cam)
            self.rec_writer.start()
            self.rec_active = True
            return True, os.path.basename(base)
        else:
            try:
                self.rec_writer.stop()
                self.rec_writer.join(timeout=5.0)
            except Exception:
                pass
            self.rec_writer = None
            self.rec_active = False
            return False, "Stopped"

    def get_latest_jpeg(self):
        with self._jpeg_lock:
            return self._latest_jpeg

    def get_status(self):
        return {
            "recording": self.rec_active,
            "fps": TARGET_FPS,
            "resolution": [TARGET_W, TARGET_H]
        }

    def _make_basepath(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"{ts}_{TARGET_FPS}fps_{TARGET_W}x{TARGET_H}_capture"
        return os.path.join(SAVE_DIR, name)

    def close(self):
        self._running = False
        try:
            self.preview_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if self.rec_active:
                self.toggle_recording()  # this will stop it
        except Exception:
            pass
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
  <title>Photron Live — Minimal</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{--bg:#0f1115;--panel:#171923;--line:#2d3142;--text:#e7eaf0;--muted:#9aa4b2;--accent:#2b6cb0;--danger:#e53e3e}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif}
    .wrap{padding:12px}
    .row{display:flex;align-items:center;gap:12px}
    .btn{background:var(--accent);border:none;color:#fff;padding:8px 12px;border-radius:10px;font-weight:600;cursor:pointer}
    .btn[data-state="on"]{background:var(--danger)}
    .indicator{display:flex;align-items:center;gap:8px;color:var(--muted);font-weight:600}
    .dot{width:10px;height:10px;border-radius:50%;background:#666}
    .dot.on{background:var(--danger);box-shadow:0 0 10px rgba(229,62,62,.7)}
    .divider{height:8px}
    .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px;display:inline-block}
    img{display:block;max-width:100%;height:auto;border-radius:10px}
  </style>
</head>
<body>
  <div class="wrap">
    <!-- Top-left controls -->
    <div class="row">
      <button id="toggle" class="btn" data-state="off">● Start Recording</button>
      <div class="indicator"><div id="dot" class="dot"></div><span id="stateText">Idle</span></div>
    </div>
    <div class="divider"></div>
    <!-- Next line: live view, left-aligned -->
    <div class="panel">
      <img id="stream" src="{{ url_for('stream') }}" alt="Live">
    </div>
  </div>

<script>
async function getStatus(){
  try{
    const r = await fetch('/status'); const j = await r.json();
    const btn = document.getElementById('toggle');
    const dot = document.getElementById('dot');
    const st  = document.getElementById('stateText');
    if(j.recording){
      btn.textContent = "■ Stop Recording";
      btn.dataset.state = "on";
      dot.classList.add('on');
      st.textContent = "Recording";
    }else{
      btn.textContent = "● Start Recording";
      btn.dataset.state = "off";
      dot.classList.remove('on');
      st.textContent = "Idle";
    }
  }catch(e){}
}

document.getElementById('toggle').onclick = async ()=>{
  const r = await fetch('/toggle', {method:'POST'});
  await r.json();
  getStatus();
};

setInterval(getStatus, 1000);
getStatus();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/status")
def status():
    return jsonify(cam_mgr.get_status())

@app.route("/toggle", methods=["POST"])
def toggle_recording():
    on, msg = cam_mgr.toggle_recording()
    return jsonify({"recording": on, "message": msg})

@app.route("/stream")
def stream():
    """
    MJPEG stream using the latest pre-encoded JPEG produced by preview thread.
    """
    def gen():
        boundary = b"--frame"
        while True:
            frame = cam_mgr.get_latest_jpeg()
            if frame is None:
                # send black placeholder to keep connection alive
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
    app.run(host="127.0.0.1", port=5000, threaded=True)
