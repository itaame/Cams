import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading, queue, time, json, os, csv, signal
import numpy as np
from datetime import datetime

import pypuclib
from pypuclib import CameraFactory

# ----------------------------
# Fixed capture settings
# ----------------------------
TARGET_FPS = 500
TARGET_W   = 1246
TARGET_H   = 1080

# ----------------------------
# File modes
# ----------------------------
FILE_CSV = 0
FILE_BINARY = 1

# ----------------------------
# Background recording writer
# ----------------------------
class RecordingWriter(threading.Thread):
    def __init__(self, basepath: str, filetype: int, cam, max_queue=2048, flush_interval=1.0):
        super().__init__(daemon=True)
        self.basepath = basepath
        self.filetype = filetype
        self.cam = cam
        self.q = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._last_seq = None
        self._last_flush = time.time()
        self.flush_interval = flush_interval
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

    def enqueue(self, seq, arr):
        try:
            self.q.put_nowait((seq, arr))
        except queue.Full:
            try: self.q.get_nowait()
            except queue.Empty: pass
            try: self.q.put_nowait((seq, arr))
            except queue.Full: pass

    def run(self):
        try:
            while not self._stop.is_set() or not self.q.empty():
                try:
                    seq, arr = self.q.get(timeout=0.2)
                except queue.Empty:
                    self._flush()
                    continue

                if self.filetype == FILE_CSV:
                    diff, drop = (0, "") if self._last_seq is None else (seq - self._last_seq, "*" if seq - self._last_seq > 1 else "")
                    self.csvw.writerow([seq, diff, drop])
                    self._last_seq = seq
                else:
                    np.save(self.bin, arr, allow_pickle=False)

                self._flush()
        finally:
            self._close_files()

    def _flush(self):
        now = time.time()
        if now - self._last_flush >= self.flush_interval:
            try:
                if self.filetype == FILE_CSV:
                    self.csvf.flush(); os.fsync(self.csvf.fileno())
                else:
                    self.bin.flush(); os.fsync(self.bin.fileno())
            except Exception: pass
            self._last_flush = now

    def _close_files(self):
        try:
            if self.filetype == FILE_CSV: self.csvf.close()
            else: self.bin.close()
        except Exception: pass

    def stop(self): self._stop.set()

# ----------------------------
# Main App
# ----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Photron Live ({TARGET_FPS} fps, {TARGET_W}x{TARGET_H})")
        self.geometry(f"{TARGET_W+32}x{TARGET_H+120}")
        self.configure(bg="#1a1a1a")

        self.cam = CameraFactory().create()
        self.decoder = self.cam.decoder()

        self._apply_settings()

        self._latest_lock = threading.Lock()
        self._latest_data = None
        self._img = None

        self.rec_writer = None
        self.rec_active = False
        self.save_mode = tk.IntVar(value=FILE_BINARY)

        self._build_ui()

        self.cam.beginXfer(self._xfer_callback)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        signal.signal(signal.SIGINT, lambda *_: self._on_close())
        self.after(1, self._ui_update)

    def _apply_settings(self):
        try: self.cam.setResolution(TARGET_W, TARGET_H)
        except Exception as e: print(f"Resolution set failed: {e}")
        try: self.cam.setFramerateShutter(TARGET_FPS, TARGET_FPS)
        except Exception as e: print(f"FPS set failed: {e}")

    def _build_ui(self):
        toolbar = ttk.Frame(self); toolbar.pack(side="top", fill="x", padx=8, pady=8)
        ttk.Label(toolbar, text=f"Fixed: {TARGET_FPS} fps @ {TARGET_W}×{TARGET_H}").pack(side="left")
        ttk.Label(toolbar, text="Save as:").pack(side="left", padx=(12,6))
        ttk.Radiobutton(toolbar, text="Binary", value=FILE_BINARY, variable=self.save_mode).pack(side="left")
        ttk.Radiobutton(toolbar, text="CSV", value=FILE_CSV, variable=self.save_mode).pack(side="left")
        self._record_btn = ttk.Button(toolbar, text="● Start Recording", command=self._toggle_record)
        self._record_btn.pack(side="left", padx=12)
        ttk.Button(toolbar, text="Close Camera & Exit", command=self._on_close).pack(side="right")
        self._status = tk.StringVar(value="Streaming preview..."); ttk.Label(toolbar, textvariable=self._status).pack(side="right", padx=(0,12))
        self.canvas = tk.Canvas(self, bg="black", width=TARGET_W, height=TARGET_H, highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)

    def _xfer_callback(self, xferData):
        try:
            seq = xferData.sequenceNo()
            with self._latest_lock: self._latest_data = (seq, xferData)
            if self.rec_active and self.rec_writer:
                if self.rec_writer.filetype == FILE_BINARY: self.rec_writer.enqueue(seq, xferData.data())
                else: self.rec_writer.enqueue(seq, None)
        except Exception: pass

    def _ui_update(self):
        try:
            with self._latest_lock:
                item = self._latest_data; self._latest_data = None
            if item:
                seq, xfer = item
                arr = self.decoder.decode(xfer)
                self._draw_frame(arr, seq)
        except Exception as e: self._status.set(f"Preview error: {e}")
        self.after(1, self._ui_update)

    def _draw_frame(self, arr, seq):
        if arr.shape[1] != TARGET_W or arr.shape[0] != TARGET_H:
            arr = np.asarray(Image.fromarray(arr).resize((TARGET_W, TARGET_H)))
        img = Image.fromarray(arr)
        self._img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._img)
        self.canvas.create_text(8, 8, anchor="nw", text=f"SequenceNo: {seq}", fill="lime")

    def _toggle_record(self):
        if not self.rec_active:
            base = self._choose_save_base()
            if not base: return
            ftype = self.save_mode.get()
            self.rec_writer = RecordingWriter(base, ftype, self.cam); self.rec_writer.start()
            self.rec_active = True
            self._record_btn.config(text="■ Stop Recording")
            self._status.set(f"Recording → {os.path.basename(base)}")
        else:
            self._stop_recording()
            self._record_btn.config(text="● Start Recording")
            self._status.set("Recording stopped. Preview continues.")

    def _stop_recording(self):
        if self.rec_writer:
            self.rec_writer.stop(); self.rec_writer.join(timeout=5.0)
            self.rec_writer = None
        self.rec_active = False

    def _choose_save_base(self):
        dirsel = filedialog.askdirectory(title="Choose folder to save")
        if not dirsel: return ""
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return os.path.join(dirsel, f"{ts}_{TARGET_FPS}fps_{TARGET_W}x{TARGET_H}_capture")

    def _on_close(self, *_):
        try: self._stop_recording()
        except Exception: pass
        try:
            if self.cam.isXferring(): self.cam.endXfer()
            self.cam.close()
        except Exception: pass
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
