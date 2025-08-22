import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import datetime
import time
import numpy as np
from pypuclib import CameraFactory


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Merged Camera GUI")

        # --- Normal camera setup (IDs 1 and 2) ---
        self.normal_cams = [cv2.VideoCapture(1), cv2.VideoCapture(2)]
        self.normal_frames = [None, None]
        self.normal_locks = [threading.Lock(), threading.Lock()]
        for idx in range(2):
            t = threading.Thread(target=self._capture_normal, args=(idx,), daemon=True)
            t.start()

        # --- Scientific camera setup ---
        self.sci_cam = CameraFactory().create()
        self.sci_cam.setFramerate(500)
        self.sci_cam.setShutter(1/500)
        self.sci_decoder = self.sci_cam.decoder()
        self.sci_frame = None
        self.sci_lock = threading.Lock()
        self.sci_cam.beginXfer(self._sci_callback)

        # --- Recording control variables ---
        self.record_norm = False
        self.norm_writers = [None, None]
        self.record_sci = False
        self.sci_file = None

        # --- UI Elements ---
        self.normal_labels = [tk.Label(self), tk.Label(self)]
        self.sci_label = tk.Label(self)
        for lbl in self.normal_labels:
            lbl.pack()
        self.sci_label.pack()

        self.norm_btn = ttk.Button(self, text="Record Normal Cams", command=self.toggle_normal_record)
        self.norm_btn.pack(pady=5)

        self.sci_btn = ttk.Button(self, text="Record Scientific Cam", command=self.toggle_sci_record)
        self.sci_btn.pack(pady=5)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(10, self._update_gui)

    # --- Normal camera functions ---
    def _capture_normal(self, idx):
        cap = self.normal_cams[idx]
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            with self.normal_locks[idx]:
                self.normal_frames[idx] = frame.copy()
            if self.record_norm and self.norm_writers[idx]:
                self.norm_writers[idx].write(frame)
            if fps:
                time.sleep(1 / fps)

    def toggle_normal_record(self):
        if not self.record_norm:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            for i, cap in enumerate(self.normal_cams):
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 20
                filename = f"camera{i}_{timestamp}.avi"
                writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
                self.norm_writers[i] = writer
            self.record_norm = True
            self.norm_btn.config(text="Stop Normal Recording")
        else:
            self.record_norm = False
            for w in self.norm_writers:
                if w:
                    w.release()
            self.norm_writers = [None, None]
            self.norm_btn.config(text="Record Normal Cams")

    # --- Scientific camera functions ---
    def _sci_callback(self, data):
        img = self.sci_decoder.decode(data)
        with self.sci_lock:
            self.sci_frame = img
        if self.record_sci and self.sci_file:
            np.save(self.sci_file, data)

    def toggle_sci_record(self):
        if not self.record_sci:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.sci_file = open(f"sci_cam_{timestamp}.npy", 'wb')
            self.record_sci = True
            self.sci_btn.config(text="Stop Sci Recording")
        else:
            self.record_sci = False
            if self.sci_file:
                self.sci_file.close()
                self.sci_file = None
            self.sci_btn.config(text="Record Scientific Cam")

    # --- GUI update ---
    def _update_gui(self):
        for i in range(2):
            with self.normal_locks[i]:
                frame = self.normal_frames[i]
            if frame is not None:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.normal_labels[i].imgtk = imgtk
                self.normal_labels[i].configure(image=imgtk)
        with self.sci_lock:
            frame = self.sci_frame
        if frame is not None:
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.sci_label.imgtk = imgtk
            self.sci_label.configure(image=imgtk)
        self.after(10, self._update_gui)

    def on_close(self):
        if self.record_sci and self.sci_file:
            self.sci_file.close()
        self.sci_cam.endXfer()
        self.sci_cam.close()
        for cap in self.normal_cams:
            cap.release()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
