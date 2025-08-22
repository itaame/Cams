from flask import Flask, Response, render_template_string, request
import cv2
import threading
import time
import datetime
from pypuclib import CameraFactory
import os, csv, json
import numpy as np
from enum import IntEnum

app = Flask(__name__)


class FILE_TYPE(IntEnum):
    CSV = 0
    BINARY = 1


class FileCreator():
    def __init__(self, name, filetype):
        if filetype == FILE_TYPE.CSV:
            self.file = open(name + ".csv", 'w')
            self.writer = csv.writer(self.file, lineterminator='\n')
            self.writer.writerow(["SequenceNo", "diff"])
        elif filetype == FILE_TYPE.BINARY:
            self.file = open(name + ".npy", 'wb')
        else:
            return

        self.oldSeq = 0
        self.opened = True
        self.filetype = filetype

    def write(self, xferData):
        if self.opened:
            if self.filetype == FILE_TYPE.CSV:
                self.write_csv(xferData.sequenceNo())
            elif self.filetype == FILE_TYPE.BINARY:
                self.write_binary(xferData.sequenceNo(),
                                  xferData.data())

    def write_csv(self, seq):
        if self.oldSeq != seq:
            self.writer.writerow(
                        [str(seq),
                        str(seq - self.oldSeq),
                        "*" if (seq - self.oldSeq) > 1 else ""])
            self.oldSeq = seq

    def write_binary(self, seq, nparray):
        if self.oldSeq != seq:
            np.save(self.file, nparray)
            self.oldSeq = seq

    def create_json(name, cam):
        data = dict()
        data["framerate"] = cam.framerate()
        data["shutter"] = cam.shutter()
        data["width"] = cam.resolution().width
        data["height"] = cam.resolution().height
        data["quantization"] = cam.decoder().quantization()
        with open(name+".json", mode='wt', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def close(self):
        if self.opened:
            self.file.close()

# ---- Standard Cameras Setup (IDs 1 and 2) ----
camera_indices = [1, 2]
cameras = [cv2.VideoCapture(idx) for idx in camera_indices]

properties = []
for cam in cameras:
    w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cam.get(cv2.CAP_PROP_FPS)) or 20
    properties.append((w, h, fps))

# ---- Scientific Camera Setup ----
sci_cam = CameraFactory().create()
sci_cam.setFramerate(500)
sci_decoder = sci_cam.decoder()
sci_jpeg = None
sci_lock = threading.Lock()
sci_is_recording = False
sci_file_creator = None
sci_filename_base = None

# Callback to receive frames from the scientific camera.
# Decode and encode to JPEG for streaming and write to file if recording.
def sci_cam_callback(data):
    global sci_jpeg, sci_is_recording, sci_file_creator
    array = sci_decoder.decode(data)
    ret, buffer = cv2.imencode('.jpg', array)
    if ret:
        with sci_lock:
            sci_jpeg = buffer.tobytes()
    if sci_is_recording and sci_file_creator:
        sci_file_creator.write(data)

def start_scientific_camera():
    sci_cam.beginXfer(sci_cam_callback)

def stop_scientific_camera():
    sci_cam.endXfer()

# ---- Frame Holders ----
latest_frames = [None, None]  # For OpenCV cams
locks = [threading.Lock(), threading.Lock()]

# ---- Frame Capture Threads for OpenCV Cameras ----
def capture_frames(idx):
    global latest_frames
    while True:
        success, frame = cameras[idx].read()
        if success:
            with locks[idx]:
                latest_frames[idx] = frame.copy()

for i in range(2):
    threading.Thread(target=capture_frames, args=(i,), daemon=True).start()

# ---- MJPEG Streaming ----
def generate_stream(idx):
    while True:
        if idx < 2:
            with locks[idx]:
                frame = latest_frames[idx]
            if frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            with sci_lock:
                frame = sci_jpeg
            if frame is None:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/<int:idx>')
def video_feed(idx):
    return Response(generate_stream(idx), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---- Recording Logic ----
is_recording = False
recording_stop_event = threading.Event()
recording_threads = []
video_writers = [None, None]
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Properties for the scientific camera (use actual resolution)
sci_res = sci_cam.resolution()
properties.append((sci_res.width, sci_res.height, 30))  # Adjust FPS if needed

def get_filename_base(idx):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'camera{idx}_{timestamp}'

def record_loop(idx):
    global video_writers
    w, h, fps = properties[idx]
    base = get_filename_base(idx)
    filename = base + '.avi'
    writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    video_writers[idx] = writer
    print(f"[INFO] Started recording: {filename}")

    while not recording_stop_event.is_set():
        with locks[idx]:
            frame = latest_frames[idx]
        if frame is not None:
            writer.write(frame)
        time.sleep(1 / fps)

    writer.release()
    video_writers[idx] = None
    print(f"[INFO] Finished recording: {filename}")

@app.route('/start_recording')
def start_recording():
    global is_recording, recording_threads, recording_stop_event
    global sci_is_recording, sci_file_creator, sci_filename_base
    if not is_recording:
        recording_stop_event.clear()
        recording_threads = [
            threading.Thread(target=record_loop, args=(0,), daemon=True),
            threading.Thread(target=record_loop, args=(1,), daemon=True)
        ]
        for t in recording_threads:
            t.start()
        sci_filename_base = get_filename_base(2)
        sci_file_creator = FileCreator(sci_filename_base, FILE_TYPE.BINARY)
        sci_is_recording = True
        is_recording = True
        return "Recording started for all cameras"
    return "Already recording"

@app.route('/stop_recording')
def stop_recording():
    global is_recording, recording_stop_event
    global sci_is_recording, sci_file_creator, sci_filename_base
    if is_recording:
        recording_stop_event.set()
        for writer in video_writers:
            if writer:
                writer.release()
        if sci_is_recording and sci_file_creator:
            sci_file_creator.close()
            FileCreator.create_json(sci_filename_base, sci_cam)
            sci_is_recording = False
            sci_file_creator = None
        is_recording = False
        return "Recording stopped for all cameras"
    return "Not recording"

@app.route('/status')
def status():
    return {"recording": is_recording}

# ---- Web Interface ----
@app.route('/')
def index():
    return render_template_string('''
    <html>
    <head>
        <style>
            #rec-indicator {
                width: 15px;
                height: 15px;
                border-radius: 50%;
                display: inline-block;
                margin-left: 10px;
                background-color: gray;
            }
            #rec-indicator.active {
                background-color: red;
            }
        </style>
    </head>
    <body>

        <button onclick="startRecording()">Start Recording</button>
        <button onclick="stopRecording()">Stop Recording</button>
        <span id="rec-indicator" title="Recording status"></span>

        <script>
        function updateIndicator() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    const indicator = document.getElementById('rec-indicator');
                    if (data.recording) {
                        indicator.classList.add('active');
                    } else {
                        indicator.classList.remove('active');
                    }
                });
        }

        function startRecording() {
            fetch('/start_recording')
                .then(() => updateIndicator());
        }

        function stopRecording() {
            fetch('/stop_recording')
                .then(() => updateIndicator());
        }

        setInterval(updateIndicator, 5000);
        updateIndicator();
        </script>

        <img src="/video_feed/0" width="480"><br>
        <img src="/video_feed/1" width="480"><br>
        <img src="/video_feed/2" width="480"><br><br>
    </body>
    </html>
    ''')

# ---- Main ----
if __name__ == '__main__':
    start_scientific_camera()
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        stop_scientific_camera()
