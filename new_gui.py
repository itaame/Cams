from flask import Flask, Response
import cv2
import threading
import queue
import numpy as np
import json
import time
from datetime import datetime
from enum import IntEnum
from pypuclib import CameraFactory

app = Flask(__name__)

class FILE_TYPE(IntEnum):
    BINARY = 1

class FileCreator:
    def __init__(self, name):
        self.file = open(name + '.npy', 'wb')
        self.name = name
        self.oldSeq = 0
        self.opened = True

    def write(self, seq, data):
        if self.opened and seq != self.oldSeq:
            np.save(self.file, data)
            self.oldSeq = seq

    @staticmethod
    def create_json(name, cam):
        data = {
            'framerate': cam.framerate(),
            'shutter': cam.shutter(),
            'width': cam.resolution().width,
            'height': cam.resolution().height,
            'quantization': cam.decoder().quantization(),
        }
        with open(name + '.json', 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def close(self):
        if self.opened:
            self.file.close()
            self.opened = False

cam = CameraFactory().create()
cam.setFramerateShutter(500, 500)  # 500 fps and 1/500 shutter

decoder = cam.decoder()
res = cam.resolution()

frame_lock = threading.Lock()
recording_lock = threading.Lock()
frame_ready = threading.Event()
frame_queue = queue.Queue()
latest_frame = None
recording = False
fcreator = None
current_file = ''
last_frame_time = time.time()

def callback(xferData):
    # Queue raw data for background processing to avoid heavy work in callback
    frame_queue.put((xferData.sequenceNo(), xferData.data().copy()))
    # Update the timestamp here so the watchdog reflects actual frame arrivals
    global last_frame_time
    last_frame_time = time.time()


def process_frames():
    global latest_frame, fcreator, last_frame_time
    while True:
        try:
            seq, data = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        try:
            frame = decoder.decode(data, res)
        except Exception as exc:
            print(f"Decode error: {exc}")
            continue
        with frame_lock:
            latest_frame = frame
            frame_ready.set()
        # Ensure exclusive access while writing to avoid races with toggle_recording
        with recording_lock:
            if recording and fcreator is not None:
                fcreator.write(seq, data)


def watchdog():
    global last_frame_time
    while True:
        time.sleep(5)
        if time.time() - last_frame_time > 5:
            print("Watchdog: restarting camera transfer")
            try:
                cam.endXfer()
                cam.beginXfer(callback)
            except RuntimeError as exc:
                # Log the failure and pause briefly before retrying so the thread survives
                print(f"Watchdog: restart failed: {exc}")
                time.sleep(1)
            else:
                last_frame_time = time.time()

def generate():
    while True:
        if not frame_ready.wait(timeout=1):
            continue
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
            frame_ready.clear()
        if frame is None:
            continue
        try:
            ret, buffer = cv2.imencode('.jpg', frame)
        except Exception as exc:
            print(f"Encode error: {exc}")
            continue
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    global recording, fcreator, current_file
    with recording_lock:
        if recording:
            recording = False
            if fcreator is not None:
                fcreator.close()
                FileCreator.create_json(current_file, cam)
                fcreator = None
            return {'status': 'stopped'}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        current_file = f'sci_cam_{timestamp}'
        fcreator = FileCreator(current_file)
        recording = True
        return {'status': 'recording'}

@app.route('/')
def index():
    return (
        '<!DOCTYPE html>'
        '<html>'
        '<head>'
        '<style>'
        'body { margin: 0; }'
        '#topbar { padding: 10px; display: flex; align-items: center; gap: 10px; }'
        '#status { width: 15px; height: 15px; border-radius: 50%; background-color: grey; display: inline-block; }'
        '#stream { display: block; width: 900px; height: auto; }'
        '</style>'
        '</head>'
        '<body>'
        '<div id="topbar">'
        '<button id="rec" onclick="toggleRecording()">Start Recording</button>'
        '<span id="status"></span>'
        '</div>'
        '<img src="/video_feed" id="stream">'
        '<script>'
        'async function toggleRecording() {'
        '  const res = await fetch("/toggle_recording", {method: "POST"});'
        '  const data = await res.json();'
        '  const btn = document.getElementById("rec");'
        '  const status = document.getElementById("status");'
        '  if (data.status === "recording") {'
        '    btn.textContent = "Stop Recording";'
        '    status.style.backgroundColor = "red";'
        '  } else {'
        '    btn.textContent = "Start Recording";'
        '    status.style.backgroundColor = "grey";'
        '  }'
        '}'
        '</script>'
        '</body>'
        '</html>'
    )

if __name__ == '__main__':
    processor_thread = threading.Thread(target=process_frames, daemon=True)
    processor_thread.start()
    watchdog_thread = threading.Thread(target=watchdog, daemon=True)
    watchdog_thread.start()
    cam.beginXfer(callback)
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        cam.endXfer()
        cam.close()
        if fcreator is not None:
            fcreator.close()
            FileCreator.create_json(current_file, cam)
