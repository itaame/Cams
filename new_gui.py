from flask import Flask, Response
import cv2
import threading
import numpy as np
import json
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

    def write(self, xferData):
        if self.opened:
            seq = xferData.sequenceNo()
            if seq != self.oldSeq:
                np.save(self.file, xferData.data())
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
latest_frame = None
recording = False
fcreator = None
current_file = ''

def callback(xferData):
    global latest_frame, fcreator, recording
    with frame_lock:
        if recording and fcreator is not None:
            fcreator.write(xferData)
        latest_frame = decoder.decode(xferData.data(), res)

cam.beginXfer(callback)

def generate():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
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
        '<img src="/video_feed">'
        '<button id="rec" onclick="toggleRecording()">Start Recording</button>'
        '<script>'
        'async function toggleRecording() {'
        '  const res = await fetch("/toggle_recording", {method: "POST"});'
        '  const data = await res.json();'
        '  const btn = document.getElementById("rec");'
        '  if (data.status === "recording") {'
        '    btn.textContent = "Stop Recording";'
        '  } else {'
        '    btn.textContent = "Start Recording";'
        '  }'
        '}'
        '</script>'
    )

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        cam.endXfer()
        if fcreator is not None:
            fcreator.close()
            FileCreator.create_json(current_file, cam)
