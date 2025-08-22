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

# ---- Standard Cameras (OpenCV) Setup ----
webcams = [cv2.VideoCapture(0), cv2.VideoCapture(1)]
webcam_frames = [None, None]
webcam_locks = [threading.Lock(), threading.Lock()]
webcam_recording = [False, False]
webcam_writers = [None, None]
webcam_props = []
fourcc = cv2.VideoWriter_fourcc(*'XVID')

for cam_idx, wcam in enumerate(webcams):
    width = int(wcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(wcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = wcam.get(cv2.CAP_PROP_FPS) or 20
    webcam_props.append((width, height, fps))


def capture_webcam(idx):
    cap = webcams[idx]
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with webcam_locks[idx]:
            webcam_frames[idx] = frame.copy()
            if webcam_recording[idx] and webcam_writers[idx] is not None:
                webcam_writers[idx].write(frame)


for i in range(2):
    threading.Thread(target=capture_webcam, args=(i,), daemon=True).start()


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


def generate_webcam(idx):
    while True:
        with webcam_locks[idx]:
            frame = webcam_frames[idx]
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video_feed_cam/<int:idx>')
def video_feed_cam(idx):
    if idx not in (0, 1):
        return ('Camera not found', 404)
    return Response(generate_webcam(idx), mimetype='multipart/x-mixed-replace; boundary=frame')

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


@app.route('/toggle_recording_cam/<int:idx>', methods=['POST'])
def toggle_recording_cam(idx):
    if idx not in (0, 1):
        return {'status': 'invalid'}
    with webcam_locks[idx]:
        if webcam_recording[idx]:
            webcam_recording[idx] = False
            if webcam_writers[idx] is not None:
                webcam_writers[idx].release()
                webcam_writers[idx] = None
            return {'status': 'stopped'}
        width, height, fps = webcam_props[idx]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'cam{idx}_{timestamp}.avi'
        writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        webcam_writers[idx] = writer
        webcam_recording[idx] = True
        return {'status': 'recording'}

@app.route('/')
def index():
    return (
        '<div>'
        '<img src="/video_feed">'
        '<button id="rec" onclick="toggleRecording()">Start Recording</button>'
        '</div>'
        '<div>'
        '<img src="/video_feed_cam/0">'
        '<button id="rec_cam0" onclick="toggleRecordingCam(0)">Start Recording</button>'
        '</div>'
        '<div>'
        '<img src="/video_feed_cam/1">'
        '<button id="rec_cam1" onclick="toggleRecordingCam(1)">Start Recording</button>'
        '</div>'
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
        'async function toggleRecordingCam(idx) {'
        '  const res = await fetch(`/toggle_recording_cam/${idx}`, {method: "POST"});'
        '  const data = await res.json();'
        '  const btn = document.getElementById(`rec_cam${idx}`);'
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
        for cap in webcams:
            cap.release()
        for writer in webcam_writers:
            if writer is not None:
                writer.release()
