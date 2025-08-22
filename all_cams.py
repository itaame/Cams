from flask import Flask, Response, render_template_string, request
import cv2
import threading
import time
import datetime
from pypuclib import CameraFactory

app = Flask(__name__)

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
sci_frame = None
sci_lock = threading.Lock()

# Callback to receive frames from the scientific camera.
# Only store the raw frame data here to keep the callback lightweight.
def sci_cam_callback(data):
    global sci_frame
    with sci_lock:
        sci_frame = data

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
        else:
            with sci_lock:
                frame_data = sci_frame
            if frame_data is None:
                continue
            frame = sci_decoder.decode(frame_data)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed/<int:idx>')
def video_feed(idx):
    return Response(generate_stream(idx), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---- Recording Logic ----
is_recording = False
recording_stop_event = threading.Event()
recording_threads = []
video_writers = [None, None, None]
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Properties for the scientific camera (use actual resolution)
sci_res = sci_cam.resolution()
properties.append((sci_res.width, sci_res.height, 30))  # Adjust FPS if needed

def get_filename(idx):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'camera{idx}_{timestamp}.avi'

def record_loop(idx):
    global video_writers
    w, h, fps = properties[idx]
    filename = get_filename(idx)
    writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    video_writers[idx] = writer
    print(f"[INFO] Started recording: {filename}")

    while not recording_stop_event.is_set():
        if idx < 2:
            with locks[idx]:
                frame = latest_frames[idx]
            if frame is not None:
                writer.write(frame)
        else:
            with sci_lock:
                frame_data = sci_frame
            if frame_data is not None:
                frame = sci_decoder.decode(frame_data)
                writer.write(frame)
        time.sleep(1 / fps)

    writer.release()
    video_writers[idx] = None
    print(f"[INFO] Finished recording: {filename}")

@app.route('/start_recording')
def start_recording():
    global is_recording, recording_threads, recording_stop_event
    if not is_recording:
        recording_stop_event.clear()
        recording_threads = [
            threading.Thread(target=record_loop, args=(0,), daemon=True),
            threading.Thread(target=record_loop, args=(1,), daemon=True),
            threading.Thread(target=record_loop, args=(2,), daemon=True)  # Scientific cam
        ]
        for t in recording_threads:
            t.start()
        is_recording = True
        return "Recording started for all cameras"
    return "Already recording"

@app.route('/stop_recording')
def stop_recording():
    global is_recording, recording_stop_event
    if is_recording:
        recording_stop_event.set()
        for writer in video_writers:
            if writer:
                writer.release()
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
