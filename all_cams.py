from flask import Flask, Response, render_template_string
import cv2
import threading
import time
import datetime

app = Flask(__name__)

# ---- Standard Cameras Setup (lazy) ----
camera_indices = [1, 2]

# Camera resources are initialized on demand to avoid slow startup
cameras = []
properties = []
latest_frames = []
locks = []
_initialized = False


def init_cameras() -> None:
    """Open camera devices and start capture threads if not already done."""
    global cameras, properties, latest_frames, locks, _initialized
    if _initialized:
        return

    cameras = [cv2.VideoCapture(idx) for idx in camera_indices]
    properties = []
    latest_frames = [None] * len(cameras)
    locks = [threading.Lock() for _ in cameras]

    for cam in cameras:
        w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cam.get(cv2.CAP_PROP_FPS)) or 30
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam.set(cv2.CAP_PROP_FPS, fps)
        properties.append((w, h, fps))

    for i in range(len(cameras)):
        threading.Thread(target=capture_frames, args=(i,), daemon=True).start()

    _initialized = True


def capture_frames(idx: int) -> None:
    cap = cameras[idx]
    while True:
        success, frame = cap.read()
        if success:
            with locks[idx]:
                latest_frames[idx] = frame.copy()


@app.before_first_request
def setup_cameras() -> None:
    init_cameras()

# ---- MJPEG Streaming ----
def generate_stream(idx):
    while True:
        with locks[idx]:
            frame = latest_frames[idx]
        if frame is None:
            time.sleep(0.01)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


@app.route('/video_feed/<int:idx>')
def video_feed(idx: int):
    return Response(generate_stream(idx),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---- Recording Logic ----
is_recording = False
recording_stop_event = threading.Event()
recording_threads = []
video_writers = [None, None]
fourcc = cv2.VideoWriter_fourcc(*'XVID')


def get_filename_base(idx: int) -> str:
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'camera{idx}_{timestamp}'


def record_loop(idx: int):
    w, h, fps = properties[idx]
    filename = get_filename_base(idx) + '.avi'
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
    global is_recording, recording_threads
    if not is_recording:
        recording_stop_event.clear()
        recording_threads = [
            threading.Thread(target=record_loop, args=(0,), daemon=True),
            threading.Thread(target=record_loop, args=(1,), daemon=True)
        ]
        for t in recording_threads:
            t.start()
        is_recording = True
        return "Recording started for both cameras"
    return "Already recording"


@app.route('/stop_recording')
def stop_recording():
    global is_recording
    if is_recording:
        recording_stop_event.set()
        for writer in video_writers:
            if writer:
                writer.release()
        is_recording = False
        return "Recording stopped for both cameras"
    return "Not recording"


@app.route('/status')
def status():
    return {"recording": is_recording}


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
    </body>
    </html>
    ''')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)

