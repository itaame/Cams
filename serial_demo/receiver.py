import serial
import time

PORT = '/dev/ttyUSB0'  # adjust to your serial port
BAUD = 115200

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    last = None
    while True:
        line = ser.readline()
        if not line:
            continue
        now = time.time()
        text = line.decode(errors='ignore').strip()
        if last is not None:
            freq = 1.0 / (now - last)
            print(f"{text} (freq: {freq:.2f} Hz)")
        else:
            print(text)
        last = now
