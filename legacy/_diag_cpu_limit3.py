import os
import sys

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "2"

import time

import cv2
import psutil
import torch

cv2.setNumThreads(2)
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

# El video se pasa como primer argumento; sin argumento, la ruta relativa
# videos/marcha_p01.mp4 (ver legacy/README.md).
video_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("videos", "marcha_p01.mp4")
cap = cv2.VideoCapture(video_path)

process = psutil.Process()
process.cpu_percent(interval=None)

samples = []
thread_log = []
count = 0
start = time.perf_counter()
while count < 30:
    ret, frame = cap.read()
    if not ret:
        break
    torch.set_num_threads(2)  # reafirmar en CADA frame
    model(frame, device="cpu", verbose=False)
    thread_log.append(torch.get_num_threads())
    samples.append(process.cpu_percent(interval=None))
    count += 1
elapsed = time.perf_counter() - start
cap.release()

print(f"Frames: {count} en {elapsed:.1f}s")
print(f"CPU%% medio: {sum(samples)/len(samples):.1f}   max: {max(samples):.1f}")
print(f"torch threads justo despues de cada frame (primeros 10): {thread_log[:10]}")
