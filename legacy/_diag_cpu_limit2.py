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
print("torch threads justo despues de cargar el modelo:", torch.get_num_threads())

# FIX: reafirmar el limite DESPUES de que Ultralytics haya cargado el modelo
# (Ultralytics lo reescribe internamente al inicializar).
torch.set_num_threads(2)
print("torch threads tras reafirmar:", torch.get_num_threads())

# El video se pasa como primer argumento; sin argumento, la ruta relativa
# videos/marcha_p01.mp4 (ver legacy/README.md).
video_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("videos", "marcha_p01.mp4")
cap = cv2.VideoCapture(video_path)

process = psutil.Process()
process.cpu_percent(interval=None)

samples = []
count = 0
start = time.perf_counter()
while count < 60:
    ret, frame = cap.read()
    if not ret:
        break
    model(frame, device="cpu", verbose=False)
    samples.append(process.cpu_percent(interval=None))
    count += 1
elapsed = time.perf_counter() - start
cap.release()

print(f"\nFrames procesados: {count} en {elapsed:.1f}s")
print(f"CPU%% medio: {sum(samples)/len(samples):.1f}")
print(f"CPU%% max: {max(samples):.1f}")
print(f"torch threads AL FINAL: {torch.get_num_threads()}")
