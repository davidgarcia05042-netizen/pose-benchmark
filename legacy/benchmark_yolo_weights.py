"""
Benchmark de UN archivo de pesos YOLO específico (cualquier familia:
YOLOv8, YOLO11, YOLO26, ...) sobre un video fijo. Complementa
benchmark_yolov8.py (que solo cubre las 5 variantes hardcodeadas de
YOLOv8 n/s/m/l/x) para candidatos nuevos como YOLO26-nano, sin tocar la
convención estable de ese script.

Mismo runner/metodología que benchmark_yolov8.py -- ver ese archivo para
el detalle de por qué --device es obligatorio.

HALLAZGO IMPORTANTE (2026-08-25): se comprobó que Ultralytics resetea
torch.set_num_threads() a su propio default durante la primera
inferencia, ignorando el límite fijado antes de cargar el modelo -- por
eso una corrida con --cpu-threads 2 FUERA de Docker medía ~730% de CPU
en vez de ~200%. Este script reafirma el límite en cada frame como
mitigación (reduce el problema pero no lo garantiza al 100%, ver
docstring de benchmark_yolov8.py). La medición confiable de verdad
sigue siendo el contenedor Docker (límite a nivel de cgroup/kernel, no
depende de que PyTorch/Ultralytics cooperen -- ya validado con YOLOv8 en
results/yolov8_docker_2cpu_8gb.csv, ~200% consistente en las 15 filas).

Uso:
    python -m scripts.benchmark_yolo_weights --video videos/mi_video.mp4 \
        --weights yolo26n-pose.pt --label yolo26n --device cpu --cpu-threads 2
"""

import argparse
import os

import cv2

from legacy.runner import run_benchmark

RENDER_CPU_COUNT = 2
RENDER_RAM_MB = 8192

_DEVICE_MAP = {"cpu": "cpu", "cuda": "cuda:0"}


def _limit_cpu_threads(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    cv2.setNumThreads(n)


def process_video_with_model(model, video_path: str, device: str, max_frames: int | None, cpu_threads: int | None, torch_module) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_count >= max_frames:
            break
        if cpu_threads is not None:
            # Mitigación del hallazgo de reseteo de threads (ver docstring del módulo).
            torch_module.set_num_threads(cpu_threads)
        model(frame, device=device, verbose=False)
        frame_count += 1

    cap.release()
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Video fijo de prueba (usar el MISMO en todas las corridas)")
    parser.add_argument("--weights", required=True, help="Archivo de pesos, ej. 'yolo26n-pose.pt'")
    parser.add_argument("--label", required=True, help="Nombre corto para el CSV, ej. 'yolo26n'")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        required=True,
        help="'cpu' simula las condiciones de Render (sin GPU). 'cuda' usa tu GPU -- dato de referencia.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help=f"Limita el paralelismo a N hilos (usa {RENDER_CPU_COUNT} para simular Render). Ver docstring "
        "del módulo -- best-effort fuera de Docker.",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--results-csv", default="results/yolo_weights_benchmark.csv")
    args = parser.parse_args()

    actual_device = _DEVICE_MAP[args.device]

    if args.cpu_threads is not None and args.device == "cuda":
        print("ADVERTENCIA: --cpu-threads no tiene efecto real en --device cuda. Se ignora.\n")

    if args.cpu_threads is not None and args.device == "cpu":
        _limit_cpu_threads(args.cpu_threads)
        print(f"Limitando a {args.cpu_threads} hilo(s) de CPU (best-effort + reafirmado por frame, ver docstring).\n")
    elif args.device == "cpu":
        print(f"AVISO: no se especificó --cpu-threads. NO representativo de Render.\n")

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("ADVERTENCIA: pediste --device cuda pero no hay GPU CUDA disponible. Abortando.")
            return
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    from ultralytics import YOLO
    import torch

    if args.cpu_threads is not None and args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)

    print(f"\n### {args.label} ({args.weights}, device={args.device}"
          f"{f', cpu_threads={args.cpu_threads}' if args.cpu_threads else ''}) ###")

    model = YOLO(args.weights)
    if args.cpu_threads is not None and args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)  # reafirmar tras cargar el modelo (Ultralytics lo resetea)

    thread_suffix = f"_{args.cpu_threads}threads" if args.cpu_threads is not None else ""
    label = f"{args.label}_{args.device}{thread_suffix}"

    def fn(model=model, video=args.video, device=actual_device, max_frames=args.max_frames):
        process_video_with_model(model, video, device, max_frames, args.cpu_threads, torch)

    run_benchmark(
        label=label,
        fn=fn,
        results_csv=args.results_csv,
        warmup_runs=args.warmup_runs,
        repetitions=args.repetitions,
    )

    print(f"\nListo. Resultados acumulados en: {args.results_csv}")
    print(f"(Presupuesto real de Render, para referencia: {RENDER_CPU_COUNT} CPU, {RENDER_RAM_MB} MB RAM)")


if __name__ == "__main__":
    main()
