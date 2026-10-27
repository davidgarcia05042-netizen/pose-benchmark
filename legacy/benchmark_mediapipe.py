"""
Benchmark de las 3 variantes de MediaPipe Pose (Lite, Full, Heavy) sobre un
video fijo, midiendo tiempo total de procesamiento, CPU, RAM y GPU (si
aplica). Fase 2 del benchmarking (ver docs/DECISIONES_TECNICAS.md) -- mismo runner/CSV que
benchmark_yolov8.py, para que ambos sean directamente comparables.

Sobre --cpu-threads en este script: a diferencia de PyTorch, MediaPipe
(basado en TensorFlow Lite) no expone un equivalente sencillo a
torch.set_num_threads(). El límite aquí es solo un intento best-effort
vía variables de entorno ANTES de importar mediapipe -- no hay garantía
de que se respete internamente durante toda la ejecución (con PyTorch,
que sí expone un control explícito, se comprobó que Ultralytics lo
resetea a mitad de ejecución -- ver el hallazgo documentado en
scripts/benchmark_yolov8.py). La medición confiable para simular el
límite real de 2 CPU de Render es el contenedor Docker
(docker-compose.yml, límite a nivel de cgroup/kernel, no depende de que
la librería coopere -- ya validado con YOLOv8 en
results/yolov8_docker_2cpu_8gb.csv, ~200% consistente).

MediaPipe (este build de Python) solo corre en CPU -- no hay ruta de GPU
sencilla como con PyTorch/CUDA. --device solo admite 'cpu', se deja el
argumento por consistencia con benchmark_yolov8.py y para que el CSV
tenga las mismas columnas/comparabilidad.

Uso:
    # Simula condiciones de Render (correr esto DENTRO del contenedor Docker,
    # ver docker-compose.yml -- es la medición que se reporta):
    python -m scripts.benchmark_mediapipe --video videos/mi_video.mp4 --device cpu

    # Probar solo una variante primero:
    python -m scripts.benchmark_mediapipe --video videos/mi_video.mp4 --device cpu --variants heavy
"""

import argparse
import os

import cv2

from legacy.runner import run_benchmark

ALL_VARIANTS = ["lite", "full", "heavy"]
_COMPLEXITY_MAP = {"lite": 0, "full": 1, "heavy": 2}

RENDER_CPU_COUNT = 2
RENDER_RAM_MB = 8192


def _limit_cpu_threads(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    cv2.setNumThreads(n)


def process_video_with_model(pose, video_path: str, max_frames: int | None) -> int:
    """Corre MediaPipe Pose sobre cada frame del video. Devuelve cuántos frames procesó."""
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
        rgb_frame = frame[:, :, ::-1]  # MediaPipe espera RGB, OpenCV entrega BGR
        pose.process(rgb_frame)
        frame_count += 1

    cap.release()
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Video fijo de prueba (usar el MISMO en todas las corridas)")
    parser.add_argument(
        "--device",
        choices=["cpu"],
        required=True,
        help="MediaPipe (este build) solo corre en CPU -- se deja el argumento por consistencia con benchmark_yolov8.py.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help=f"Intento best-effort de limitar hilos vía variables de entorno (ver docstring: NO garantizado, "
        f"la medición confiable es vía Docker). Usa {RENDER_CPU_COUNT} para intentar simular Render.",
    )
    parser.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=ALL_VARIANTS)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--results-csv", default="results/mediapipe_benchmark.csv")
    args = parser.parse_args()

    if args.cpu_threads is not None:
        _limit_cpu_threads(args.cpu_threads)
        print(
            f"Intentando limitar a {args.cpu_threads} hilo(s) vía variables de entorno "
            f"(best-effort, ver docstring -- la medición confiable es vía Docker).\n"
        )
    else:
        print(
            f"AVISO: no se especificó --cpu-threads. Esta corrida va a usar todos los hilos "
            f"disponibles de tu máquina, NO el límite de {RENDER_CPU_COUNT} CPU de Render.\n"
        )

    import mediapipe as mp

    for variant in args.variants:
        complexity = _COMPLEXITY_MAP[variant]
        print(
            f"\n### Variante: mediapipe_{variant} (model_complexity={complexity}, device={args.device}"
            f"{f', cpu_threads={args.cpu_threads}' if args.cpu_threads else ''}) ###"
        )

        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=complexity,
            min_detection_confidence=0.5,
        )

        thread_suffix = f"_{args.cpu_threads}threads" if args.cpu_threads is not None else ""
        label = f"mediapipe_{variant}_{args.device}{thread_suffix}"

        def fn(pose=pose, video=args.video, max_frames=args.max_frames):
            process_video_with_model(pose, video, max_frames)

        run_benchmark(
            label=label,
            fn=fn,
            results_csv=args.results_csv,
            warmup_runs=args.warmup_runs,
            repetitions=args.repetitions,
        )

        pose.close()

    print(f"\nListo. Resultados acumulados en: {args.results_csv}")
    print(f"(Presupuesto real de Render, para referencia: {RENDER_CPU_COUNT} CPU, {RENDER_RAM_MB} MB RAM)")


if __name__ == "__main__":
    main()
