"""
Benchmark de las 5 variantes de YOLOv8-Pose (n, s, m, l, x) sobre un video
fijo, midiendo tiempo total de procesamiento, CPU, RAM y GPU (si aplica).

IMPORTANTE: --device es obligatorio y a propósito, no tiene default. Así
no hay forma de correr esto "sin darse cuenta" con la GPU cuando la
intención era simular CPU (Render), o viceversa.

Sobre --cpu-threads: por defecto, PyTorch usa TODOS los hilos disponibles
en tu máquina (en un Ryzen 5600G, hasta 12). Render solo da 2 CPU. Sin
limitar esto, tus tiempos de --device cpu miden "qué tan rápido es con
paralelismo casi ilimitado", no "qué tan rápido sería en Render". Usa
--cpu-threads 2 para una medición fiel al hardware real de despliegue.

HALLAZGO IMPORTANTE (2026-08-25): se comprobó que Ultralytics resetea
torch.set_num_threads() a su propio default DURANTE la primera
inferencia, ignorando el límite fijado antes de cargar el modelo -- con
--cpu-threads 2 fuera de Docker, el CPU% real medido era ~730%, no
~200%. Este script reafirma el límite en cada frame como mitigación
(process_video_with_model), pero eso reduce el problema, no lo
garantiza al 100% (el primer frame de cada corrida sigue escapando el
límite hasta que se reafirma). La medición confiable de verdad es el
contenedor Docker (límite a nivel de cgroup/kernel, no depende de que
PyTorch/Ultralytics cooperen -- ver docker-compose.yml, ya validado:
results/yolov8_docker_2cpu_8gb.csv muestra ~200% consistente en las 15
filas, a diferencia de results/yolov8_benchmark.csv que SÍ está
contaminado por este bug).

Uso:
    # Simula condiciones de Render de verdad (sin GPU, 2 CPU) -- esta es
    # la medición que más importa reportar en el documento final:
    python -m scripts.benchmark_yolov8 --video videos/mi_video.mp4 --device cpu --cpu-threads 2

    # CPU sin límite de hilos (usa todo el poder de tu máquina) -- dato de
    # referencia, NO representativo de Render:
    python -m scripts.benchmark_yolov8 --video videos/mi_video.mp4 --device cpu

    # Con tu GPU, para tener el dato de referencia (no el que se reporta
    # como "desempeño del sistema", pero útil para discusión/trabajo futuro):
    python -m scripts.benchmark_yolov8 --video videos/mi_video.mp4 --device cuda

    # Probar solo un par de variantes primero, para confirmar que todo
    # funciona antes de correr las 5 completas (que tardan bastante):
    python -m scripts.benchmark_yolov8 --video videos/mi_video.mp4 --device cpu --cpu-threads 2 --variants n s
"""

import argparse
import os

import cv2

from legacy.runner import run_benchmark

ALL_VARIANTS = ["n", "s", "m", "l", "x"]

# Ultralytics espera 'cpu' o algo como 'cuda:0', no simplemente 'cuda'
_DEVICE_MAP = {"cpu": "cpu", "cuda": "cuda:0"}

# Presupuesto real de hardware en Render, para referencia en los mensajes impresos
RENDER_CPU_COUNT = 2
RENDER_RAM_MB = 8192


def _limit_cpu_threads(n: int) -> None:
    """
    Limita el paralelismo de CPU a `n` hilos, en dos capas:
      1. Variables de entorno (deben fijarse ANTES de importar torch/ultralytics,
         porque MKL/OpenBLAS leen esto una sola vez al inicializarse).
      2. Llamadas explícitas en tiempo de ejecución (torch.set_num_threads,
         cv2.setNumThreads), como respaldo por si algún backend no respeta
         las variables de entorno.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    cv2.setNumThreads(n)


def process_video_with_model(
    model, video_path: str, device: str, max_frames: int | None, cpu_threads: int | None = None
) -> int:
    """Corre el modelo sobre cada frame del video. Devuelve cuántos frames procesó."""
    import torch

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
            # Mitigación del hallazgo de reseteo de threads, ver docstring del módulo.
            torch.set_num_threads(cpu_threads)
        model(frame, device=device, verbose=False)
        frame_count += 1

    cap.release()
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Video fijo de prueba (usar el MISMO en todas las corridas)")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        required=True,
        help="'cpu' simula las condiciones de Render (sin GPU) -- la medición principal a reportar. "
        "'cuda' usa tu GPU -- dato de referencia, no el resultado principal.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help=f"Limita el paralelismo a N hilos (usa {RENDER_CPU_COUNT} para simular el límite real de Render). "
        "Sin esta bandera, se usan todos los hilos disponibles de tu máquina -- NO representativo de producción. "
        "Solo aplica cuando --device cpu.",
    )
    parser.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=ALL_VARIANTS)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limitar frames por corrida (útil para una primera prueba rápida antes de correr todo el video)",
    )
    parser.add_argument("--results-csv", default="results/yolov8_benchmark.csv")
    args = parser.parse_args()

    actual_device = _DEVICE_MAP[args.device]

    if args.cpu_threads is not None and args.device == "cuda":
        print("ADVERTENCIA: --cpu-threads no tiene efecto real en --device cuda (la inferencia corre en GPU).")
        print("Se ignora para esta corrida.\n")

    if args.cpu_threads is not None and args.device == "cpu":
        _limit_cpu_threads(args.cpu_threads)
        print(f"Limitando a {args.cpu_threads} hilo(s) de CPU "
              f"({'igual al' if args.cpu_threads == RENDER_CPU_COUNT else 'distinto del'} "
              f"límite real de Render, que es {RENDER_CPU_COUNT} CPU).\n")
    elif args.device == "cpu":
        print(f"AVISO: no se especificó --cpu-threads. Esta corrida va a usar todos los hilos "
              f"disponibles de tu máquina, NO el límite de {RENDER_CPU_COUNT} CPU de Render. "
              f"Los resultados no son representativos de producción.\n")

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("ADVERTENCIA: pediste --device cuda pero PyTorch no detecta ninguna GPU CUDA disponible.")
            print("Verifica que instalaste la versión de PyTorch con soporte CUDA (no la de solo CPU),")
            print("y que los drivers de NVIDIA están instalados correctamente. Abortando.")
            return
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    from ultralytics import YOLO

    if args.cpu_threads is not None and args.device == "cpu":
        import torch

        torch.set_num_threads(args.cpu_threads)

    for variant in args.variants:
        weights = f"yolov8{variant}-pose.pt"
        print(f"\n### Variante: {weights} (device={args.device}"
              f"{f', cpu_threads={args.cpu_threads}' if args.cpu_threads else ''}) ###")

        # La descarga de pesos (primera vez) y la carga a memoria ocurren
        # aquí, FUERA de la función medida por run_benchmark -- a propósito:
        # se está midiendo el costo de procesar video, no el costo de
        # descargar/cargar el modelo (eso es una medición distinta, útil
        # para otra pregunta: "tiempo de arranque en frío del servicio").
        model = YOLO(weights)
        if args.cpu_threads is not None and args.device == "cpu":
            import torch

            torch.set_num_threads(args.cpu_threads)  # reafirmar tras cargar (Ultralytics lo resetea)

        thread_suffix = f"_{args.cpu_threads}threads" if args.cpu_threads is not None else ""
        label = f"yolov8{variant}_{args.device}{thread_suffix}"

        cpu_threads_for_run = args.cpu_threads if args.device == "cpu" else None

        def fn(model=model, video=args.video, device=actual_device, max_frames=args.max_frames, cpu_threads=cpu_threads_for_run):
            process_video_with_model(model, video, device, max_frames, cpu_threads)

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

