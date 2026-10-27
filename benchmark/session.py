"""
Una sesión de medición: UN modelo sobre UN video, en un proceso propio.

`scripts/run_benchmark.py` lanza una sesión por cada combinación modelo-video
(y, con --concurrency N, N sesiones a la vez). Separar procesos no es un
detalle: el pico de RAM y los hilos de un proceso que ya cargó torch no son
los de uno que solo cargó MediaPipe, y un OOM-kill a mitad de un modelo no se
lleva por delante al resto de la suite.

Qué hace cada repetición, igual a lo que paga el servicio por cada job:

1. Construye el modelo desde cero (`model_init_s`). El servicio crea el modelo
   en cada job (`analyze_video`), así que ese costo es parte del job.
2. Abre el video con `open_video` (rotación aplicada) y lo procesa completo
   bajo el monitor de recursos (`processing_s`): decodificar + rotar +
   convertir + inferir, frame por frame.
3. Emite una fila `BENCHMARK_ROW` en el log.

Primero corren `--warmup-runs` repeticiones que se emiten con
phase="warmup": no entran al análisis, pero la primera es el arranque en frío
del proceso (carga de librerías, caché de disco vacía) y queda registrada.

La sesión no escribe archivos: solo imprime. Quien la lanza recoge las filas.

Uso directo (para depurar; normalmente la lanza run_benchmark):
    python -m benchmark.session --model mediapipe_lite --video ruta.mp4 --video-id prueba
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Solo biblioteca estándar hasta fijar los hilos: numpy (OpenBLAS), torch y
# mediapipe leen las variables de entorno UNA vez, al importarse.
from benchmark.cgroup import Cgroup, effective_cpu_count
from benchmark.models import MODELS, apply_thread_env

_MB = 1024 * 1024


def resolve_threads(value: str, cgroup: Cgroup, concurrency: int = 1) -> tuple[int | None, str]:
    """
    'auto' = los núcleos que el contenedor deja usar (cuota del cgroup o
    afinidad), repartidos entre las `concurrency` sesiones que corren a la vez;
    un número = ese número; 'library' = no tocar nada y dejar el default de
    cada librería (solo tiene sentido para comparar).

    Por qué repartir: medido el 2026-09-28 en Docker (2 CPU, 5EJ6), dos
    procesos de YOLO26-nano con 2 hilos cada uno (4 hilos para 2 núcleos)
    tardaron 199 ms/frame; con 1 hilo cada uno, 145 ms/frame.
    """
    if value == "auto":
        cores, source = effective_cpu_count(cgroup.limits())
        if concurrency > 1:
            return max(1, cores // concurrency), f"{source}/{concurrency}"
        return cores, source
    if value == "library":
        return None, "library_default"
    threads = int(value)
    if threads < 1:
        raise ValueError("--threads debe ser 1 o mayor")
    return threads, "explicit"


def _rounded(row: dict) -> dict:
    return {key: round(value, 4) if isinstance(value, float) else value for key, value in row.items()}


def measure_repetition(runner_factory, video_path: Path, max_frames: int | None, cgroup: Cgroup, sample_interval_s: float):
    """
    Una repetición. `runner_factory` construye el modelo (se cronometra aparte).
    Devuelve un dict con tiempos por frame y el resultado del monitor.
    """
    import psutil

    from benchmark.monitor import ResourceMonitor
    from benchmark.video_io import open_video

    process = psutil.Process()
    # RSS antes de construir el modelo. En la primera repetición es el proceso
    # recién arrancado; en las siguientes incluye lo que dejaron los modelos
    # anteriores. Esa diferencia es la deriva de memoria de un worker que crea
    # un modelo por job (ver docs/criterios_rendimiento.md, "Deriva de memoria").
    ram_start_mb = process.memory_info().rss / _MB
    started = time.perf_counter()
    runner = runner_factory()
    model_init_s = time.perf_counter() - started
    ram_after_init_mb = process.memory_info().rss / _MB

    decode_ms: list[float] = []
    infer_ms: list[float] = []
    detected = 0
    try:
        with ResourceMonitor(interval_s=sample_interval_s, cgroup=cgroup) as monitor:
            cap = open_video(video_path)
            try:
                while max_frames is None or len(infer_ms) < max_frames:
                    t_read = time.perf_counter()
                    ok, frame = cap.read()  # incluye la rotación, si el video la declara
                    t_infer = time.perf_counter()
                    if not ok:
                        break
                    detected += bool(runner.infer(frame))
                    t_done = time.perf_counter()
                    decode_ms.append((t_infer - t_read) * 1000)
                    infer_ms.append((t_done - t_infer) * 1000)
            finally:
                cap.release()
        threads_effective = runner.threads_effective()
    finally:
        runner.close()

    if not infer_ms:
        raise ValueError(f"El video no tiene frames legibles: {video_path}")
    return {
        "model_init_s": model_init_s,
        "ram_start_mb": ram_start_mb,
        "ram_after_init_mb": ram_after_init_mb,
        "decode_ms": decode_ms,
        "infer_ms": infer_ms,
        "detected": detected,
        "threads_effective": threads_effective,
        "monitor": monitor.result,
    }


def build_row(measurement: dict, context: dict) -> dict:
    """Arma la fila a partir de una repetición medida y del contexto de la sesión."""
    from benchmark import config
    from benchmark.stats import percentile

    monitor = measurement["monitor"]
    frames = len(measurement["infer_ms"])
    processing_s = monitor.wall_s
    fps = context.get("video_fps") or 0
    processed_duration_s = frames / fps if fps else None
    row = dict(context)
    row.update(monitor.as_row())
    row.pop("wall_s", None)
    nr_periods = row.get("cgroup_nr_periods")
    nr_throttled = row.get("cgroup_nr_throttled")
    row.update(
        {
            "frames_processed": frames,
            "model_init_s": measurement["model_init_s"],
            "processing_s": processing_s,
            "job_s": measurement["model_init_s"] + processing_s,
            "ms_per_frame": processing_s * 1000 / frames,
            "fps_processed": frames / processing_s if processing_s else None,
            "realtime_factor": processing_s / processed_duration_s if processed_duration_s else None,
            "first_frame_ms": measurement["decode_ms"][0] + measurement["infer_ms"][0],
            "infer_ms_p50": percentile(measurement["infer_ms"], 50),
            "infer_ms_p90": percentile(measurement["infer_ms"], 90),
            "infer_ms_p99": percentile(measurement["infer_ms"], 99),
            "infer_ms_max": max(measurement["infer_ms"]),
            "decode_ms_mean": sum(measurement["decode_ms"]) / frames,
            "decode_ms_p90": percentile(measurement["decode_ms"], 90),
            "detection_rate": measurement["detected"] / frames,
            "threads_effective": measurement["threads_effective"],
            "cpu_budget_pct": monitor.cpu_cores_used / config.BUDGET_CPU_CORES * 100 if monitor.cpu_cores_used is not None else None,
            "cgroup_throttled_pct": nr_throttled / nr_periods * 100 if nr_periods and nr_throttled is not None else None,
            "ram_start_mb": measurement["ram_start_mb"],
            "ram_after_init_mb": measurement["ram_after_init_mb"],
            "ram_budget_pct": monitor.ram_mb_peak / config.BUDGET_RAM_MB * 100 if monitor.ram_mb_peak is not None else None,
        }
    )
    return _rounded(row)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmark.session", description="Mide UN modelo sobre UN video.")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video-sha256", default=None)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--threads", default="auto", help="'auto' (del cgroup), un número, o 'library'")
    parser.add_argument("--weights-dir", type=Path, default=Path("."))
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--parallel-index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repetitions < 1 or args.warmup_runs < 0:
        _parser().error("--repetitions debe ser >= 1 y --warmup-runs >= 0")

    cgroup = Cgroup.detect()
    threads, threads_source = resolve_threads(args.threads, cgroup, args.concurrency)
    if threads:
        apply_thread_env(threads)

    # Recién ahora las librerías pesadas.
    import cv2

    from benchmark import environment, results
    from benchmark.models import load_runner
    from benchmark.video_io import VIDEO_ORIENTATION_TAG, probe_video

    spec = MODELS[args.model]
    base = {
        "schema_version": results.SCHEMA_VERSION,
        "run_id": args.run_id,
        "model": spec.name,
        "model_family": spec.family,
        "model_role": spec.role,
        "video_id": args.video_id,
        "video_sha256": args.video_sha256,
        "video_orientation": VIDEO_ORIENTATION_TAG,
        "trial": args.trial,
        "concurrency": args.concurrency,
        "parallel_index": args.parallel_index,
        "threads_requested": threads,
        "threads_source": threads_source,
    }
    label = f"[{spec.name} | {args.video_id}" + (f" | proceso {args.parallel_index + 1}/{args.concurrency}" if args.concurrency > 1 else "") + "]"

    try:
        base.update(probe_video(args.video).as_row())
        base.update(environment.row_fields(environment.snapshot(cgroup)))
        plan = [("warmup", i) for i in range(1, args.warmup_runs + 1)] + [("measured", i) for i in range(1, args.repetitions + 1)]
        for phase, repetition in plan:
            context = dict(base, phase=phase, repetition=repetition,
                           timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            measurement = measure_repetition(
                lambda: load_runner(spec, threads, args.weights_dir),
                args.video,
                args.max_frames,
                cgroup,
                args.sample_interval,
            )
            context["cv2_threads"] = cv2.getNumThreads()
            row = build_row(measurement, context)
            results.emit(results.ROW_PREFIX, row)
            name = "calentamiento" if phase == "warmup" else "medición"
            total = args.warmup_runs if phase == "warmup" else args.repetitions
            print(
                f"{label} {name} {repetition}/{total}: {row['processing_s']:.2f} s "
                f"({row['ms_per_frame']:.1f} ms/frame, init {row['model_init_s']:.2f} s), "
                f"{row['cpu_cores_used']:.2f} núcleos, RAM pico {row['ram_mb_peak']:.0f} MB "
                f"[{row['ram_peak_source']}], detección {row['detection_rate']:.0%}",
                flush=True,
            )
    except Exception as error:  # la fila de error sale igual por el log
        results.emit(results.ERROR_PREFIX, {"run_id": args.run_id, "model": args.model, "video_id": args.video_id,
                                            "parallel_index": args.parallel_index, "error": f"{type(error).__name__}: {error}"})
        print(f"{label} FALLÓ: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
