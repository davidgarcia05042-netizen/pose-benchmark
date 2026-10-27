"""
Runner genérico de benchmarking. No sabe nada de MediaPipe ni de YOLOv8 a
propósito -- solo recibe una función sin argumentos (`fn`) que representa
"procesar un video con tal modelo, tal variante", la mide, y guarda una
fila en un CSV. Así el mismo runner sirve para todas las 8 configuraciones
(5 de YOLOv8 + 3 de MediaPipe) sin duplicar lógica de medición.
"""

import csv
import time
from pathlib import Path
from statistics import mean
from typing import Callable

from legacy.resource_monitor import ResourceMonitor

CSV_COLUMNS = [
    "label",
    "repetition",
    "elapsed_seconds",
    "cpu_percent_mean",
    "cpu_percent_max",
    "ram_mb_mean",
    "ram_mb_peak",
    "gpu_util_percent_mean",
    "gpu_memory_mb_peak",
]


def run_benchmark(
    label: str,
    fn: Callable[[], None],
    results_csv: str,
    warmup_runs: int = 1,
    repetitions: int = 3,
    sample_interval_seconds: float = 0.2,
) -> None:
    """
    label: identificador de la configuración medida (ej. "yolov8n", "mediapipe_heavy")
    fn: función sin argumentos que ejecuta el trabajo a medir (ej. procesar un video fijo)
    results_csv: ruta del CSV donde se acumulan los resultados (se agrega, no se sobreescribe)
    warmup_runs: repeticiones antes de medir, para no contar el costo de
                 cargar el modelo la primera vez (carga de pesos, inicialización)
    repetitions: cuántas veces medir de verdad, para poder ver variabilidad
                 entre corridas, no solo un número suelto
    """
    print(f"=== Calentando: {label} ({warmup_runs} corrida(s) sin medir) ===")
    for _ in range(warmup_runs):
        fn()

    results_path = Path(results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    file_is_new = not results_path.exists()

    with open(results_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if file_is_new:
            writer.writerow(CSV_COLUMNS)

        for rep in range(1, repetitions + 1):
            with ResourceMonitor(interval_seconds=sample_interval_seconds) as monitor:
                start = time.perf_counter()
                fn()
                elapsed = time.perf_counter() - start

            samples = monitor.stats.samples
            cpu_values = [s.cpu_percent for s in samples]
            ram_values = [s.ram_mb for s in samples]
            gpu_util_values = [s.gpu_utilization_percent for s in samples if s.gpu_utilization_percent is not None]
            gpu_mem_values = [s.gpu_memory_mb for s in samples if s.gpu_memory_mb is not None]

            row = {
                "label": label,
                "repetition": rep,
                "elapsed_seconds": round(elapsed, 4),
                "cpu_percent_mean": round(mean(cpu_values), 2) if cpu_values else "",
                "cpu_percent_max": round(max(cpu_values), 2) if cpu_values else "",
                "ram_mb_mean": round(mean(ram_values), 2) if ram_values else "",
                "ram_mb_peak": round(max(ram_values), 2) if ram_values else "",
                "gpu_util_percent_mean": round(mean(gpu_util_values), 2) if gpu_util_values else "",
                "gpu_memory_mb_peak": round(max(gpu_mem_values), 2) if gpu_mem_values else "",
            }
            writer.writerow([row[col] for col in CSV_COLUMNS])
            csv_file.flush()

            gpu_note = (
                f", GPU media {row['gpu_util_percent_mean']}%, VRAM pico {row['gpu_memory_mb_peak']}MB"
                if gpu_util_values
                else " (sin GPU NVIDIA detectada)"
            )
            print(
                f"[{label}] rep {rep}/{repetitions}: {elapsed:.2f}s, "
                f"CPU medio {row['cpu_percent_mean']}%, RAM pico {row['ram_mb_peak']}MB{gpu_note}"
            )

    print(f"-> Resultados acumulados en: {results_path}\n")
