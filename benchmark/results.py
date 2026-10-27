"""
Esquema de una fila de resultados, y cómo sale del contenedor.

EL DISCO DE RENDER ES EFÍMERO: lo que se escriba en `results/` desaparece al
terminar o al redesplegar. Por eso cada fila sale por DOS caminos:

1. Por el log (stdout), como una línea `BENCHMARK_ROW {json}`. Es el canal que
   sobrevive al contenedor mientras no se decida dónde guardar los resultados
   (bloqueador #3). `scripts/collect_from_log.py` reconstruye el CSV a partir
   de un log descargado de Render.
2. A un CSV en `results/<run_id>/rows.csv`, que sirve localmente (Docker con
   el volumen montado) y en Render mientras el contenedor viva.

El log se escribe PRIMERO: si algo falla al escribir el CSV, la fila ya salió.

Las columnas de CPU y RAM que ya existían en los CSV del 2026-08-25
(`cpu_percent_mean`, `cpu_percent_max`, `ram_mb_mean`, `ram_mb_peak`) se
conservan con el mismo nombre y significado, para que lo nuevo se pueda poner
al lado de lo viejo. `elapsed_seconds` de aquellos CSV corresponde a
`processing_s` de acá.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterable, TextIO

SCHEMA_VERSION = 2

ROW_PREFIX = "BENCHMARK_ROW "
RUN_PREFIX = "BENCHMARK_RUN "
SUMMARY_PREFIX = "BENCHMARK_SUMMARY "
ERROR_PREFIX = "BENCHMARK_ERROR "

ROW_COLUMNS = [
    # --- Identificación ---
    "schema_version",
    "run_id",
    "timestamp_utc",
    "environment",
    "model",
    "model_family",
    "model_role",
    "video_id",
    "phase",  # "warmup" (no se analiza) o "measured"
    "repetition",
    "trial",
    "concurrency",  # procesos midiendo a la vez (1 = un video por vez)
    "parallel_index",
    # --- Video ---
    "video_file_name",
    "video_sha256",
    "video_size_mb",
    "video_fps",
    "video_frames_reported",
    "video_duration_s",
    "video_stored_resolution",
    "video_decoded_resolution",
    "video_rotation_meta_deg",
    "video_codec",
    "video_orientation",
    # --- Tiempo ---
    "frames_processed",
    "model_init_s",
    "processing_s",
    "job_s",
    "ms_per_frame",
    "fps_processed",
    "realtime_factor",
    "first_frame_ms",
    "infer_ms_p50",
    "infer_ms_p90",
    "infer_ms_p99",
    "infer_ms_max",
    "decode_ms_mean",
    "decode_ms_p90",
    "detection_rate",
    # --- CPU ---
    "threads_requested",
    "threads_source",
    "threads_effective",
    "cv2_threads",
    "cpu_s",
    "cpu_user_s",
    "cpu_system_s",
    "cpu_cores_used",
    "cpu_budget_pct",
    "cpu_percent_mean",
    "cpu_percent_max",
    # cgroup_*: de TODO el contenedor, no del proceso. Con --concurrency 2
    # incluyen a las dos sesiones que corren a la vez.
    "cgroup_cpu_s",
    "cgroup_nr_periods",
    "cgroup_nr_throttled",
    "cgroup_throttled_pct",
    "cgroup_throttled_s",
    # --- RAM ---
    "ram_start_mb",
    "ram_after_init_mb",
    "ram_mb_mean",
    "ram_mb_peak",
    "ram_mb_peak_sampled",
    "ram_peak_source",
    "ram_budget_pct",
    "cgroup_memory_peak_mb",
    "cgroup_oom_kill",
    "samples",
    # --- Entorno ---
    "cpu_model",
    "cgroup_version",
    "cpu_limit_cores",
    "memory_limit_mb",
    "host_cpu_count",
    "python_version",
    "mediapipe_version",
    "cv2_version",
    "numpy_version",
    "torch_version",
    "ultralytics_version",
    "versions_match_service",
    "render_instance_id",
    "git_commit",
]


def emit(prefix: str, payload: dict, stream: TextIO | None = None) -> None:
    """Una línea `PREFIJO {json}` en el log, sin saltos de línea dentro."""
    stream = stream or sys.stdout
    stream.write(prefix + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
    stream.flush()


class CsvSink:
    """Agrega filas a un CSV con el encabezado de ROW_COLUMNS."""

    def __init__(self, path: Path, columns: list[str] = ROW_COLUMNS):
        self.path = Path(path)
        self.columns = columns
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as existing:
                header = next(csv.reader(existing), None)
            if header != columns:
                # Mezclar esquemas en un mismo archivo lo vuelve ilegible sin
                # avisar. Mejor fallar y que se use otro archivo.
                raise ValueError(f"{self.path} ya existe con otras columnas; usar otro --run-id o --results-dir")

    def write(self, row: dict) -> None:
        is_new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            writer.writerow({column: _csv_value(row.get(column)) for column in self.columns})


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 4)
    return value


def write_summary_csv(path: Path, summaries: list[dict]) -> None:
    """El resumen de benchmark.stats.summarize, una fila por modelo y video."""
    if not summaries:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: _csv_value(value) for key, value in summary.items()})


def parse_log_lines(lines: Iterable[str]) -> dict[str, list[dict]]:
    """
    Extrae las líneas BENCHMARK_* de un log. Tolera prefijos antes del
    marcador (Render antepone fecha y hora a cada línea al descargar el log) y
    líneas cortadas o corruptas, que se cuentan en "invalid" en vez de abortar.
    """
    prefixes = {"rows": ROW_PREFIX, "runs": RUN_PREFIX, "summaries": SUMMARY_PREFIX, "errors": ERROR_PREFIX}
    parsed: dict[str, list[dict]] = {key: [] for key in prefixes}
    parsed["invalid"] = []
    for line in lines:
        for key, prefix in prefixes.items():
            position = line.find(prefix)
            if position < 0:
                continue
            payload = line[position + len(prefix) :].strip()
            try:
                parsed[key].append(json.loads(payload))
            except json.JSONDecodeError:
                parsed["invalid"].append({"kind": key, "line": line.rstrip("\n")})
            break
    return parsed
