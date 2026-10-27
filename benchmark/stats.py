"""
Resumen estadístico de las repeticiones y juicio de aceptabilidad.

Solo biblioteca estándar: corre dentro de la imagen de Render (que no lleva
pandas) para imprimir el resumen al final de cada suite, y lo reutiliza
`scripts/analyze_results.py` en el portátil.

Metodología (detalle y referencias en docs/criterios_rendimiento.md):

- Se resume con MEDIANA y se acompaña de dispersión (desviación estándar,
  coeficiente de variación, intervalo de confianza del 95 % de la media con t
  de Student), no con un número suelto (Georges et al., 2007; Hoefler & Belli,
  2015).
- Las corridas de calentamiento no entran (Kalibera & Jones, 2013).
- Para juzgar contra los límites del servicio se usa el PEOR caso medido
  (máximo), no la mediana: un límite operativo se incumple con una sola
  corrida que se pase.

Los criterios de aceptabilidad salen de los límites operativos del servicio
(benchmark/config.py), no de un umbral genérico: ISO/IEC 25010 define la
eficiencia de desempeño como el grado en que tiempos y recursos "cumplen los
requisitos", y los requisitos acá son los del servicio desplegado.
"""

from __future__ import annotations

import math
from statistics import mean, median, stdev
from typing import Iterable

from benchmark import config

ACCEPTABLE = "aceptable"
CONDITIONAL = "condicionado"
NOT_ACCEPTABLE = "no_aceptable"

# Margen de RAM propuesto (no viene de la literatura: es una decisión del
# equipo, pendiente de confirmar). Dejar 20 % libre cubre lo que el muestreo no
# ve (buffers de decodificación, fragmentación) y el propio proceso de Celery.
RAM_SAFE_FRACTION = 0.8
# Variabilidad máxima para dar la medición por estable. Hoefler & Belli (2015)
# recomiendan repetir hasta que el intervalo de confianza sea lo bastante
# estrecho; el 5 % es nuestra elección de "lo bastante".
MAX_CV_PCT = 5.0
MIN_REPETITIONS = 3
# Por debajo de esto, el modelo casi no encontró a la persona y su costo no
# representa el trabajo real (MediaPipe sin detección se salta los landmarks).
MIN_DETECTION_RATE = 0.5

# t de Student bilateral al 95 % por grados de libertad (1..30).
_T95 = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
]


def t95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise ValueError("Hacen falta al menos 2 valores para un intervalo de confianza")
    if degrees_of_freedom <= len(_T95):
        return _T95[degrees_of_freedom - 1]
    return 2.000 if degrees_of_freedom <= 60 else 1.980 if degrees_of_freedom <= 120 else 1.960


def percentile(values: list[float], q: float) -> float | None:
    """Percentil `q` (0-100) con interpolación lineal, como numpy por defecto."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(v) for v in values if v is not None]
    n = len(data)
    if n == 0:
        return {"n": 0, "median": None, "mean": None, "std": None, "cv_pct": None, "min": None, "max": None,
                "ci95_low": None, "ci95_high": None}
    average = mean(data)
    spread = stdev(data) if n > 1 else 0.0
    half_width = t95(n - 1) * spread / math.sqrt(n) if n > 1 else None
    return {
        "n": n,
        "median": median(data),
        "mean": average,
        "std": spread,
        "cv_pct": spread / average * 100 if average else None,
        "min": min(data),
        "max": max(data),
        "ci95_low": average - half_width if half_width is not None else None,
        "ci95_high": average + half_width if half_width is not None else None,
    }


def _number(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def group_key(row: dict) -> tuple:
    return (row.get("environment"), int(float(row.get("concurrency") or 1)), row.get("model"), row.get("video_id"))


def summarize(rows: Iterable[dict]) -> list[dict]:
    """Una fila de resumen por (entorno, concurrencia, modelo, video), solo con repeticiones medidas."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("phase") == "measured":
            groups.setdefault(group_key(row), []).append(row)

    summaries = []
    for (environment, concurrency, model, video_id), members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        values = {key: [v for v in (_number(r, key) for r in members) if v is not None] for key in (
            "processing_s", "job_s", "model_init_s", "ms_per_frame", "fps_processed", "realtime_factor",
            "infer_ms_p90", "cpu_cores_used", "cpu_budget_pct", "cpu_percent_mean", "cgroup_throttled_pct",
            "ram_mb_peak", "ram_start_mb", "detection_rate", "video_fps", "video_duration_s",
            "frames_processed", "video_frames_reported",
        )}
        processing = describe(values["processing_s"])
        job = describe(values["job_s"])
        rtf = median(values["realtime_factor"]) if values["realtime_factor"] else None
        init = median(values["model_init_s"]) if values["model_init_s"] else 0.0
        ram_peak = max(values["ram_mb_peak"]) if values["ram_mb_peak"] else None
        ram_worker = ram_peak * config.SERVICE_WORKER_CONCURRENCY if ram_peak is not None else None
        # --max-frames recorta el video: los tiempos no son los del video
        # completo. El conteo del contenedor puede errar por un frame o dos.
        processed, reported = values["frames_processed"], values["video_frames_reported"]
        truncated = bool(processed and reported and median(processed) < 0.98 * median(reported))

        summary = {
            "environment": environment,
            "concurrency": concurrency,
            "model": model,
            "video_id": video_id,
            "video_duration_s": median(values["video_duration_s"]) if values["video_duration_s"] else None,
            "video_truncated": truncated,
            "n": processing["n"],
            "processing_s_median": processing["median"],
            "processing_s_mean": processing["mean"],
            "processing_s_std": processing["std"],
            "processing_s_cv_pct": processing["cv_pct"],
            "processing_s_ci95_low": processing["ci95_low"],
            "processing_s_ci95_high": processing["ci95_high"],
            "job_s_median": job["median"],
            "job_s_max": job["max"],
            "model_init_s_median": init,
            "ms_per_frame_median": median(values["ms_per_frame"]) if values["ms_per_frame"] else None,
            "fps_processed_median": median(values["fps_processed"]) if values["fps_processed"] else None,
            "infer_ms_p90_median": median(values["infer_ms_p90"]) if values["infer_ms_p90"] else None,
            "realtime_factor_median": rtf,
            "cpu_cores_used_median": median(values["cpu_cores_used"]) if values["cpu_cores_used"] else None,
            "cpu_budget_pct_median": median(values["cpu_budget_pct"]) if values["cpu_budget_pct"] else None,
            "cpu_percent_mean_median": median(values["cpu_percent_mean"]) if values["cpu_percent_mean"] else None,
            "cgroup_throttled_pct_median": median(values["cgroup_throttled_pct"]) if values["cgroup_throttled_pct"] else None,
            "ram_mb_peak_max": ram_peak,
            "ram_worker_estimate_mb": ram_worker,
            # Cuánto subió la RAM de partida entre jobs del mismo proceso: la
            # memoria que los modelos anteriores no devolvieron.
            "ram_drift_mb": max(values["ram_start_mb"]) - min(values["ram_start_mb"]) if values["ram_start_mb"] else None,
            "detection_rate_min": min(values["detection_rate"]) if values["detection_rate"] else None,
            # Cuánto video (segundos) cabe en cada límite, al ritmo medido. El
            # costo escala con los frames, no con los MB: MAX_VIDEO_SIZE_MB no
            # acota la duración, así que esto es lo que hay que mirar.
            "max_video_s_within_shutdown": (config.SERVICE_MAX_SHUTDOWN_DELAY_S - init) / rtf if rtf else None,
            "max_video_s_within_stuck_timeout": (config.SERVICE_STUCK_JOB_TIMEOUT_S - init) / rtf if rtf else None,
            "throughput_videos_per_hour": concurrency * 3600 / job["median"] if job["median"] else None,
        }
        summary.update(evaluate(summary))
        summaries.append(summary)
    return summaries


def evaluate(summary: dict) -> dict:
    """Veredictos por criterio. Ver docs/criterios_rendimiento.md."""
    worst_job = summary.get("job_s_max")
    if worst_job is None:
        time_verdict = None
    elif worst_job <= config.SERVICE_MAX_SHUTDOWN_DELAY_S:
        time_verdict = ACCEPTABLE
    elif worst_job <= config.SERVICE_STUCK_JOB_TIMEOUT_S:
        time_verdict = CONDITIONAL
    else:
        time_verdict = NOT_ACCEPTABLE

    ram = summary.get("ram_worker_estimate_mb")
    if ram is None:
        ram_verdict = None
    elif ram <= config.BUDGET_RAM_MB * RAM_SAFE_FRACTION:
        ram_verdict = ACCEPTABLE
    elif ram <= config.BUDGET_RAM_MB:
        ram_verdict = CONDITIONAL
    else:
        ram_verdict = NOT_ACCEPTABLE

    # Problemas de la medición en sí: si hay alguno, el número no sirve para decidir.
    validity = []
    if summary.get("n", 0) < MIN_REPETITIONS:
        validity.append(f"menos de {MIN_REPETITIONS} repeticiones")
    cv = summary.get("processing_s_cv_pct")
    if cv is not None and cv > MAX_CV_PCT:
        validity.append(f"CV {cv:.1f} % > {MAX_CV_PCT:g} %: repetir con más repeticiones")
    detection = summary.get("detection_rate_min")
    if detection is not None and detection < MIN_DETECTION_RATE:
        validity.append(f"detección {detection:.0%}: el costo no representa trabajo real")
    # Contexto: la medición vale, pero hay que leerla sabiendo esto.
    context = []
    if summary.get("video_truncated"):
        context.append("video recortado con --max-frames: el veredicto de tiempo no es el del video completo")
    if summary.get("concurrency") != config.SERVICE_WORKER_CONCURRENCY:
        context.append(
            f"medido con concurrencia {summary.get('concurrency')}; el worker corre "
            f"{config.SERVICE_WORKER_CONCURRENCY} videos a la vez"
        )

    return {
        "verdict_time": time_verdict,
        "verdict_ram": ram_verdict,
        "measurement_valid": not validity,
        "warnings": "; ".join(validity + context),
    }


def format_summary(summary: dict) -> str:
    def fmt(value, pattern):
        return pattern.format(value) if value is not None else "n/d"

    return (
        f"{summary['model']:16s} {str(summary['video_id']):10s} conc={summary['concurrency']} n={summary['n']}  "
        f"proceso {fmt(summary['processing_s_median'], '{:.1f}')} s (CV {fmt(summary['processing_s_cv_pct'], '{:.1f}')} %)  "
        f"{fmt(summary['ms_per_frame_median'], '{:.0f}')} ms/frame  x{fmt(summary['realtime_factor_median'], '{:.2f}')} tiempo real  "
        f"{fmt(summary['cpu_cores_used_median'], '{:.2f}')} núcleos  RAM pico {fmt(summary['ram_mb_peak_max'], '{:.0f}')} MB  "
        f"-> tiempo: {summary['verdict_time']}, RAM: {summary['verdict_ram']}"
        + (f"  [{summary['warnings']}]" if summary["warnings"] else "")
    )
