"""
Análisis LOCAL de resultados (no va en la imagen): resumen estadístico,
veredicto contra los límites del servicio, gráficas y un PDF.

Lee dos esquemas y los pone uno al lado del otro, con su origen marcado:

- el actual (`results/<run_id>/rows.csv`, de scripts/run_benchmark.py o
  scripts/collect_from_log.py), y
- el de la primera fase (`results/*_docker_2cpu_8gb.csv`, columnas `label`,
  `elapsed_seconds`...), que es la referencia contra la que se compara Render.
  Esas filas no traen entorno, video ni frames: se indican con --legacy-*.

Uso:
    python -m scripts.analyze_results --csv results/<run_id>/rows.csv --out results/<run_id>/analisis

    # Render al lado de la referencia de Docker (Marcha P01, 334 frames):
    python -m scripts.analyze_results --csv results/<run_id>/rows.csv \
        --legacy-csv results/mediapipe_docker_2cpu_8gb.csv results/yolo26n_docker_2cpu_8gb.csv \
        --out results/comparacion

Genera en --out:
    resumen.csv               una fila por entorno, modelo y video (benchmark/stats.py)
    ms_por_frame_<env>.png    costo por frame, comparable entre videos de distinta duración
    nucleos_<env>.png         núcleos usados frente al presupuesto de 2
    ram_<env>.png             RAM pico por proceso frente al margen por proceso
    informe_benchmark.pdf     tablas + gráficas
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from benchmark import config, stats
from scripts.charts import GRID, TEXT_PRIMARY, video_bars

_LEGACY_LABEL = re.compile(r"^(?P<model>.+?)_(?P<device>cpu|cuda)(?:_(?P<threads>\d+)threads)?$")


def load_rows(path: Path) -> list[dict]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame.to_dict("records")


def legacy_rows(path: Path, environment: str, video_id: str, frames: int | None, fps: float | None) -> list[dict]:
    """Convierte un CSV de la primera fase al esquema actual (lo que se pueda)."""
    duration = frames / fps if frames and fps else None
    rows = []
    for record in pd.read_csv(path).to_dict("records"):
        match = _LEGACY_LABEL.match(str(record["label"]))
        processing = float(record["elapsed_seconds"])
        rows.append(
            {
                "environment": environment,
                "model": match.group("model") if match else record["label"],
                "video_id": video_id,
                "phase": "measured",
                "repetition": record["repetition"],
                "concurrency": 1,
                "processing_s": processing,
                # La primera fase cargaba el modelo fuera de la medición.
                "job_s": processing,
                "model_init_s": 0.0,
                "ms_per_frame": processing * 1000 / frames if frames else None,
                "frames_processed": frames,
                "video_frames_reported": frames,
                "video_fps": fps,
                "video_duration_s": duration,
                "realtime_factor": processing / duration if duration else None,
                # Aproximación: la primera fase no guardaba segundos de CPU,
                # solo el promedio de las muestras de psutil.
                "cpu_cores_used": float(record["cpu_percent_mean"]) / 100,
                "cpu_percent_mean": record["cpu_percent_mean"],
                "ram_mb_peak": record["ram_mb_peak"],
                "threads_requested": match.group("threads") if match else None,
            }
        )
    return rows


def _table_page(pdf: PdfPages, title: str, frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=13, loc="left", color=TEXT_PRIMARY)
    table = ax.table(cellText=frame.astype(str).values, colLabels=list(frame.columns), loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.4)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_text_props(weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.analyze_results")
    parser.add_argument("--csv", nargs="*", type=Path, default=[], help="rows.csv del esquema actual")
    parser.add_argument("--legacy-csv", nargs="*", type=Path, default=[], help="CSV de la primera fase")
    parser.add_argument("--legacy-environment", default="docker_2cpu_8gb_2026-08-25")
    parser.add_argument("--legacy-video-id", default="marcha_p01")
    # Marcha P01, medido con benchmark.video_io.probe_video el 2026-09-28.
    parser.add_argument("--legacy-frames", type=int, default=334, help="frames del video de la primera fase")
    parser.add_argument("--legacy-fps", type=float, default=30.133, help="fps del video de la primera fase")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.csv and not args.legacy_csv:
        parser.error("hace falta al menos un --csv o --legacy-csv")

    rows: list[dict] = []
    for path in args.csv:
        rows += load_rows(path)
    for path in args.legacy_csv:
        rows += legacy_rows(path, args.legacy_environment, args.legacy_video_id, args.legacy_frames, args.legacy_fps)

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = stats.summarize(rows)
    if not summaries:
        print("No hay repeticiones medidas (phase=measured) en los CSV.")
        return 1
    for record in summaries:
        print(stats.format_summary(record))
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out / "resumen.csv", index=False)

    share = config.BUDGET_RAM_MB * stats.RAM_SAFE_FRACTION / config.SERVICE_WORKER_CONCURRENCY
    charts = []
    for environment, group in summary.groupby("environment"):
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(environment))
        specs = [
            ("ms_per_frame_median", f"ms_por_frame_{tag}.png", "Costo por frame", "ms por frame (mediana)", None, 0),
            ("cpu_cores_used_median", f"nucleos_{tag}.png", "Núcleos usados", "núcleos (mediana)",
             (config.BUDGET_CPU_CORES, f"presupuesto {config.RENDER_PLAN}: {config.BUDGET_CPU_CORES} núcleos"), 2),
            ("ram_mb_peak_max", f"ram_{tag}.png", "RAM pico por proceso", "MB (máximo)",
             (share, f"margen por proceso con {config.SERVICE_WORKER_CONCURRENCY} en paralelo: {share:.0f} MB"), 0),
        ]
        for column, name, title, ylabel, reference, decimals in specs:
            if group[column].notna().any():
                path = args.out / name
                video_bars(group, column, list(dict.fromkeys(group["video_id"])), f"{title} — {environment}", ylabel,
                           path, reference, decimals)
                print(f"  -> {path}")
                charts.append(path)

    table_columns = ["environment", "model", "video_id", "n", "processing_s_median", "processing_s_cv_pct",
                     "ms_per_frame_median", "realtime_factor_median", "cpu_cores_used_median", "ram_mb_peak_max",
                     "max_video_s_within_shutdown", "verdict_time", "verdict_ram", "measurement_valid"]
    with PdfPages(args.out / "informe_benchmark.pdf") as pdf:
        table = summary[[c for c in table_columns if c in summary.columns]].round(2)
        for start in range(0, len(table), 28):
            _table_page(pdf, "Resumen por entorno, modelo y video", table.iloc[start:start + 28])
        for chart in charts:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(plt.imread(chart))
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
    print(f"  -> {args.out / 'informe_benchmark.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
