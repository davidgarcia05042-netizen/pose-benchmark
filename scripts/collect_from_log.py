"""
Reconstruye los resultados a partir de un log descargado de Render.

El disco de Render es efímero y todavía no hay un destino decidido para los
resultados (bloqueador #3), así que el log es el canal que sale del
contenedor: cada fila va como una línea `BENCHMARK_ROW {json}`. Esta
herramienta las extrae y deja lo mismo que habría quedado en `results/`:

    results/<run_id>/rows.csv        todas las filas, mismo esquema que el runner
    results/<run_id>/summary.csv     el resumen recalculado a partir de las filas
    results/<run_id>/run_info.json   el entorno, si el log lo trae

Uso (local, no va en la imagen):

    python -m scripts.collect_from_log render.log
    python -m scripts.collect_from_log render.log --results-dir results/render

Tolera lo que agrega Render al descargar (fecha y hora antes de cada línea),
líneas cortadas y logs con más de una corrida: separa por run_id.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark import results, stats


def collect(log_path: Path, results_dir: Path) -> dict[str, int]:
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        parsed = results.parse_log_lines(handle)

    by_run: dict[str, list[dict]] = {}
    for row in parsed["rows"]:
        by_run.setdefault(str(row.get("run_id")), []).append(row)
    run_infos = {str(info.get("run_id")): info for info in parsed["runs"]}

    counts = {}
    for run_id, rows in by_run.items():
        out_dir = results_dir / run_id
        rows_path = out_dir / "rows.csv"
        if rows_path.exists():
            # No mezclar con lo que ya haya: una segunda descarga del mismo log
            # duplicaría cada fila.
            print(f"{rows_path} ya existe: se omite {run_id}", file=sys.stderr)
            continue
        sink = results.CsvSink(rows_path)
        for row in rows:
            sink.write(row)
        results.write_summary_csv(out_dir / "summary.csv", stats.summarize(rows))
        if run_id in run_infos:
            (out_dir / "run_info.json").write_text(json.dumps(run_infos[run_id], indent=2, ensure_ascii=False), encoding="utf-8")
        counts[run_id] = len(rows)

    for error in parsed["errors"]:
        print(f"Sesión con error en el log: {error}", file=sys.stderr)
    if parsed["invalid"]:
        print(f"{len(parsed['invalid'])} línea(s) BENCHMARK_* no se pudieron leer (¿cortadas?)", file=sys.stderr)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.collect_from_log", description="Extrae las filas de un log de Render.")
    parser.add_argument("log", type=Path, help="archivo de log descargado de Render")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    counts = collect(args.log, args.results_dir)
    if not counts:
        print("No se encontró ninguna fila nueva en el log.", file=sys.stderr)
        return 1
    for run_id, count in counts.items():
        print(f"{run_id}: {count} filas -> {args.results_dir / run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
