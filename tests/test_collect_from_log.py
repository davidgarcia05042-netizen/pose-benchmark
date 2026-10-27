"""Recuperar los resultados desde un log descargado de Render."""

import csv
import json
from pathlib import Path

from benchmark import results
from scripts import collect_from_log


def _log(tmp_path: Path) -> Path:
    lines = [f"2026-09-29T10:00:00Z {results.RUN_PREFIX}{json.dumps({'run_id': 'r1', 'environment': 'render'})}"]
    for rep in (1, 2, 3):
        row = {"run_id": "r1", "environment": "render", "concurrency": 2, "model": "mediapipe_lite", "video_id": "2EJ1",
               "phase": "measured", "repetition": rep, "processing_s": 12 + rep / 10, "job_s": 12.5, "ram_mb_peak": 290}
        lines.append(f"2026-09-29T10:00:0{rep}Z {results.ROW_PREFIX}{json.dumps(row)}")
    lines.append("2026-09-29T10:00:05Z [mediapipe_lite | 2EJ1] medición 3/3: 12.3 s")
    path = tmp_path / "render.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_collect_rebuilds_rows_summary_and_run_info(tmp_path: Path):
    counts = collect_from_log.collect(_log(tmp_path), tmp_path / "results")

    run_dir = tmp_path / "results" / "r1"
    rows = list(csv.DictReader((run_dir / "rows.csv").open(encoding="utf-8")))
    summary = list(csv.DictReader((run_dir / "summary.csv").open(encoding="utf-8")))
    assert counts == {"r1": 3}
    assert [r["repetition"] for r in rows] == ["1", "2", "3"]
    assert summary[0]["n"] == "3" and summary[0]["model"] == "mediapipe_lite"
    assert json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))["environment"] == "render"


def test_collecting_the_same_log_twice_does_not_duplicate_rows(tmp_path: Path):
    log = _log(tmp_path)
    collect_from_log.collect(log, tmp_path / "results")

    assert collect_from_log.collect(log, tmp_path / "results") == {}
    assert len(list(csv.DictReader((tmp_path / "results" / "r1" / "rows.csv").open(encoding="utf-8")))) == 3


def test_main_returns_1_when_the_log_has_no_rows(tmp_path: Path):
    empty = tmp_path / "vacio.log"
    empty.write_text("nada que ver acá\n", encoding="utf-8")

    assert collect_from_log.main([str(empty), "--results-dir", str(tmp_path / "results")]) == 1
