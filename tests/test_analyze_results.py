"""Análisis local: lectura del esquema viejo y del actual, resumen, gráficas y PDF."""

import csv
from pathlib import Path

import pytest

pytest.importorskip("pandas")  # dependencia de desarrollo: no está en la imagen

from benchmark import results  # noqa: E402
from scripts import analyze_results  # noqa: E402

LEGACY = """label,repetition,elapsed_seconds,cpu_percent_mean,cpu_percent_max,ram_mb_mean,ram_mb_peak,gpu_util_percent_mean,gpu_memory_mb_peak
mediapipe_lite_cpu_2threads,1,9.8566,119.24,129.8,284.08,284.37,,
mediapipe_lite_cpu_2threads,2,9.6298,119.25,129.8,288.51,288.62,,
yolo26n_cpu_2threads,1,24.5553,198.27,209.6,392.09,401.09,,
"""


def _legacy_csv(tmp_path: Path) -> Path:
    path = tmp_path / "legacy.csv"
    path.write_text(LEGACY, encoding="utf-8")
    return path


def test_legacy_rows_are_mapped_to_the_current_schema(tmp_path: Path):
    rows = analyze_results.legacy_rows(_legacy_csv(tmp_path), "docker_2026-08-25", "marcha_p01", frames=334, fps=30.133)

    assert [r["model"] for r in rows] == ["mediapipe_lite", "mediapipe_lite", "yolo26n"]
    assert rows[0]["processing_s"] == pytest.approx(9.8566)
    assert rows[0]["ms_per_frame"] == pytest.approx(9856.6 / 334)
    assert rows[0]["realtime_factor"] == pytest.approx(9.8566 / (334 / 30.133))
    assert rows[2]["cpu_cores_used"] == pytest.approx(1.9827)
    assert rows[0]["threads_requested"] == "2"


def test_main_writes_summary_charts_and_pdf(tmp_path: Path):
    current = tmp_path / "rows.csv"
    sink = results.CsvSink(current)
    for rep in (1, 2, 3):
        sink.write({"environment": "render", "concurrency": 2, "model": "mediapipe_heavy", "video_id": "2EJ1",
                    "phase": "measured", "repetition": rep, "processing_s": 35 + rep, "job_s": 36, "ms_per_frame": 108,
                    "cpu_cores_used": 1.05, "ram_mb_peak": 520, "realtime_factor": 3.3})
    out = tmp_path / "analisis"

    code = analyze_results.main(["--csv", str(current), "--legacy-csv", str(_legacy_csv(tmp_path)), "--out", str(out)])

    assert code == 0
    summary = list(csv.DictReader((out / "resumen.csv").open(encoding="utf-8")))
    assert {(r["environment"], r["model"]) for r in summary} == {
        ("render", "mediapipe_heavy"), ("docker_2cpu_8gb_2026-08-25", "mediapipe_lite"), ("docker_2cpu_8gb_2026-08-25", "yolo26n")
    }
    assert (out / "ms_por_frame_render.png").exists()
    assert (out / "nucleos_render.png").exists()
    assert (out / "informe_benchmark.pdf").stat().st_size > 0


def test_main_without_measured_rows_returns_1(tmp_path: Path):
    empty = tmp_path / "rows.csv"
    results.CsvSink(empty).write({"model": "x", "phase": "warmup"})

    assert analyze_results.main(["--csv", str(empty), "--out", str(tmp_path / "out")]) == 1
