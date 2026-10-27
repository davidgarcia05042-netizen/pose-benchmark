"""Cómo sale una fila del contenedor: log (canal durable) y CSV."""

import csv
import io
import json
from pathlib import Path

import pytest

from benchmark import results


def test_emit_writes_one_parseable_line():
    stream = io.StringIO()
    results.emit(results.ROW_PREFIX, {"model": "yolo26n", "nota": "línea\ncon salto", "x": 1.5}, stream)

    text = stream.getvalue()
    assert text.count("\n") == 1  # un salto dentro del valor no parte la línea
    assert text.startswith(results.ROW_PREFIX)
    assert json.loads(text[len(results.ROW_PREFIX):]) == {"model": "yolo26n", "nota": "línea\ncon salto", "x": 1.5}


def test_parse_log_lines_tolerates_render_timestamps_and_broken_lines():
    lines = [
        "2026-09-29T10:00:00.123Z BENCHMARK_RUN {\"run_id\": \"r1\"}\n",
        "2026-09-29T10:00:01.000Z [mediapipe_lite | 2EJ1] medición 1/3: 12.0 s\n",
        "2026-09-29T10:00:02.000Z BENCHMARK_ROW {\"run_id\": \"r1\", \"model\": \"mediapipe_lite\"}\n",
        "BENCHMARK_ROW {\"run_id\": \"r1\", \"model\": \"yol",  # cortada
        "BENCHMARK_ERROR {\"model\": \"yolo26n\", \"error\": \"x\"}",
        "BENCHMARK_SUMMARY {\"model\": \"mediapipe_lite\"}",
    ]

    parsed = results.parse_log_lines(lines)

    assert parsed["runs"] == [{"run_id": "r1"}]
    assert parsed["rows"] == [{"run_id": "r1", "model": "mediapipe_lite"}]
    assert parsed["errors"] == [{"model": "yolo26n", "error": "x"}]
    assert parsed["summaries"] == [{"model": "mediapipe_lite"}]
    assert len(parsed["invalid"]) == 1


def test_csv_sink_writes_the_header_once_and_appends(tmp_path: Path):
    path = tmp_path / "run" / "rows.csv"
    sink = results.CsvSink(path)
    sink.write({"run_id": "r1", "model": "a", "processing_s": 1.234567, "no_es_columna": "x"})
    sink.write({"run_id": "r1", "model": "b", "processing_s": None})

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [row["model"] for row in rows] == ["a", "b"]
    assert rows[0]["processing_s"] == "1.2346"
    assert rows[1]["processing_s"] == ""
    assert list(rows[0]) == results.ROW_COLUMNS


def test_csv_sink_refuses_a_file_with_another_schema(tmp_path: Path):
    path = tmp_path / "rows.csv"
    path.write_text("label,repetition,elapsed_seconds\nx,1,2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        results.CsvSink(path)


def test_csv_sink_reopens_a_file_with_the_same_schema(tmp_path: Path):
    path = tmp_path / "rows.csv"
    results.CsvSink(path).write({"model": "a"})
    results.CsvSink(path).write({"model": "b"})

    assert len(list(csv.DictReader(path.open(encoding="utf-8")))) == 2


def test_legacy_columns_keep_their_names():
    """Para poner lo nuevo al lado de las tablas de Docker del 2026-08-25."""
    for legacy in ("cpu_percent_mean", "cpu_percent_max", "ram_mb_mean", "ram_mb_peak"):
        assert legacy in results.ROW_COLUMNS


def test_every_row_says_whether_orientation_was_applied():
    assert "video_orientation" in results.ROW_COLUMNS
    assert "video_rotation_meta_deg" in results.ROW_COLUMNS


def test_write_summary_csv(tmp_path: Path):
    path = tmp_path / "summary.csv"
    results.write_summary_csv(path, [{"model": "a", "processing_s_median": 1.23456789, "verdict_time": None}])

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows == [{"model": "a", "processing_s_median": "1.2346", "verdict_time": ""}]


def test_write_summary_csv_with_nothing_writes_nothing(tmp_path: Path):
    results.write_summary_csv(tmp_path / "summary.csv", [])

    assert not (tmp_path / "summary.csv").exists()
