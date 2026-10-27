"""
La sesión de medición (un modelo, un video, un proceso), con el modelo
sustituido por un doble: acá se prueba el protocolo y las cuentas, no el
rendimiento real.
"""

import json
from pathlib import Path

import pytest

import benchmark.models
from benchmark import results, session
from benchmark.cgroup import Cgroup, CgroupLimits
from benchmark.video_io import VIDEO_ORIENTATION_TAG

NO_CGROUP = Cgroup(None, None, None)


class _FixedCgroup:
    def __init__(self, cores):
        self._cores = cores

    def limits(self):
        return CgroupLimits("v2", self._cores, None)


def test_resolve_threads(monkeypatch):
    import os

    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert session.resolve_threads("auto", _FixedCgroup(2.0)) == (2, "cgroup_quota")
    assert session.resolve_threads("library", _FixedCgroup(2.0)) == (None, "library_default")
    assert session.resolve_threads("3", _FixedCgroup(2.0)) == (3, "explicit")
    with pytest.raises(ValueError):
        session.resolve_threads("0", _FixedCgroup(2.0))


def test_auto_threads_are_shared_between_concurrent_sessions(monkeypatch):
    """Dos sesiones en 2 núcleos: 1 hilo cada una, no 2 (evita sobresuscribir)."""
    import os

    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert session.resolve_threads("auto", _FixedCgroup(2.0), concurrency=2) == (1, "cgroup_quota/2")
    assert session.resolve_threads("auto", _FixedCgroup(2.0), concurrency=3) == (1, "cgroup_quota/3")
    assert session.resolve_threads("4", _FixedCgroup(2.0), concurrency=2) == (4, "explicit")


def test_measure_repetition_processes_every_frame_of_the_rotated_video(vertical_video: Path, fake_runner_class):
    measurement = session.measure_repetition(fake_runner_class, vertical_video, None, NO_CGROUP, 0.05)
    runner = fake_runner_class.instances[0]

    assert len(measurement["infer_ms"]) == len(measurement["decode_ms"]) == 10
    assert runner.frames == 10
    assert runner.closed
    assert measurement["detected"] == 5  # el doble "detecta" en frames pares
    assert measurement["model_init_s"] >= 0
    assert measurement["monitor"].wall_s > 0


def test_max_frames_cuts_the_video(vertical_video: Path, fake_runner_class):
    measurement = session.measure_repetition(fake_runner_class, vertical_video, 4, NO_CGROUP, 0.05)

    assert len(measurement["infer_ms"]) == 4


def test_the_model_is_closed_even_when_it_fails(vertical_video: Path, fake_runner_class):
    with pytest.raises(RuntimeError):
        session.measure_repetition(lambda: fake_runner_class(fail_on_frame=3), vertical_video, None, NO_CGROUP, 0.05)

    assert fake_runner_class.instances[0].closed


def test_build_row_derives_the_rates(vertical_video: Path, fake_runner_class):
    measurement = session.measure_repetition(fake_runner_class, vertical_video, None, NO_CGROUP, 0.05)
    row = session.build_row(measurement, {"video_fps": 10.0, "model": "x"})

    processing = measurement["monitor"].wall_s
    assert row["frames_processed"] == 10
    assert row["ms_per_frame"] == pytest.approx(round(processing * 100, 4), abs=1e-3)
    # 10 frames a 10 fps = 1 s de video.
    assert row["realtime_factor"] == pytest.approx(round(processing, 4), abs=1e-3)
    assert row["job_s"] == pytest.approx(row["model_init_s"] + row["processing_s"], abs=1e-3)
    assert row["detection_rate"] == 0.5
    assert row["infer_ms_p50"] <= row["infer_ms_p90"] <= row["infer_ms_max"]
    assert "wall_s" not in row


def _run_main(monkeypatch, capsys, fake_runner_class, video: Path, *extra: str) -> tuple[int, list[dict], str]:
    monkeypatch.setattr(benchmark.models, "load_runner", lambda spec, threads, weights_dir: fake_runner_class(spec))
    code = session.main(["--model", "mediapipe_lite", "--video", str(video), "--video-id", "V1", "--run-id", "r1",
                         "--threads", "2", "--sample-interval", "0.05", *extra])
    out = capsys.readouterr().out
    parsed = results.parse_log_lines(out.splitlines())
    return code, parsed["rows"], out


def test_main_emits_one_row_per_warmup_and_repetition(monkeypatch, capsys, vertical_video, fake_runner_class):
    code, rows, _ = _run_main(monkeypatch, capsys, fake_runner_class, vertical_video, "--repetitions", "2", "--warmup-runs", "1")

    assert code == 0
    assert [(r["phase"], r["repetition"]) for r in rows] == [("warmup", 1), ("measured", 1), ("measured", 2)]
    # Un modelo nuevo por repetición, como un job del servicio.
    assert len(fake_runner_class.instances) == 3
    row = rows[-1]
    assert row["video_orientation"] == VIDEO_ORIENTATION_TAG
    assert row["video_rotation_meta_deg"] == 90
    assert row["video_decoded_resolution"] == "48x64"
    assert row["threads_requested"] == 2
    assert row["run_id"] == "r1" and row["video_id"] == "V1" and row["model"] == "mediapipe_lite"


def test_every_emitted_field_is_a_csv_column_and_vice_versa(monkeypatch, capsys, vertical_video, fake_runner_class):
    """Si una columna nueva no se agrega a ROW_COLUMNS, el CSV la perdería en silencio."""
    _, rows, _ = _run_main(monkeypatch, capsys, fake_runner_class, vertical_video, "--repetitions", "1", "--warmup-runs", "0")

    assert set(rows[0]) == set(results.ROW_COLUMNS)


def test_a_failing_model_emits_an_error_line_and_exits_1(monkeypatch, capsys, vertical_video, fake_runner_class):
    monkeypatch.setattr(benchmark.models, "load_runner", lambda spec, threads, weights_dir: fake_runner_class(spec, fail_on_frame=2))
    code = session.main(["--model", "yolo26n", "--video", str(vertical_video), "--video-id", "V1", "--repetitions", "1", "--warmup-runs", "0"])
    out = capsys.readouterr().out

    errors = results.parse_log_lines(out.splitlines())["errors"]
    assert code == 1
    assert errors and "fallo simulado" in errors[0]["error"]


def test_a_missing_video_is_an_error_line_not_a_crash(monkeypatch, capsys, tmp_path, fake_runner_class):
    code, rows, out = _run_main(monkeypatch, capsys, fake_runner_class, tmp_path / "no.mp4")

    assert code == 1
    assert rows == []
    assert "BENCHMARK_ERROR" in out
    assert json.loads(out.split("BENCHMARK_ERROR ", 1)[1].splitlines()[0])["video_id"] == "V1"
