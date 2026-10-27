"""
El orquestador. Las sesiones se sustituyen por un proceso Python mínimo que
imprime filas: se prueba el reparto, la recolección y los códigos de salida,
sin cargar modelos.
"""

import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from benchmark import results
from benchmark.suite import VideoEntry
from scripts import run_benchmark


def _entries(n=2):
    return [VideoEntry(f"V{i}", Path(f"v{i}.mp4")) for i in range(1, n + 1)]


def test_plan_without_seed_is_video_by_video_model_by_model():
    plan = run_benchmark.build_plan(_entries(2), ["a", "b"], trials=1, shuffle_seed=None)

    assert [(e.video_id, m) for _, e, m in plan] == [("V1", "a"), ("V1", "b"), ("V2", "a"), ("V2", "b")]


def test_plan_with_seed_shuffles_each_trial_reproducibly():
    first = run_benchmark.build_plan(_entries(3), ["a", "b", "c"], trials=2, shuffle_seed=7)
    again = run_benchmark.build_plan(_entries(3), ["a", "b", "c"], trials=2, shuffle_seed=7)
    trial_1 = [(e.video_id, m) for t, e, m in first if t == 1]
    trial_2 = [(e.video_id, m) for t, e, m in first if t == 2]

    assert [(t, e.video_id, m) for t, e, m in first] == [(t, e.video_id, m) for t, e, m in again]
    assert sorted(trial_1) == sorted(trial_2)  # mismas combinaciones...
    assert trial_1 != trial_2  # ...en otro orden
    assert len(first) == 18


def test_session_command_carries_hash_and_frame_limit(tmp_path):
    args = Namespace(repetitions=3, warmup_runs=1, threads="auto", weights_dir=tmp_path, sample_interval=0.1,
                     concurrency=2, max_frames=30)
    command = run_benchmark.session_command(args, VideoEntry("V1", Path("v.mp4"), "ab" * 32), "yolo26n", "r1", 1, 1)

    assert command[:3] == [sys.executable, "-m", "benchmark.session"]
    assert command[command.index("--video-sha256") + 1] == "ab" * 32
    assert command[command.index("--max-frames") + 1] == "30"
    assert command[command.index("--parallel-index") + 1] == "1"
    assert command[command.index("--concurrency") + 1] == "2"


def _fake_session(exit_code=0, processing_s=10.0):
    """Un comando que imprime una fila medida como lo haría benchmark.session."""
    def factory(args, entry, model, run_id, trial, parallel_index):
        row = {"run_id": run_id, "model": model, "video_id": entry.video_id, "phase": "measured", "repetition": 1,
               "concurrency": args.concurrency, "parallel_index": parallel_index, "environment": "test",
               "processing_s": processing_s, "job_s": processing_s, "model_init_s": 0.0}
        code = f"print('otra línea del modelo'); print({results.ROW_PREFIX!r} + {json.dumps(json.dumps(row))}); raise SystemExit({exit_code})"
        return [sys.executable, "-c", code]
    return factory


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"no hace falta que sea un video: la sesion es un doble")
    return path


def test_main_collects_rows_from_parallel_sessions(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_benchmark, "session_command", _fake_session())
    code = run_benchmark.main(["--video", str(_video(tmp_path)), "--models", "mediapipe_lite", "mediapipe_heavy",
                               "--concurrency", "2", "--results-dir", str(tmp_path / "res"), "--run-id", "r1"])
    out = capsys.readouterr().out

    rows = list(csv.DictReader((tmp_path / "res" / "r1" / "rows.csv").open(encoding="utf-8")))
    assert code == 0
    assert len(rows) == 4  # 2 modelos x 2 procesos a la vez
    assert sorted(r["parallel_index"] for r in rows) == ["0", "0", "1", "1"]
    assert "otra línea del modelo" in out  # lo que imprime la sesión llega al log
    assert (tmp_path / "res" / "r1" / "summary.csv").exists()
    info = json.loads((tmp_path / "res" / "r1" / "run_info.json").read_text(encoding="utf-8"))
    assert info["run_id"] == "r1" and info["models"] == ["mediapipe_lite", "mediapipe_heavy"]
    assert "BENCHMARK_RUN " in out and "BENCHMARK_SUMMARY " in out
    # La lista de paquetes va en run_info.json, no en la línea del log.
    assert "packages" in info
    assert "packages" not in results.parse_log_lines(out.splitlines())["runs"][0]


def test_a_failing_session_makes_the_run_exit_1_but_the_rest_continues(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_benchmark, "session_command", _fake_session(exit_code=3))
    code = run_benchmark.main(["--video", str(_video(tmp_path)), "--models", "mediapipe_lite", "mediapipe_heavy",
                               "--results-dir", str(tmp_path / "res"), "--run-id", "r1"])
    err = capsys.readouterr().err

    assert code == 1
    assert err.count("salió con 3") == 2  # los dos modelos se intentaron


def test_missing_videos_stop_before_measuring(tmp_path, capsys):
    code = run_benchmark.main(["--video", str(tmp_path / "no.mp4"), "--results-dir", str(tmp_path / "res")])

    assert code == 2
    assert "no existe" in capsys.readouterr().err
    assert not (tmp_path / "res").exists()


def test_missing_yolo_weights_stop_before_measuring(tmp_path, capsys):
    code = run_benchmark.main(["--video", str(_video(tmp_path)), "--models", "yolo26n",
                               "--weights-dir", str(tmp_path), "--results-dir", str(tmp_path / "res")])

    assert code == 2
    assert "pesos" in capsys.readouterr().err


def test_strict_versions_refuses_a_non_comparable_environment(tmp_path, monkeypatch):
    from benchmark import environment

    real_snapshot = environment.snapshot
    monkeypatch.setattr(environment, "snapshot", lambda **kw: dict(real_snapshot(**kw), version_mismatches=["numpy 2.0"]))
    monkeypatch.setattr(run_benchmark, "session_command", _fake_session())

    code = run_benchmark.main(["--video", str(_video(tmp_path)), "--models", "mediapipe_lite", "--strict-versions",
                               "--results-dir", str(tmp_path / "res")])

    assert code == 3


def test_parallel_sessions_share_one_deadline():
    """Dos sesiones colgadas se cortan juntas, no una después de la otra."""
    import os
    import time

    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
    started = time.monotonic()
    codes = run_benchmark.run_sessions([sleeper, sleeper], lambda line: None, 1.0, dict(os.environ))

    assert codes == [None, None]
    assert time.monotonic() - started < 10


def test_suite_requires_videos_dir():
    with pytest.raises(SystemExit):
        run_benchmark.main(["--suite", "suites/toma_frontal.json"])


def test_environment_label_reaches_the_sessions(tmp_path, monkeypatch):
    seen = {}

    def fake_run(commands, on_line, timeout_s, env):
        seen.update(env)
        return [0 for _ in commands]

    monkeypatch.setattr(run_benchmark, "run_sessions", fake_run)
    monkeypatch.delenv("POSE_BENCHMARK_ENVIRONMENT", raising=False)
    run_benchmark.main(["--video", str(_video(tmp_path)), "--models", "mediapipe_lite", "--environment", "docker_2cpu_8gb",
                        "--results-dir", str(tmp_path / "res")])

    assert seen["POSE_BENCHMARK_ENVIRONMENT"] == "docker_2cpu_8gb"


# --- Videos desde Drive y resultados hacia Drive (rclone sustituido por un doble) ---


def _remote_suite(tmp_path: Path, content: bytes = b"video de prueba") -> tuple[Path, Path]:
    """Un manifiesto de un video y la carpeta que hace de Drive, con ese video adentro."""
    import hashlib

    drive = tmp_path / "drive"
    video = drive / "TOMA FRONTAL" / "V1" / "V1.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(content)
    manifest = tmp_path / "suite.json"
    manifest.write_text(json.dumps({"name": "prueba", "videos": [
        {"id": "V1", "file": "V1/V1.mp4", "sha256": hashlib.sha256(b"video de prueba").hexdigest()}]}), encoding="utf-8")
    return manifest, drive


def _use_fake_rclone(monkeypatch, drive: Path, fail_on: str | None = None):
    from benchmark import storage
    from tests.test_storage import FakeRclone

    fake = FakeRclone(drive, fail_on=fail_on)
    real_class = storage.Rclone
    monkeypatch.setattr(storage, "Rclone", lambda: real_class("rclone", run=fake))
    return fake


def test_videos_come_from_the_remote_and_results_go_back(tmp_path, monkeypatch, capsys):
    manifest, drive = _remote_suite(tmp_path)
    _use_fake_rclone(monkeypatch, drive)
    monkeypatch.setattr(run_benchmark, "session_command", _fake_session())

    code = run_benchmark.main(["--suite", str(manifest), "--videos-remote", "fake:TOMA FRONTAL",
                               "--results-remote", "fake:resultados", "--models", "mediapipe_lite",
                               "--results-dir", str(tmp_path / "res"), "--run-id", "r1"])
    out = capsys.readouterr().out

    assert code == 0
    assert "no entra en la medición" in out
    assert sorted(p.name for p in (drive / "resultados" / "r1").iterdir()) == ["rows.csv", "run_info.json", "summary.csv"]


def test_a_video_changed_in_the_remote_stops_the_run(tmp_path, monkeypatch, capsys):
    manifest, drive = _remote_suite(tmp_path, content=b"otro contenido")
    _use_fake_rclone(monkeypatch, drive)
    monkeypatch.setattr(run_benchmark, "session_command", _fake_session())

    code = run_benchmark.main(["--suite", str(manifest), "--videos-remote", "fake:TOMA FRONTAL",
                               "--results-dir", str(tmp_path / "res")])

    assert code == 2
    assert "SHA-256" in capsys.readouterr().err


def test_an_unwritable_results_remote_stops_before_measuring(tmp_path, monkeypatch, capsys):
    manifest, drive = _remote_suite(tmp_path)
    _use_fake_rclone(monkeypatch, drive, fail_on="mkdir")
    monkeypatch.setattr(run_benchmark, "run_sessions", lambda *a: pytest.fail("no debía medir"))

    code = run_benchmark.main(["--suite", str(manifest), "--videos-dir", str(drive / "TOMA FRONTAL"),
                               "--results-remote", "fake:resultados", "--results-dir", str(tmp_path / "res")])

    assert code == 2
    assert "rclone mkdir" in capsys.readouterr().err


def test_a_failed_upload_exits_4_and_points_to_the_log(tmp_path, monkeypatch, capsys):
    manifest, drive = _remote_suite(tmp_path)
    _use_fake_rclone(monkeypatch, drive, fail_on="copy")
    monkeypatch.setattr(run_benchmark, "session_command", _fake_session())

    code = run_benchmark.main(["--suite", str(manifest), "--videos-dir", str(drive / "TOMA FRONTAL"),
                               "--results-remote", "fake:resultados", "--models", "mediapipe_lite",
                               "--results-dir", str(tmp_path / "res"), "--run-id", "r1"])

    assert code == 4
    assert "collect_from_log" in capsys.readouterr().err
    assert (tmp_path / "res" / "r1" / "rows.csv").exists()


def test_videos_remote_requires_a_suite(tmp_path):
    with pytest.raises(SystemExit):
        run_benchmark.main(["--video", str(_video(tmp_path)), "--videos-remote", "fake:x"])


def test_storage_credentials_do_not_reach_the_sessions(tmp_path, monkeypatch):
    seen = {}

    def fake_run(commands, on_line, timeout_s, env):
        seen.update(env)
        return [0 for _ in commands]

    monkeypatch.setattr(run_benchmark, "run_sessions", fake_run)
    monkeypatch.setenv("RCLONE_CONFIG_GDRIVE_TOKEN", "secreto")
    run_benchmark.main(["--video", str(_video(tmp_path)), "--models", "mediapipe_lite", "--results-dir", str(tmp_path / "res")])

    assert seen and "RCLONE_CONFIG_GDRIVE_TOKEN" not in seen
