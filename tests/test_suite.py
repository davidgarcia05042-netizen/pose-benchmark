"""Manifiestos de videos: rutas relativas, filtros y verificación por SHA-256."""

import hashlib
import json
from pathlib import Path

import pytest

from benchmark.suite import VideoEntry, check_entries, entries_from_paths, load_suite, sha256_of

REPO = Path(__file__).resolve().parents[1]


def _manifest(tmp_path: Path, videos: list[dict]) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"name": "prueba", "videos": videos}), encoding="utf-8")
    return path


def test_load_suite_resolves_relative_paths_on_any_os(tmp_path: Path):
    manifest = _manifest(tmp_path, [{"id": "A", "file": "carpeta/a.mp4", "sha256": "ab", "joint": "right_knee"}])

    meta, entries = load_suite(manifest, tmp_path / "videos")

    assert meta == {"name": "prueba"}
    assert entries == [VideoEntry("A", tmp_path / "videos" / "carpeta" / "a.mp4", "ab", "right_knee")]


def test_only_filters_by_id_and_keeps_the_manifest_order(tmp_path: Path):
    manifest = _manifest(tmp_path, [{"id": i, "file": f"{i}.mp4"} for i in ("A", "B", "C")])

    _, entries = load_suite(manifest, tmp_path, only=["C", "A"])

    assert [entry.video_id for entry in entries] == ["A", "C"]


def test_only_rejects_unknown_ids(tmp_path: Path):
    manifest = _manifest(tmp_path, [{"id": "A", "file": "a.mp4"}])

    with pytest.raises(ValueError, match="Z"):
        load_suite(manifest, tmp_path, only=["Z"])


def test_duplicate_ids_are_rejected(tmp_path: Path):
    manifest = _manifest(tmp_path, [{"id": "A", "file": "a.mp4"}, {"id": "A", "file": "b.mp4"}])

    with pytest.raises(ValueError):
        load_suite(manifest, tmp_path)


def test_check_entries_reports_missing_files_and_hash_mismatches(tmp_path: Path):
    good = tmp_path / "bueno.mp4"
    good.write_bytes(b"contenido")
    digest = hashlib.sha256(b"contenido").hexdigest()
    entries = [
        VideoEntry("ok", good, digest),
        VideoEntry("otro", good, "0" * 64),
        VideoEntry("falta", tmp_path / "no.mp4"),
    ]

    problems = check_entries(entries)

    assert len(problems) == 2
    assert any(p.startswith("otro:") and "SHA-256" in p for p in problems)
    assert any(p.startswith("falta:") for p in problems)
    assert check_entries(entries[:2], verify_hash=False) == []


def test_sha256_of_reads_in_chunks(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"a" * 3000)

    assert sha256_of(path, chunk_size=1024) == hashlib.sha256(b"a" * 3000).hexdigest()


def test_entries_from_paths_uses_the_file_stem(tmp_path: Path):
    assert entries_from_paths([tmp_path / "2EJ1.mp4"])[0].video_id == "2EJ1"
    with pytest.raises(ValueError):
        entries_from_paths([tmp_path / "a" / "x.mp4", tmp_path / "b" / "x.mp4"])


def test_the_toma_frontal_manifest_has_the_eight_requested_videos():
    meta, entries = load_suite(REPO / "suites" / "toma_frontal.json", Path("/data/videos"))

    assert meta["name"] == "toma_frontal"
    assert [e.video_id for e in entries] == ["2EJ1", "2EJ3", "6EJ3", "3EJ9", "3EJ10", "5EJ6", "5EJ9", "5EJ10"]
    assert all(e.sha256 and len(e.sha256) == 64 for e in entries)
    joints = {e.video_id: e.joint for e in entries}
    assert joints["2EJ1"] == "left_shoulder"
    assert joints["2EJ3"] == joints["6EJ3"] == "right_elbow"
    assert {joints[i] for i in ("3EJ9", "3EJ10", "5EJ6", "5EJ9", "5EJ10")} == {"right_knee"}


def test_the_manifest_mediapipe_points_match_each_joint():
    """Los índices de MediaPipe (convención del servicio) de cada articulación."""
    expected = {"left_shoulder": [23, 11, 13], "right_elbow": [12, 14, 16], "right_knee": [24, 26, 28]}
    manifest = json.loads((REPO / "suites" / "toma_frontal.json").read_text(encoding="utf-8"))

    for video in manifest["videos"]:
        assert video["mediapipe_points"] == expected[video["joint"]], video["id"]
