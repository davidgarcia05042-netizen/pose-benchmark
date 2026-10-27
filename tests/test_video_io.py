"""
La orientación de los videos de celular (docs/DECISIONES_TECNICAS.md, "Ajustes heredados", 1).

Estos tests corren también dentro de la imagen de Linux (y el build ejecuta
la misma comprobación con `benchmark.environment --check`), que es donde el
comportamiento de OpenCV no estaba verificado.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmark.synthetic import write_test_video
from benchmark.video_io import VIDEO_ORIENTATION_TAG, open_video, probe_video


def test_open_video_applies_metadata_orientation(tmp_path: Path):
    """Equivalente a test_open_video_applies_metadata_orientation de entorno-pruebas."""
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    cap = open_video(str(path))
    try:
        assert cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) == 1
        ret, frame = cap.read()
        assert ret and frame.shape == (48, 64, 3)
    finally:
        cap.release()


def test_open_video_raises_for_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        open_video(tmp_path / "no_existe.mp4")


def test_the_synthetic_vertical_video_declares_its_rotation(vertical_video: Path):
    cap = cv2.VideoCapture(str(vertical_video))
    try:
        assert cap.get(cv2.CAP_PROP_ORIENTATION_META) == 90
    finally:
        cap.release()


def test_without_the_correction_the_person_arrives_lying_down(vertical_video: Path):
    """El problema que open_video corrige, con el valor fijado explícito (no el default)."""
    cap = cv2.VideoCapture(str(vertical_video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    assert ok and frame.shape[:2] == (48, 64)


def test_open_video_delivers_the_vertical_frame_turned_the_right_way(vertical_video: Path):
    cap = open_video(vertical_video)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    assert ok and frame.shape[:2] == (64, 48)
    # La franja blanca del borde izquierdo almacenado queda ARRIBA tras girar 90°.
    assert frame[0].mean() > 200
    assert frame[-1].mean() < 150


@pytest.mark.parametrize(
    ("rotation", "shape", "white_edge"),
    [(90, (64, 48), "top"), (180, (48, 64), "right"), (270, (64, 48), "bottom")],
)
def test_every_phone_rotation_is_applied(tmp_path: Path, rotation, shape, white_edge):
    path = write_test_video(tmp_path / f"r{rotation}.mp4", frames=3, width=64, height=48, rotation=rotation)
    cap = open_video(path)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    assert ok and frame.shape[:2] == shape
    edges = {"top": frame[0], "bottom": frame[-1], "left": frame[:, 0], "right": frame[:, -1]}
    assert edges[white_edge].mean() > 200


def test_probe_video_reports_stored_and_decoded_resolution(vertical_video: Path):
    info = probe_video(vertical_video)

    assert info.stored_resolution == "64x48"
    assert info.decoded_resolution == "48x64"
    assert info.rotation_meta_deg == 90
    assert info.frames_reported == 10
    assert info.fps == pytest.approx(10.0)
    assert info.duration_s == pytest.approx(1.0)
    assert info.file_name == "vertical.mp4"


def test_probe_video_of_a_video_without_rotation(plain_video: Path):
    info = probe_video(plain_video)

    assert info.rotation_meta_deg == 0
    assert info.stored_resolution == info.decoded_resolution == "64x48"


def test_probe_video_fields_become_prefixed_row_columns(vertical_video: Path):
    row = probe_video(vertical_video).as_row()

    assert row["video_rotation_meta_deg"] == 90
    assert row["video_decoded_resolution"] == "48x64"
    assert all(key.startswith("video_") for key in row)


def test_probe_video_rejects_a_file_without_frames(tmp_path: Path):
    broken = tmp_path / "roto.mp4"
    broken.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    with pytest.raises((FileNotFoundError, ValueError)):
        probe_video(broken)


def test_the_orientation_tag_matches_pose_experimental():
    # Mismo valor que "video_orientation" en el run_info.json de entorno-pruebas.
    assert VIDEO_ORIENTATION_TAG == "metadata_applied"
