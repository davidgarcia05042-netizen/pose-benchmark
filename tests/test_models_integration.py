"""
Los modelos REALES sobre un video sintético vertical: que carguen, que corran
con la rotación aplicada y que respeten el número de hilos pedido. No mide
rendimiento (el video no tiene a nadie): eso se hace con videos reales en el
hardware de Render.

Marcados `models`: `pytest -m "not models"` los excluye.
"""

from pathlib import Path

import pytest

from benchmark import results, session
from benchmark.models import MODELS, load_runner
from benchmark.synthetic import write_test_video

pytestmark = pytest.mark.models
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def phone_video(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("videos")
    return write_test_video(directory / "phone.mp4", frames=6, width=320, height=180, fps=30, rotation=90)


def test_mediapipe_lite_runs_end_to_end(phone_video, capsys):
    code = session.main(["--model", "mediapipe_lite", "--video", str(phone_video), "--video-id", "sintetico",
                         "--repetitions", "1", "--warmup-runs", "0", "--threads", "2"])
    rows = results.parse_log_lines(capsys.readouterr().out.splitlines())["rows"]

    assert code == 0
    assert rows[0]["frames_processed"] == 6
    assert rows[0]["video_decoded_resolution"] == "180x320"
    assert rows[0]["processing_s"] > 0


@pytest.mark.skipif(not (REPO / "yolo26n-pose.pt").is_file(), reason="faltan los pesos de yolo26n")
def test_yolo26n_keeps_the_requested_threads_after_the_first_inference(phone_video):
    """
    Ultralytics resetea torch.set_num_threads en la primera inferencia
    (hallazgo del 2026-08-25). El corredor tiene que dejarlo en lo pedido.
    """
    import cv2

    runner = load_runner(MODELS["yolo26n"], threads=2, weights_dir=REPO)
    try:
        cap = cv2.VideoCapture(str(phone_video))
        ok, frame = cap.read()
        cap.release()
        assert ok
        # Una sola inferencia: el reset ocurre DENTRO de ella, así que esto
        # comprueba el reemplazo de NUM_THREADS y no la reafirmación por frame.
        runner.infer(frame)
        assert runner.threads_effective() == 2
        runner.infer(frame)
        assert runner.threads_effective() == 2
    finally:
        runner.close()


def test_yolo_without_weights_fails_instead_of_downloading(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_runner(MODELS["yolo26n"], threads=2, weights_dir=tmp_path)
