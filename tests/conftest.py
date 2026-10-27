"""
Fixtures compartidas. Todos los videos son sintéticos (benchmark/synthetic.py):
ningún test necesita, ni debe tocar, un video de paciente.
"""

from pathlib import Path

import pytest

from benchmark.synthetic import write_test_video


@pytest.fixture
def plain_video(tmp_path: Path) -> Path:
    """64x48, 10 frames a 10 fps, sin metadato de rotación."""
    return write_test_video(tmp_path / "plain.mp4", frames=10, width=64, height=48, fps=10)


@pytest.fixture
def vertical_video(tmp_path: Path) -> Path:
    """
    Como un celular en vertical: 64x48 almacenado + rotación de 90° en los
    metadatos. Con la rotación aplicada, el modelo debe recibir 48x64.
    """
    return write_test_video(tmp_path / "vertical.mp4", frames=10, width=64, height=48, fps=10, rotation=90)


class FakeRunner:
    """Doble de un modelo: 'detecta' en los frames pares y registra si se cerró."""

    instances: list["FakeRunner"] = []

    def __init__(self, spec=None, fail_on_frame: int | None = None):
        self.spec = spec
        self.frames = 0
        self.closed = False
        self.fail_on_frame = fail_on_frame
        FakeRunner.instances.append(self)

    def infer(self, frame) -> bool:
        if self.fail_on_frame is not None and self.frames == self.fail_on_frame:
            raise RuntimeError("fallo simulado del modelo")
        self.frames += 1
        return self.frames % 2 == 0

    def threads_effective(self):
        return 2

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_runner_class():
    FakeRunner.instances = []
    return FakeRunner
