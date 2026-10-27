"""
Que lo que este repositorio COPIA del servicio siga siendo igual al servicio.

Una medición solo sirve para decidir sobre el servicio si corre sus mismas
librerías y se juzga contra sus mismos límites. Estos valores están copiados
en benchmark/config.py y requirements.txt; si alguien los cambia en un lado y
no en el otro, las cifras dejan de ser comparables sin que nada lo avise.

Los tests contra el repositorio hermano se SALTAN si no está al lado (en
Render, en CI): solo corren en el portátil, que es donde se editan los dos.
"""

import re
from pathlib import Path

import pytest

from benchmark import config
from benchmark.models import MODELS

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO.parent / "Servicio de procesamiento asincrono"

needs_service = pytest.mark.skipif(not SERVICE.is_dir(), reason="el repositorio del servicio no está al lado")


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*([A-Za-z0-9_.\-\[\]]+)==([^\s;#]+)", line)
        if match:
            pins[match.group(1).split("[")[0].lower()] = match.group(2)
    return pins


def test_requirements_pin_the_service_versions():
    pins = _pins(REPO / "requirements.txt")

    for name, version in config.SERVICE_PINNED_VERSIONS.items():
        assert pins.get(name) == version, name


def test_requirements_do_not_pin_another_opencv():
    pins = _pins(REPO / "requirements.txt")

    assert not set(pins) & set(config.FORBIDDEN_OPENCV_DISTRIBUTIONS)


def test_dockerfile_installs_the_pinned_torch_and_cleans_opencv():
    pins = _pins(REPO / "requirements.txt")
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")

    assert f"torch=={pins['torch']}" in dockerfile
    assert f"torchvision=={pins['torchvision']}" in dockerfile
    assert "pip uninstall -y opencv-python opencv-python-headless" in dockerfile
    assert f"opencv-contrib-python=={pins['opencv-contrib-python']}" in dockerfile
    assert "FROM python:3.12" in dockerfile


def test_the_image_only_takes_what_is_measured():
    """Lista blanca: ningún video, ni legacy, ni pesos descartados, ni análisis local."""
    lines = [line.strip() for line in (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()]
    allowed = {line[1:] for line in lines if line.startswith("!")}

    assert "*" in lines
    assert allowed == {"requirements.txt", "benchmark/", "scripts/__init__.py", "scripts/run_benchmark.py", "suites/", "yolo26n-pose.pt"}


@needs_service
def test_same_versions_as_the_service():
    ours = _pins(REPO / "requirements.txt")
    theirs = _pins(SERVICE / "requirements.txt")

    for name in config.SERVICE_PINNED_VERSIONS:
        assert ours[name] == theirs[name], f"{name}: benchmark {ours[name]} vs servicio {theirs[name]}"


@needs_service
def test_same_base_image_as_the_service():
    base = re.search(r"^FROM\s+(\S+)", (SERVICE / "Dockerfile").read_text(encoding="utf-8"), re.MULTILINE).group(1)

    assert re.search(rf"^FROM\s+{re.escape(base)}\s*$", (REPO / "Dockerfile").read_text(encoding="utf-8"), re.MULTILINE)


@needs_service
def test_same_worker_limits_as_the_service():
    blueprint = (SERVICE / "render.yaml").read_text(encoding="utf-8")
    worker = blueprint.split("- name: pose-worker", 1)[1].split("- name:", 1)[0]

    assert re.search(r"^\s*plan:\s*(\S+)", worker, re.MULTILINE).group(1) == config.RENDER_PLAN
    assert int(re.search(r"maxShutdownDelaySeconds:\s*(\d+)", worker).group(1)) == config.SERVICE_MAX_SHUTDOWN_DELAY_S
    assert int(re.search(r"--concurrency=(\d+)", worker).group(1)) == config.SERVICE_WORKER_CONCURRENCY
    max_size = re.search(r"key: MAX_VIDEO_SIZE_MB\s+value:\s*\"?(\d+)", blueprint).group(1)
    assert int(max_size) == config.SERVICE_MAX_VIDEO_SIZE_MB
    stuck = re.search(r"key: STUCK_JOB_TIMEOUT_MINUTES\s+value:\s*\"?(\d+)", blueprint).group(1)
    assert int(stuck) * 60 == config.SERVICE_STUCK_JOB_TIMEOUT_S


@needs_service
def test_same_visibility_timeout_as_the_service():
    worker = (SERVICE / "app" / "core" / "worker.py").read_text(encoding="utf-8")

    assert int(re.search(r"^VISIBILITY_TIMEOUT_SECONDS\s*=\s*(\d+)", worker, re.MULTILINE).group(1)) == config.SERVICE_VISIBILITY_TIMEOUT_S


@needs_service
def test_mediapipe_models_have_the_service_complexity():
    constants = (SERVICE / "app" / "constants" / "pose_models.py").read_text(encoding="utf-8")
    service = {name: int(value) for name, value in re.findall(r'"(mediapipe_\w+)":\s*(\d+)', constants)}

    ours = {name: spec.complexity for name, spec in MODELS.items() if spec.family == "mediapipe"}
    assert ours == service
