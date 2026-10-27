"""
En qué entorno corrió la medición: plataforma, CPU, límites del contenedor y
versiones de las librerías.

Cada resultado lleva esto pegado porque, sin ello, una fila no se puede
interpretar: 35 s de MediaPipe Heavy significan cosas distintas en el
portátil, en Docker Desktop con 2 CPU o en la instancia `2c-8g` de Render, y
con otra versión de mediapipe dejan de ser comparables con el servicio.

Correrlo solo imprime el reporte (es el comando por defecto de la imagen):

    python -m benchmark.environment           # reporte legible
    python -m benchmark.environment --json    # el mismo reporte, en JSON
    python -m benchmark.environment --check   # además, sale con 1 si las
                                              # versiones no son las del servicio

El primer despliegue en Render puede ser solo esto: confirma qué hardware y
qué límites da de verdad la instancia, sin tocar ningún video.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from benchmark import config
from benchmark.cgroup import Cgroup, effective_cpu_count

_RENDER_VARIABLES = (
    "RENDER_CPU_COUNT",
    "RENDER_SERVICE_ID",
    "RENDER_SERVICE_NAME",
    "RENDER_SERVICE_TYPE",
    "RENDER_INSTANCE_ID",
    "RENDER_GIT_COMMIT",
    "RENDER_GIT_BRANCH",
)
_TRACKED_DISTRIBUTIONS = (
    "mediapipe",
    "opencv-contrib-python",
    "numpy",
    "torch",
    "torchvision",
    "ultralytics",
    "psutil",
)


ENVIRONMENT_OVERRIDE_VARIABLE = "POSE_BENCHMARK_ENVIRONMENT"
GIT_COMMIT_VARIABLE = "POSE_BENCHMARK_GIT_COMMIT"


def detect_environment_label() -> str:
    """
    'render', 'docker' o 'local' (Render define RENDER=true en sus servicios).
    POSE_BENCHMARK_ENVIRONMENT lo reemplaza, para distinguir por ejemplo dos
    configuraciones de Docker.
    """
    if override := os.environ.get(ENVIRONMENT_OVERRIDE_VARIABLE, "").strip():
        return override
    if os.environ.get("RENDER", "").lower() == "true":
        return "render"
    if Path("/.dockerenv").exists():
        return "docker"
    return "local"


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "desconocido"


def distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def installed_opencv_distributions() -> list[str]:
    """Todas las distribuciones de OpenCV instaladas (deberia haber UNA)."""
    found = {
        dist.metadata["Name"].lower(): dist.version
        for dist in metadata.distributions()
        if (dist.metadata["Name"] or "").lower().startswith("opencv")
    }
    return [f"{name}=={version}" for name, version in sorted(found.items())]


def cv2_version() -> str | None:
    try:
        import cv2
    except ImportError:
        return None
    return cv2.__version__


def version_mismatches(snapshot: dict) -> list[str]:
    """Diferencias contra lo que corre el servicio. Lista vacía = comparable."""
    problems: list[str] = []
    python = tuple(int(part) for part in snapshot["python_version"].split(".")[:2])
    if python != config.SERVICE_PYTHON:
        problems.append(f"python {snapshot['python_version']} (el servicio usa {'.'.join(map(str, config.SERVICE_PYTHON))})")
    for name, pinned in config.SERVICE_PINNED_VERSIONS.items():
        installed = snapshot["versions"].get(name)
        if installed != pinned:
            problems.append(f"{name} {installed or 'no instalado'} (el servicio usa {pinned})")
    if snapshot["cv2_version"] != config.SERVICE_CV2_VERSION:
        problems.append(f"cv2 efectivo {snapshot['cv2_version']} (el servicio usa {config.SERVICE_CV2_VERSION})")
    for entry in snapshot["opencv_distributions"]:
        if entry.split("==")[0] in config.FORBIDDEN_OPENCV_DISTRIBUTIONS:
            problems.append(f"{entry} instalado junto a la contrib: pisa el directorio cv2/")
    return problems


def orientation_auto_default() -> float | None:
    """
    Valor de CAP_PROP_ORIENTATION_AUTO al abrir un video SIN fijarlo. En
    Windows es 0; en el contenedor de Render es justo lo que se quiere
    averiguar. open_video lo fija igual, esto es solo para dejarlo registrado.
    """
    import cv2

    from benchmark.synthetic import write_test_video

    with tempfile.TemporaryDirectory() as directory:
        path = write_test_video(Path(directory) / "probe.mp4", frames=2, rotation=90)
        cap = cv2.VideoCapture(str(path))
        try:
            return cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) if cap.isOpened() else None
        finally:
            cap.release()


def orientation_is_applied() -> bool:
    """Comprueba de punta a punta que open_video entrega el frame ya girado."""
    from benchmark.synthetic import write_test_video
    from benchmark.video_io import open_video

    with tempfile.TemporaryDirectory() as directory:
        path = write_test_video(Path(directory) / "probe.mp4", frames=2, width=64, height=48, rotation=90)
        cap = open_video(path)
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
    return bool(ok) and frame.shape[:2] == (64, 48)


def git_commit() -> str | None:
    """
    Qué código produjo la medición. RENDER_GIT_COMMIT solo existe cuando Render
    construye desde git; con la imagen de Docker Hub lo trae la propia imagen
    (POSE_BENCHMARK_GIT_COMMIT, del --build-arg GIT_COMMIT del Dockerfile).
    """
    for name in ("RENDER_GIT_COMMIT", GIT_COMMIT_VARIABLE):
        if value := os.environ.get(name, "").strip():
            return value
    return None


def snapshot(cgroup: Cgroup | None = None, include_packages: bool = False) -> dict:
    cgroup = cgroup or Cgroup.detect()
    limits = cgroup.limits()
    effective, effective_source = effective_cpu_count(limits)
    info = {
        "environment": detect_environment_label(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": cpu_model(),
        # Solo informativo: dentro de un contenedor puede ser el del host.
        "host_cpu_count": os.cpu_count(),
        "cgroup_version": limits.version,
        "cpu_limit_cores": limits.cpu_limit_cores,
        "memory_limit_mb": round(limits.memory_limit_mb, 1) if limits.memory_limit_mb else None,
        "cpu_effective": effective,
        "cpu_effective_source": effective_source,
        "budget": {"plan": config.RENDER_PLAN, "cpu_cores": config.BUDGET_CPU_CORES, "ram_mb": config.BUDGET_RAM_MB},
        "versions": {name: distribution_version(name) for name in _TRACKED_DISTRIBUTIONS},
        "cv2_version": cv2_version(),
        "opencv_distributions": installed_opencv_distributions(),
        "render": {name: os.environ.get(name) for name in _RENDER_VARIABLES if os.environ.get(name)},
        "git_commit": git_commit(),
    }
    info["version_mismatches"] = version_mismatches(info)
    if include_packages:
        info["packages"] = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in metadata.distributions())
    return info


def row_fields(info: dict) -> dict:
    """El subconjunto del entorno que va en CADA fila de resultados."""
    versions = info["versions"]
    return {
        "environment": info["environment"],
        "cpu_model": info["cpu_model"],
        "cgroup_version": info["cgroup_version"],
        "cpu_limit_cores": info["cpu_limit_cores"],
        "memory_limit_mb": info["memory_limit_mb"],
        "host_cpu_count": info["host_cpu_count"],
        "python_version": info["python_version"],
        "mediapipe_version": versions.get("mediapipe"),
        "cv2_version": info["cv2_version"],
        "numpy_version": versions.get("numpy"),
        "torch_version": versions.get("torch"),
        "ultralytics_version": versions.get("ultralytics"),
        "versions_match_service": not info["version_mismatches"],
        "render_instance_id": info["render"].get("RENDER_INSTANCE_ID"),
        "git_commit": info.get("git_commit"),
    }


def format_report(info: dict) -> str:
    limit_cpu = f"{info['cpu_limit_cores']:g} núcleos" if info["cpu_limit_cores"] else "sin cuota"
    limit_ram = f"{info['memory_limit_mb']:.0f} MB" if info["memory_limit_mb"] else "sin límite"
    lines = [
        f"Entorno:            {info['environment']}  ({info['platform']})",
        f"CPU:                {info['cpu_model']}",
        f"cgroup:             {info['cgroup_version'] or 'no disponible'}",
        f"  límite de CPU:    {limit_cpu}",
        f"  límite de RAM:    {limit_ram}",
        f"  núcleos usables:  {info['cpu_effective']} (según {info['cpu_effective_source']}; "
        f"os.cpu_count()={info['host_cpu_count']})",
        f"Presupuesto:        plan {info['budget']['plan']} = {info['budget']['cpu_cores']} CPU / {info['budget']['ram_mb']} MB",
        f"Código (commit):    {info.get('git_commit') or 'desconocido'}",
        f"Python:             {info['python_version']}",
        f"cv2 efectivo:       {info['cv2_version']}  (instaladas: {', '.join(info['opencv_distributions']) or 'ninguna'})",
    ]
    lines += [f"  {name:17s} {version or '-'}" for name, version in info["versions"].items()]
    if "orientation_auto_default" in info:
        lines.append(f"ORIENTATION_AUTO por defecto: {info['orientation_auto_default']}")
        lines.append(f"open_video aplica la rotación: {'sí' if info['orientation_applied'] else 'NO'}")
    if info["render"]:
        lines.append("Render:             " + ", ".join(f"{k}={v}" for k, v in info["render"].items()))
    if info["version_mismatches"]:
        lines.append("DIFERENCIAS con el servicio (las cifras no serían comparables):")
        lines += [f"  - {problem}" for problem in info["version_mismatches"]]
    else:
        lines.append("Versiones: iguales a las del servicio.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark.environment", description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="imprimir el reporte en JSON")
    parser.add_argument("--check", action="store_true", help="salir con 1 si las versiones no son las del servicio")
    args = parser.parse_args(argv)

    info = snapshot()
    info["orientation_auto_default"] = orientation_auto_default()
    info["orientation_applied"] = orientation_is_applied()

    print(json.dumps(info, ensure_ascii=False, indent=2) if args.json else format_report(info))
    if args.check and (info["version_mismatches"] or not info["orientation_applied"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
