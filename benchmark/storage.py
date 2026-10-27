"""
Videos de entrada y resultados de salida a través de rclone.

El disco de Render es efímero y la imagen no lleva videos, así que cada corrida
baja los videos que va a medir al arrancar y sube su `results/<run_id>/` al
terminar. El almacén es una carpeta de Google Drive que también ve el grupo de
investigación; rclone la habla con el mismo comando que S3, R2 u OneDrive, de
modo que cambiar de almacén es cambiar variables de entorno, no código.

La configuración del remoto NO vive en la imagen ni en el repositorio: rclone
la lee de variables de entorno (en Render, secretas). Para un remoto llamado
`gdrive`:

    RCLONE_CONFIG_GDRIVE_TYPE=drive
    RCLONE_CONFIG_GDRIVE_SCOPE=drive.file
    RCLONE_CONFIG_GDRIVE_TOKEN={"access_token": ..., "refresh_token": ...}

y las rutas se escriben `gdrive:carpeta/subcarpeta`. Ver README, "Videos y
resultados en Google Drive".

La descarga ocurre ANTES de medir y no entra en ningún tiempo: el servicio
recibe el video por otra vía y eso no es costo del modelo. Después de bajarlos,
el SHA-256 de cada video se compara contra el manifiesto (suite.check_entries),
así que un archivo cambiado en Drive detiene la corrida en vez de medir otro
video con el mismo nombre.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

RCLONE_BINARY_VARIABLE = "POSE_BENCHMARK_RCLONE"
VIDEOS_REMOTE_VARIABLE = "POSE_BENCHMARK_VIDEOS_REMOTE"
RESULTS_REMOTE_VARIABLE = "POSE_BENCHMARK_RESULTS_REMOTE"
# Las sesiones de medición no hablan con el almacén: no reciben las credenciales.
SECRET_PREFIX = "RCLONE_CONFIG_"

# -v lista cada archivo transferido (8 videos, 3 archivos de resultados: poco
# log); --stats cada 30 s deja ver que una descarga grande sigue viva.
_LOG_FLAGS = ["-v", "--stats-one-line", "--stats", "30s"]


class StorageError(RuntimeError):
    """rclone no está, no pudo autenticarse o la transferencia falló."""


def join_remote(remote: str, *parts: str) -> str:
    """`gdrive:base` + `r1` -> `gdrive:base/r1`, sin barras dobles."""
    path = remote.rstrip("/")
    for part in (part.strip("/") for part in parts):
        if part:
            path = f"{path}{part}" if path.endswith(":") else f"{path}/{part}"
    return path


def without_secrets(env: dict) -> dict:
    return {key: value for key, value in env.items() if not key.upper().startswith(SECRET_PREFIX)}


class Rclone:
    """
    Envoltorio mínimo de la línea de comandos de rclone. `run` es inyectable
    para los tests (por defecto, subprocess.run sin capturar: la salida de
    rclone va directo al log).
    """

    def __init__(self, binary: str | None = None, run=subprocess.run):
        self.binary = binary or os.environ.get(RCLONE_BINARY_VARIABLE, "rclone")
        self._run = run

    def _call(self, *args: str) -> None:
        command = [self.binary, *args]
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            completed = self._run(command)
        except FileNotFoundError as error:
            raise StorageError(f"rclone no está instalado o no está en el PATH ({self.binary})") from error
        if completed.returncode != 0:
            raise StorageError(f"rclone {args[0]} terminó con código {completed.returncode} (detalle arriba en el log)")

    def download(self, remote_dir: str, relative_files: list[str], local_dir: Path) -> None:
        """
        Baja SOLO `relative_files` (rutas con "/" relativas a `remote_dir`) a
        `local_dir`, conservando las subcarpetas. Lo que ya está y coincide en
        tamaño y checksum no se vuelve a bajar.
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as listing:
            listing.write("\n".join(relative_files) + "\n")
        try:
            self._call("copy", remote_dir, str(local_dir), "--files-from-raw", listing.name, "--checksum", *_LOG_FLAGS)
        finally:
            Path(listing.name).unlink(missing_ok=True)

    def ensure_dir(self, remote_dir: str) -> None:
        """Crea la carpeta remota. Sirve también para probar, antes de medir, que se puede escribir."""
        self._call("mkdir", remote_dir)

    def upload_dir(self, local_dir: Path, remote_dir: str) -> None:
        self._call("copy", str(local_dir), remote_dir, *_LOG_FLAGS)
