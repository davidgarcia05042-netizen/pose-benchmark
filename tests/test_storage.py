"""
El transporte con rclone, con un doble que hace de rclone sobre una carpeta
local: se prueban los comandos y el manejo de errores, sin red ni credenciales.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from benchmark import storage


class FakeRclone:
    """
    Entiende `copy` (con o sin --files-from-raw) y `mkdir` sobre remotos de la
    forma `fake:<ruta>`, que apuntan a carpetas dentro de `root`.
    """

    def __init__(self, root: Path, fail_on: str | None = None):
        self.root = root
        self.fail_on = fail_on
        self.commands: list[list[str]] = []

    def _local(self, location: str) -> Path:
        return self.root / location.split(":", 1)[1] if location.startswith("fake:") else Path(location)

    def __call__(self, command):
        self.commands.append(command)
        verb = command[1]
        if verb == self.fail_on:
            return subprocess.CompletedProcess(command, 1)
        if verb == "mkdir":
            self._local(command[2]).mkdir(parents=True, exist_ok=True)
        elif verb == "copy":
            source, destination = self._local(command[2]), self._local(command[3])
            if "--files-from-raw" in command:
                listing = Path(command[command.index("--files-from-raw") + 1])
                names = [line for line in listing.read_text(encoding="utf-8").splitlines() if line]
            else:
                names = [p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()]
            for name in names:
                if not (source / name).is_file():  # rclone copy con --files-from ignora lo que falta
                    continue
                (destination / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, destination / name)
        return subprocess.CompletedProcess(command, 0)


def test_join_remote_avoids_double_slashes():
    assert storage.join_remote("gdrive:base/", "/r1/") == "gdrive:base/r1"
    assert storage.join_remote("gdrive:", "r1") == "gdrive:r1"
    assert storage.join_remote("gdrive:pose benchmark/TOMA FRONTAL") == "gdrive:pose benchmark/TOMA FRONTAL"


def test_download_brings_only_the_listed_files_keeping_subfolders(tmp_path):
    remote = tmp_path / "drive" / "TOMA FRONTAL"
    for name in ("2EJ1/2EJ1.mp4", "5EJ6/5EJ6.mp4", "5EJ6/otro.png"):
        (remote / name).parent.mkdir(parents=True, exist_ok=True)
        (remote / name).write_bytes(name.encode())
    fake = FakeRclone(tmp_path / "drive")

    storage.Rclone("rclone", run=fake).download("fake:TOMA FRONTAL", ["2EJ1/2EJ1.mp4", "5EJ6/5EJ6.mp4"], tmp_path / "local")

    assert sorted(p.relative_to(tmp_path / "local").as_posix() for p in (tmp_path / "local").rglob("*") if p.is_file()) == [
        "2EJ1/2EJ1.mp4", "5EJ6/5EJ6.mp4"]
    assert "--checksum" in fake.commands[0]


def test_a_failing_transfer_raises_storage_error(tmp_path):
    fake = FakeRclone(tmp_path, fail_on="copy")

    with pytest.raises(storage.StorageError, match="código 1"):
        storage.Rclone("rclone", run=fake).upload_dir(tmp_path, "fake:x")


def test_missing_rclone_binary_is_a_storage_error(tmp_path):
    with pytest.raises(storage.StorageError, match="no está instalado"):
        storage.Rclone(str(tmp_path / "no-existe-rclone")).ensure_dir("fake:x")


def test_credentials_do_not_reach_the_measurement_sessions():
    env = {"RCLONE_CONFIG_GDRIVE_TOKEN": "secreto", "rclone_config_gdrive_type": "drive", "PATH": "/bin"}

    assert storage.without_secrets(env) == {"PATH": "/bin"}
