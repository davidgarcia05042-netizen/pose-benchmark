"""
Qué videos se miden: un manifiesto JSON (suites/*.json) o rutas sueltas.

El manifiesto guarda, por video, un identificador, la ruta RELATIVA a
`--videos-dir` y su SHA-256. Los videos no viven en el repositorio ni en la
imagen (son del grupo de investigación, y algún día serán de pacientes): cada
entorno los pone en una carpeta y apunta `--videos-dir` ahí, o los baja de
Drive con `--videos-remote` (benchmark/storage.py). El hash garantiza que el portátil, Docker y
Render miden exactamente el mismo archivo, que es lo que hace comparables sus
números.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoEntry:
    video_id: str
    path: Path
    sha256: str | None = None
    joint: str | None = None


def load_suite(manifest_path: Path, videos_dir: Path, only: list[str] | None = None) -> tuple[dict, list[VideoEntry]]:
    """
    Lee el manifiesto y resuelve cada archivo contra `videos_dir`. `only`
    filtra por id, en el orden del manifiesto. Lanza ValueError si `only` pide
    un id que no existe o si hay ids repetidos.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    videos = manifest.get("videos", [])
    ids = [video["id"] for video in videos]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{manifest_path} tiene ids de video repetidos")
    if only:
        unknown = sorted(set(only) - set(ids))
        if unknown:
            raise ValueError(f"No están en {manifest_path}: {', '.join(unknown)}")
        videos = [video for video in videos if video["id"] in set(only)]
    entries = [
        VideoEntry(
            video_id=video["id"],
            # El manifiesto usa "/" siempre; pathlib lo resuelve en Windows y en Linux.
            path=Path(videos_dir) / Path(*video["file"].split("/")),
            sha256=video.get("sha256"),
            joint=video.get("joint"),
        )
        for video in videos
    ]
    meta = {key: value for key, value in manifest.items() if key != "videos"}
    return meta, entries


def entries_from_paths(paths: list[Path]) -> list[VideoEntry]:
    """Videos sueltos (--video): el id es el nombre del archivo sin extensión."""
    entries = [VideoEntry(video_id=Path(path).stem, path=Path(path)) for path in paths]
    ids = [entry.video_id for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Dos videos con el mismo nombre: los resultados se mezclarían")
    return entries


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_entries(entries: list[VideoEntry], verify_hash: bool = True) -> list[str]:
    """Problemas encontrados (archivo que falta, hash distinto). Vacía = todo en orden."""
    problems = []
    for entry in entries:
        if not entry.path.is_file():
            problems.append(f"{entry.video_id}: no existe {entry.path}")
        elif verify_hash and entry.sha256 and sha256_of(entry.path) != entry.sha256.lower():
            problems.append(f"{entry.video_id}: el SHA-256 de {entry.path} no coincide con el manifiesto")
    return problems
