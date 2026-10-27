"""
Apertura de videos con la orientación correcta, y lectura de sus metadatos.

Los celulares graban en vertical guardando los píxeles "acostados" más un
metadato de rotación (ej. rotation=90) que los reproductores aplican al
mostrar. OpenCV no lo aplica por defecto: sin esto, los modelos reciben a la
persona girada 90°. Es la misma función que
`entorno-pruebas/app/core/video_io.py::open_video`, y TODO video que se mida
en este repositorio se abre acá.

Para el costo importa por dos razones:
- La rotación se hace después de decodificar cada frame, así que es trabajo
  que el servicio va a pagar con los videos de la plataforma. Queda dentro de
  la medición, no fuera.
- Un modelo que no encuentra a la persona trabaja distinto (MediaPipe, sin
  detección, no corre la etapa de landmarks), así que medir sobre frames
  acostados mide otra cosa.

El valor se fija explícito y no se confía en el default: en Windows es 0
(verificado el 2026-09-28) y en el contenedor de Render no está verificado.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

# Lo que se escribe en cada fila de resultados, igual que el
# "video_orientation" del run_info.json de entorno-pruebas.
VIDEO_ORIENTATION_TAG = "metadata_applied"


def open_video(video_path: str | Path) -> cv2.VideoCapture:
    """Abre `video_path` aplicando la rotación declarada en sus metadatos."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


@dataclass(frozen=True)
class VideoInfo:
    """Metadatos de un video, tal como los va a ver el modelo."""

    file_name: str
    size_mb: float
    fps: float
    frames_reported: int
    duration_s: float
    # Lo que dice el contenedor (píxeles almacenados, antes de rotar).
    stored_resolution: str
    # Lo que entrega open_video (después de aplicar la rotación).
    decoded_resolution: str
    rotation_meta_deg: int
    codec: str

    def as_row(self) -> dict:
        """Campos con prefijo `video_`, listos para una fila de resultados."""
        return {f"video_{key}": value for key, value in asdict(self).items()}


def _fourcc_to_text(value: float) -> str:
    code = int(value)
    text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return text if text.isprintable() else str(code)


def probe_video(video_path: str | Path) -> VideoInfo:
    """
    Lee los metadatos de `video_path` y decodifica el primer frame con
    `open_video`, para saber la resolución que de verdad recibe el modelo.

    Lanza FileNotFoundError si el video no se puede abrir y ValueError si no
    tiene ni un frame legible.
    """
    path = Path(video_path)
    cap = open_video(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        rotation = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0))
        codec = _fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"El video no tiene un primer frame legible: {path}")
        decoded_height, decoded_width = frame.shape[:2]
    finally:
        cap.release()

    # Las dimensiones almacenadas son las decodificadas "des-rotadas".
    if rotation % 180 == 90:
        stored_width, stored_height = decoded_height, decoded_width
    else:
        stored_width, stored_height = decoded_width, decoded_height

    return VideoInfo(
        file_name=path.name,
        size_mb=round(path.stat().st_size / (1024 * 1024), 2),
        fps=round(fps, 3),
        frames_reported=frames,
        duration_s=round(frames / fps, 3) if fps else 0.0,
        stored_resolution=f"{stored_width}x{stored_height}",
        decoded_resolution=f"{decoded_width}x{decoded_height}",
        rotation_meta_deg=rotation,
        codec=codec,
    )
