"""
Videos sintéticos para tests y pruebas de humo, sin ningún dato de paciente.

`write_test_video` puede declarar una rotación en los metadatos, igual que un
celular que graba en vertical. `cv2.VideoWriter` no sabe escribir ese
metadato, así que se parchea a mano la matriz de transformación del átomo
`tkhd` del MP4 (ISO/IEC 14496-12, 8.3.2), que es de donde FFmpeg -- y por lo
tanto OpenCV -- lee la rotación. Así los tests ejercen el camino de verdad:
OpenCV lee `CAP_PROP_ORIENTATION_META` = 90 y, con `open_video`, entrega el
frame girado.

Cada frame lleva una franja blanca en el borde IZQUIERDO de la imagen
almacenada. Después de aplicar una rotación de 90°, la franja queda ARRIBA:
eso permite comprobar no solo que el frame cambió de forma sino que giró hacia
el lado correcto.
"""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np

# Matrices 3x3 de `tkhd` en punto fijo (a, b, u, c, d, v, x, y, w): a..d y x, y
# en 16.16; u, v, w en 2.30. Son las que escribe FFmpeg para cada rotación.
_ONE = 1 << 16
_W = 1 << 30
_ROTATION_MATRICES = {
    0: (_ONE, 0, 0, 0, _ONE, 0, 0, 0, _W),
    90: (0, _ONE, 0, -_ONE, 0, 0, 0, 0, _W),
    180: (-_ONE, 0, 0, 0, -_ONE, 0, 0, 0, _W),
    270: (0, -_ONE, 0, _ONE, 0, 0, 0, 0, _W),
}
STRIPE_WIDTH_FRACTION = 0.125


def set_mp4_rotation(path: Path, degrees: int) -> None:
    """Reescribe la matriz del primer `tkhd` de `path` para declarar `degrees`."""
    if degrees not in _ROTATION_MATRICES:
        raise ValueError(f"Rotación no soportada: {degrees} (usar 0, 90, 180 o 270)")
    data = bytearray(Path(path).read_bytes())
    box = data.find(b"tkhd")
    if box < 0:
        raise ValueError(f"No se encontró el átomo tkhd en {path}")
    version = data[box + 4]
    # tipo(4) + versión/flags(4) + tiempos, track_ID, reservado y duración
    # (20 bytes en v0, 32 en v1) + reservado/layer/grupo/volumen (16).
    matrix_offset = box + 4 + 4 + (32 if version == 1 else 20) + 16
    data[matrix_offset : matrix_offset + 36] = struct.pack(">9i", *_ROTATION_MATRICES[degrees])
    Path(path).write_bytes(bytes(data))


def write_test_video(
    path: Path,
    frames: int = 10,
    width: int = 64,
    height: int = 48,
    fps: float = 10.0,
    rotation: int = 0,
) -> Path:
    """
    Escribe un MP4 de `width`x`height` (dimensiones ALMACENADAS, antes de
    rotar) y, si `rotation` no es 0, le declara esa rotación.
    """
    path = Path(path)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el video de prueba en {path}")
    stripe = max(1, int(width * STRIPE_WIDTH_FRACTION))
    try:
        for index in range(frames):
            # Un degradado que cambia por frame, para que el códec no reduzca
            # todo a frames idénticos.
            frame = np.full((height, width, 3), (index * 7) % 128, dtype=np.uint8)
            frame[:, :stripe] = 255
            writer.write(frame)
    finally:
        writer.release()
    if rotation:
        set_mp4_rotation(path, rotation)
    return path
