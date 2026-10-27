"""
Los modelos que se miden y cómo se corre cada uno sobre un frame.

Los tres activos se decidieron en `entorno-pruebas` (fuente de verdad de qué
modelo ganó): YOLO26-nano, MediaPipe Heavy y MediaPipe Lite. MediaPipe Full se
mide como referencia, aunque no es candidato. Los nombres son los mismos que
usan el servicio (`MODEL_COMPLEXITY`) y entorno-pruebas, para poder cruzar
resultados por nombre.

Cada corredor hace lo mínimo que haría el servicio con el modelo, sin nada
encima:

- MediaPipe: `mp.solutions.pose.Pose(model_complexity=...)` con los valores
  por defecto, y conversión BGR->RGB con `cv2.cvtColor`, igual que
  `app/core/pose_analysis.py` del servicio.
- YOLO: el frame BGR tal cual (Ultralytics lo espera así), en CPU, con los
  umbrales por defecto, igual que el wrapper de entorno-pruebas.

Los imports pesados (mediapipe, torch, ultralytics) son perezosos: cada
modelo se mide en su propio proceso (ver benchmark/session.py) y no debe
cargar las librerías del otro. Importar ultralytics, por ejemplo, llama a
`cv2.setNumThreads(0)` y carga torch, y eso ensuciaría la RAM y los hilos de
una medición de MediaPipe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # importar numpy acá cargaría OpenBLAS antes de fijar sus hilos
    import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str  # "mediapipe" | "yolo"
    role: str  # "candidato" | "referencia"
    complexity: int | None = None  # solo MediaPipe
    weights: str | None = None  # solo YOLO: nombre del archivo de pesos


MODELS: dict[str, ModelSpec] = {
    "yolo26n": ModelSpec("yolo26n", "yolo", "candidato", weights="yolo26n-pose.pt"),
    "mediapipe_lite": ModelSpec("mediapipe_lite", "mediapipe", "candidato", complexity=0),
    "mediapipe_full": ModelSpec("mediapipe_full", "mediapipe", "referencia", complexity=1),
    "mediapipe_heavy": ModelSpec("mediapipe_heavy", "mediapipe", "candidato", complexity=2),
}
DEFAULT_MODELS = tuple(MODELS)

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def apply_thread_env(threads: int) -> None:
    """
    Fija las variables de hilos de las librerías numéricas. Tiene efecto solo
    si se llama ANTES de importar torch/mediapipe: MKL y OpenBLAS las leen una
    sola vez al inicializarse.
    """
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(threads)


class PoseRunner(Protocol):
    spec: ModelSpec

    def infer(self, frame_bgr: np.ndarray) -> bool:
        """Corre el modelo sobre un frame BGR. Devuelve si detectó a una persona."""

    def threads_effective(self) -> int | None:
        """Hilos que la librería reporta estar usando, si lo expone."""

    def close(self) -> None: ...


class MediaPipeRunner:
    def __init__(self, spec: ModelSpec, threads: int | None):
        import cv2
        import mediapipe as mp

        self.spec = spec
        self._cv2 = cv2
        if threads:
            cv2.setNumThreads(threads)
        self._pose = mp.solutions.pose.Pose(model_complexity=spec.complexity)

    def infer(self, frame_bgr: np.ndarray) -> bool:
        result = self._pose.process(self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB))
        return result.pose_landmarks is not None

    def threads_effective(self) -> int | None:
        # TFLite no expone cuántos hilos usa el grafo de MediaPipe.
        return None

    def close(self) -> None:
        self._pose.close()


class YoloRunner:
    """
    HALLAZGO (2026-08-25, ver legacy/benchmark_yolov8.py): Ultralytics resetea
    `torch.set_num_threads` en la primera inferencia, a
    `min(8, os.cpu_count() - 1)`. Dentro de un contenedor, `os.cpu_count()`
    puede ver los núcleos del HOST: con una cuota de 2 CPU, YOLO lanzaría
    hasta 8 hilos peleando por 2 núcleos. Se corrige en dos capas:

    1. Se reemplaza `NUM_THREADS` en `ultralytics.utils.torch_utils` antes de
       la primera inferencia, para que el reset deje el valor pedido.
    2. Se reafirma `torch.set_num_threads` antes de cada frame, por si una
       versión futura de Ultralytics mueve esa constante (cuesta microsegundos).

    La fila guarda `threads_effective` (lo que reporta torch al terminar) para
    comprobar que el límite se respetó.
    """

    def __init__(self, spec: ModelSpec, threads: int | None, weights_dir: Path):
        weights = Path(weights_dir) / spec.weights
        # Sin esto, Ultralytics intenta DESCARGAR un peso que no encuentra, y
        # en Render eso es una dependencia de red escondida en la medición.
        if not weights.is_file():
            raise FileNotFoundError(f"No están los pesos de {spec.name}: {weights}")

        import torch
        import ultralytics.utils.torch_utils as torch_utils
        from ultralytics import YOLO

        self.spec = spec
        self._torch = torch
        self._threads = threads
        if threads:
            torch.set_num_threads(threads)
            if hasattr(torch_utils, "NUM_THREADS"):
                torch_utils.NUM_THREADS = threads
        self._model = YOLO(str(weights))

    def infer(self, frame_bgr: np.ndarray) -> bool:
        if self._threads:
            self._torch.set_num_threads(self._threads)
        result = self._model(frame_bgr, device="cpu", verbose=False)[0]
        return result.boxes is not None and len(result.boxes) > 0

    def threads_effective(self) -> int | None:
        return self._torch.get_num_threads()

    def close(self) -> None:
        # Ultralytics no requiere liberar recursos explícitamente.
        self._model = None


def load_runner(spec: ModelSpec, threads: int | None, weights_dir: Path) -> PoseRunner:
    if spec.family == "mediapipe":
        return MediaPipeRunner(spec, threads)
    if spec.family == "yolo":
        return YoloRunner(spec, threads, weights_dir)
    raise ValueError(f"Familia de modelo desconocida: {spec.family}")
