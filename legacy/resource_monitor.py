"""
Monitor de recursos: muestrea CPU%, RAM y (si hay GPU NVIDIA disponible)
utilización de GPU / VRAM **específicamente del proceso actual**, en un
hilo en segundo plano, mientras se ejecuta el bloque de código dentro de
un `with ResourceMonitor():`.

Sobre CPU y RAM: ya estaban correctamente aisladas al proceso desde el
inicio, vía `psutil.Process()` -- no miden el sistema completo.

Sobre GPU: se usa `benchmark.nvml_process_gpu`, que filtra por PID usando
NVML. Esto es deliberado -- la versión anterior usaba `nvidia-smi`, que
reporta toda la tarjeta (contaminado por lo que sea que el sistema
operativo o el navegador tengan corriendo en ese momento). Si no hay GPU
NVIDIA o no está instalado `nvidia-ml-py`, simplemente no reporta esos
campos, no cae de vuelta a la medición de sistema completo.
"""

import threading
import time
from dataclasses import dataclass, field

import psutil

from legacy import nvml_process_gpu


@dataclass
class ResourceSample:
    timestamp: float
    cpu_percent: float
    ram_mb: float
    gpu_utilization_percent: float | None = None
    gpu_memory_mb: float | None = None


@dataclass
class ResourceStats:
    samples: list[ResourceSample] = field(default_factory=list)


class ResourceMonitor:
    """
    Uso:
        with ResourceMonitor() as monitor:
            hacer_el_trabajo_que_se_quiere_medir()
        print(monitor.stats.samples)

    Nota sobre cpu_percent: psutil lo reporta relativo a un solo núcleo,
    así que en una máquina de 6 núcleos, un proceso usando 2 núcleos
    completos se reporta como ~200%, no ~33%. Hay que tenerlo en cuenta al
    comparar con el límite de "2 CPU" de Render -- ahí, 200% ya sería el
    límite completo del contenedor, no una fracción de él.

    Nota sobre GPU: si `gpu_utilization_percent` sale como None en todas
    las muestras aunque sí tengas una GPU NVIDIA, es porque tu combinación
    de driver/GPU no soporta la consulta de utilización por proceso (ver
    nvml_process_gpu.py). La VRAM por proceso suele estar disponible con
    más frecuencia que la utilización por proceso.
    """

    def __init__(self, interval_seconds: float = 0.2):
        self.interval_seconds = interval_seconds
        self.stats = ResourceStats()
        self._process = psutil.Process()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.has_gpu = nvml_process_gpu.gpu_available()

    def _sample_loop(self) -> None:
        # La primera llamada a cpu_percent siempre devuelve 0.0 (necesita
        # un intervalo de referencia previo), así que se "gasta" aquí.
        self._process.cpu_percent(interval=None)

        while not self._stop_event.is_set():
            cpu = self._process.cpu_percent(interval=None)
            ram_mb = self._process.memory_info().rss / (1024 * 1024)

            gpu_util, gpu_mem = None, None
            if self.has_gpu:
                gpu_util = nvml_process_gpu.get_process_gpu_utilization_percent()
                gpu_mem = nvml_process_gpu.get_process_vram_mb()

            self.stats.samples.append(
                ResourceSample(
                    timestamp=time.time(),
                    cpu_percent=cpu,
                    ram_mb=ram_mb,
                    gpu_utilization_percent=gpu_util,
                    gpu_memory_mb=gpu_mem,
                )
            )
            self._stop_event.wait(self.interval_seconds)

    def __enter__(self) -> "ResourceMonitor":
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

