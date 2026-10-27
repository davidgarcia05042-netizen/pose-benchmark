"""
Medición de GPU aislada al proceso actual (por PID), usando NVML a través
del paquete oficial `nvidia-ml-py` de NVIDIA (se importa como `pynvml`,
aunque el paquete correcto a instalar es `nvidia-ml-py`, no el antiguo
`pynvml` de PyPI, que está descontinuado).

Por qué esto reemplaza la medición anterior con `nvidia-smi`:
`nvidia-smi --query-gpu=...` reporta el estado de TODA la tarjeta, sin
distinguir qué proceso la está usando -- si tenías el navegador, Discord,
o cualquier otra cosa usando aceleración por GPU al mismo tiempo, ese
consumo se mezclaba con el de nuestro benchmark. Aquí se filtra
explícitamente por el PID del proceso Python que está corriendo, usando
`nvmlDeviceGetComputeRunningProcesses` (memoria) y
`nvmlDeviceGetProcessUtilization` (utilización).

Si algo falla (no hay GPU NVIDIA, no está instalado `nvidia-ml-py`, o el
driver no soporta alguna de estas consultas), las funciones devuelven
`None` -- a propósito, en vez de "caer de vuelta" a la medición de toda
la tarjeta, que reintroduciría el mismo problema que esto resuelve.
"""

import atexit
import os
import time

try:
    import pynvml

    _NVML_IMPORTABLE = True
except ImportError:
    _NVML_IMPORTABLE = False

_current_pid = os.getpid()
_handle = None
_last_timestamp_us: int | None = None
_initialized = False


def _ensure_initialized() -> bool:
    global _handle, _initialized

    if _initialized:
        return _handle is not None

    _initialized = True

    if not _NVML_IMPORTABLE:
        return False

    try:
        pynvml.nvmlInit()
        _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        atexit.register(_shutdown)
        return True
    except Exception:
        _handle = None
        return False


def _shutdown() -> None:
    if _NVML_IMPORTABLE and _handle is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def gpu_available() -> bool:
    return _ensure_initialized()


def get_process_vram_mb() -> float | None:
    """VRAM usada específicamente por ESTE proceso (no el total de la tarjeta), en MB."""
    if not _ensure_initialized():
        return None
    try:
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(_handle)
        for proc in processes:
            if proc.pid == _current_pid:
                return proc.usedGpuMemory / (1024 * 1024)
        return 0.0  # el proceso existe pero no está usando GPU en este instante
    except Exception:
        return None


def get_process_gpu_utilization_percent() -> float | None:
    """
    % de utilización de GPU específicamente de ESTE proceso.

    Nota: esta consulta (nvmlDeviceGetProcessUtilization) no está
    soportada en todas las combinaciones de driver/GPU. Si no está
    disponible, devuelve None -- deliberadamente no cae de vuelta a la
    utilización de toda la tarjeta, para no mezclar de nuevo la señal.
    """
    global _last_timestamp_us

    if not _ensure_initialized():
        return None
    try:
        since = _last_timestamp_us or int((time.time() - 1) * 1_000_000)
        samples = pynvml.nvmlDeviceGetProcessUtilization(_handle, since)
        _last_timestamp_us = int(time.time() * 1_000_000)
        for sample in samples:
            if sample.pid == _current_pid:
                return float(sample.smUtil)
        return 0.0
    except Exception:
        return None
