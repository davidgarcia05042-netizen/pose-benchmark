"""
Monitor de recursos de UNA corrida: CPU y RAM del proceso actual, más lo que
el cgroup del contenedor registró mientras tanto.

Qué mide y de dónde sale cada número:

- `cpu_cores_used`: segundos de CPU del proceso (todos sus hilos, vía
  `psutil.Process().cpu_times()`) divididos por el tiempo de reloj. Es exacto,
  no depende del muestreo, y se lee directo contra el presupuesto: 2,0 es el
  plan `2c-8g` completo.
- `cpu_percent_mean`/`cpu_percent_max`: las mismas columnas que las mediciones
  en Docker del 2026-08-25, muestreadas con psutil (relativas a UN núcleo:
  200 % es el presupuesto completo). Se conservan para comparar con esas
  tablas; para decidir, vale `cpu_cores_used`.
- `ram_mb_peak`: en Linux, el pico EXACTO del proceso (`VmHWM` de
  /proc/self/status), reseteado al empezar cada corrida con
  /proc/self/clear_refs. Si el reset no se puede (Windows, o un kernel que no
  lo permite), cae al máximo de las muestras, que puede perder un pico más
  corto que el intervalo. `ram_peak_source` dice cuál de los dos se usó.
- `cgroup_*`: lo que registró el kernel para todo el contenedor. En especial
  `cgroup_nr_throttled`: cuántos periodos el proceso quiso más CPU de la que
  le da la cuota y el kernel lo frenó.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

import psutil

from benchmark.cgroup import Cgroup, CgroupCounters, counters_delta

_MB = 1024 * 1024
_PROC_STATUS = Path("/proc/self/status")
_CLEAR_REFS = Path("/proc/self/clear_refs")


def reset_peak_rss(clear_refs: Path = _CLEAR_REFS) -> bool:
    """Resetea el pico de RSS del proceso (Linux >= 4.0). Devuelve si se pudo."""
    try:
        clear_refs.write_text("5", encoding="ascii")
        return True
    except OSError:
        return False


def read_peak_rss_mb(status: Path = _PROC_STATUS) -> float | None:
    """`VmHWM` de /proc/self/status, en MB, o None si no existe."""
    try:
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024  # viene en kB
    except (OSError, ValueError, IndexError):
        return None
    return None


@dataclass
class MonitorResult:
    wall_s: float
    cpu_s: float
    cpu_user_s: float
    cpu_system_s: float
    cpu_cores_used: float | None
    cpu_percent_mean: float | None
    cpu_percent_max: float | None
    ram_mb_mean: float | None
    ram_mb_peak: float | None
    ram_mb_peak_sampled: float | None
    ram_peak_source: str
    samples: int
    cgroup: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        row = asdict(self)
        row.update(row.pop("cgroup"))
        return row


class ResourceMonitor:
    """
    Uso:
        with ResourceMonitor(cgroup=Cgroup.detect()) as monitor:
            trabajo_a_medir()
        monitor.result  # MonitorResult
    """

    def __init__(
        self,
        interval_s: float = 0.1,
        cgroup: Cgroup | None = None,
        process: psutil.Process | None = None,
        proc_status: Path = _PROC_STATUS,
        clear_refs: Path = _CLEAR_REFS,
    ):
        self.interval_s = interval_s
        self.result: MonitorResult | None = None
        self._cgroup = cgroup
        self._process = process or psutil.Process()
        self._proc_status = proc_status
        self._clear_refs = clear_refs
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_samples: list[float] = []
        self._rss_samples: list[float] = []

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._cpu_samples.append(self._process.cpu_percent(interval=None))
            self._rss_samples.append(self._process.memory_info().rss / _MB)

    def __enter__(self) -> "ResourceMonitor":
        self._hwm_reset = reset_peak_rss(self._clear_refs)
        self._cgroup_start = self._cgroup.counters() if self._cgroup else CgroupCounters()
        self._cpu_start = self._process.cpu_times()
        # La primera llamada a cpu_percent solo fija la referencia (devuelve 0).
        self._process.cpu_percent(interval=None)
        self._rss_samples.append(self._process.memory_info().rss / _MB)
        self._wall_start = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name="resource-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        wall = time.perf_counter() - self._wall_start
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        cpu_end = self._process.cpu_times()
        self._rss_samples.append(self._process.memory_info().rss / _MB)
        cgroup_end = self._cgroup.counters() if self._cgroup else CgroupCounters()

        user = cpu_end.user - self._cpu_start.user
        system = cpu_end.system - self._cpu_start.system
        sampled_peak = max(self._rss_samples)
        hwm = read_peak_rss_mb(self._proc_status) if self._hwm_reset else None

        self.result = MonitorResult(
            wall_s=wall,
            cpu_s=user + system,
            cpu_user_s=user,
            cpu_system_s=system,
            cpu_cores_used=(user + system) / wall if wall > 0 else None,
            cpu_percent_mean=mean(self._cpu_samples) if self._cpu_samples else None,
            cpu_percent_max=max(self._cpu_samples) if self._cpu_samples else None,
            ram_mb_mean=mean(self._rss_samples),
            # El pico exacto nunca puede ser menor que una muestra; el max()
            # cubre un VmHWM leído de un proceso que no es este (tests).
            ram_mb_peak=max(hwm, sampled_peak) if hwm is not None else sampled_peak,
            ram_mb_peak_sampled=sampled_peak,
            ram_peak_source="vmhwm" if hwm is not None else "sampled",
            samples=len(self._cpu_samples),
            cgroup=counters_delta(self._cgroup_start, cgroup_end),
        )
        return False
