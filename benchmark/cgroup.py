"""
Límites y contadores del cgroup del contenedor (Linux).

POR QUÉ: las cifras de CPU y RAM que van al documento tienen que salir de un
límite real (docs/DECISIONES_TECNICAS.md), y dentro de un contenedor `os.cpu_count()` puede
devolver los núcleos del HOST, no los asignados. El límite que de verdad aplica
el kernel está en el cgroup:

- cgroup v2 (`/sys/fs/cgroup/cpu.max`, `memory.max`, `cpu.stat`...), y
- cgroup v1 (`cpu.cfs_quota_us`, `memory.limit_in_bytes`...) en hosts viejos.

Leerlo sirve para tres cosas: dejar escrito en cada fila con qué límite se
midió (si Render da 2 CPU, acá tiene que decir 2), decidir cuántos hilos usar,
y ver cuánto tiempo el kernel frenó al proceso por pasarse de su cuota
(`nr_throttled`): un modelo que satura los dos núcleos se ve ahí antes que en
el porcentaje de CPU.

Fuera de Linux (el portátil con Windows) no hay cgroup: todo sale en None y la
medición lo deja registrado como "sin límite".

Semántica de los archivos: https://docs.kernel.org/admin-guide/cgroup-v2.html
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("/sys/fs/cgroup")
DEFAULT_PROC_SELF_CGROUP = Path("/proc/self/cgroup")

_MB = 1024 * 1024
# cgroup v1 expresa "sin límite de memoria" con un número enorme (el máximo
# del contador, alineado a página), no con una palabra.
_V1_UNLIMITED_BYTES = 1 << 60


@dataclass(frozen=True)
class CgroupLimits:
    version: str | None  # "v2", "v1" o None si no hay cgroup legible
    cpu_limit_cores: float | None  # None = sin cuota de CPU
    memory_limit_mb: float | None  # None = sin límite de memoria


@dataclass(frozen=True)
class CgroupCounters:
    """Foto de los contadores en un instante. Acumulados salvo memory_*."""

    cpu_usage_s: float | None = None
    nr_periods: int | None = None
    nr_throttled: int | None = None
    throttled_s: float | None = None
    memory_current_mb: float | None = None
    memory_peak_mb: float | None = None
    oom_kill: int | None = None


def _read_text(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _read_int(path: Path | None) -> int | None:
    text = _read_text(path)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _read_key_values(path: Path | None) -> dict[str, int]:
    """Archivos de la forma "clave valor" por línea (cpu.stat, memory.events...)."""
    text = _read_text(path)
    values: dict[str, int] = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                values[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return values


def _own_cgroup_paths(proc_self_cgroup: Path) -> dict[str, str]:
    """
    Ruta del cgroup propio por controlador, según /proc/self/cgroup.

    v2 aparece como "0::/ruta" (clave ""); v1 como "N:cpu,cpuacct:/ruta" (una
    clave por controlador). Dentro de un contenedor con su propio namespace de
    cgroup, la ruta es "/" y los archivos están directamente en la raíz.
    """
    paths: dict[str, str] = {}
    for line in (_read_text(proc_self_cgroup) or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if not controllers:
            paths[""] = path
        for controller in controllers.split(","):
            if controller:
                paths[controller] = path
    return paths


def _descend(base: Path, relative: str | None, marker: str) -> Path:
    """`base/relative` si ahí existe `marker`; si no, `base`."""
    if relative and relative != "/":
        candidate = base / relative.lstrip("/")
        if (candidate / marker).exists():
            return candidate
    return base


class Cgroup:
    """Acceso de solo lectura al cgroup del proceso actual."""

    def __init__(self, version: str | None, cpu_dir: Path | None, memory_dir: Path | None, cpuacct_dir: Path | None = None):
        self.version = version
        self._cpu_dir = cpu_dir
        self._memory_dir = memory_dir
        self._cpuacct_dir = cpuacct_dir or cpu_dir

    @classmethod
    def detect(cls, root: Path = DEFAULT_ROOT, proc_self_cgroup: Path = DEFAULT_PROC_SELF_CGROUP) -> "Cgroup":
        own = _own_cgroup_paths(proc_self_cgroup)

        if (root / "cgroup.controllers").exists():
            directory = _descend(root, own.get(""), "cpu.max")
            return cls("v2", directory, directory)

        cpu_base = next((root / name for name in ("cpu,cpuacct", "cpu") if (root / name).is_dir()), None)
        cpuacct_base = next((root / name for name in ("cpu,cpuacct", "cpuacct") if (root / name).is_dir()), None)
        memory_base = root / "memory" if (root / "memory").is_dir() else None
        if cpu_base is None and memory_base is None:
            return cls(None, None, None)

        return cls(
            "v1",
            _descend(cpu_base, own.get("cpu"), "cpu.cfs_quota_us") if cpu_base else None,
            _descend(memory_base, own.get("memory"), "memory.limit_in_bytes") if memory_base else None,
            _descend(cpuacct_base, own.get("cpuacct"), "cpuacct.usage") if cpuacct_base else None,
        )

    # --- Límites ---

    def limits(self) -> CgroupLimits:
        if self.version == "v2":
            return CgroupLimits("v2", self._v2_cpu_limit(), self._v2_memory_limit())
        if self.version == "v1":
            return CgroupLimits("v1", self._v1_cpu_limit(), self._v1_memory_limit())
        return CgroupLimits(None, None, None)

    def _v2_cpu_limit(self) -> float | None:
        text = _read_text(self._cpu_dir / "cpu.max") if self._cpu_dir else None
        parts = (text or "").split()
        if len(parts) != 2 or parts[0] == "max":
            return None
        try:
            quota, period = int(parts[0]), int(parts[1])
        except ValueError:
            return None
        return quota / period if period > 0 else None

    def _v2_memory_limit(self) -> float | None:
        text = _read_text(self._memory_dir / "memory.max") if self._memory_dir else None
        if text is None or text == "max":
            return None
        try:
            return int(text) / _MB
        except ValueError:
            return None

    def _v1_cpu_limit(self) -> float | None:
        if self._cpu_dir is None:
            return None
        quota = _read_int(self._cpu_dir / "cpu.cfs_quota_us")
        period = _read_int(self._cpu_dir / "cpu.cfs_period_us")
        if quota is None or period is None or quota <= 0 or period <= 0:
            return None
        return quota / period

    def _v1_memory_limit(self) -> float | None:
        value = _read_int(self._memory_dir / "memory.limit_in_bytes") if self._memory_dir else None
        if value is None or value >= _V1_UNLIMITED_BYTES:
            return None
        return value / _MB

    # --- Contadores ---

    def counters(self) -> CgroupCounters:
        if self.version == "v2":
            return self._v2_counters()
        if self.version == "v1":
            return self._v1_counters()
        return CgroupCounters()

    def _v2_counters(self) -> CgroupCounters:
        cpu = _read_key_values(self._cpu_dir / "cpu.stat") if self._cpu_dir else {}
        memory = self._memory_dir
        current = _read_int(memory / "memory.current") if memory else None
        # memory.peak existe desde Linux 5.19; en kernels anteriores queda None.
        peak = _read_int(memory / "memory.peak") if memory else None
        events = _read_key_values(memory / "memory.events") if memory else {}
        return CgroupCounters(
            cpu_usage_s=cpu["usage_usec"] / 1e6 if "usage_usec" in cpu else None,
            nr_periods=cpu.get("nr_periods"),
            nr_throttled=cpu.get("nr_throttled"),
            throttled_s=cpu["throttled_usec"] / 1e6 if "throttled_usec" in cpu else None,
            memory_current_mb=current / _MB if current is not None else None,
            memory_peak_mb=peak / _MB if peak is not None else None,
            oom_kill=events.get("oom_kill"),
        )

    def _v1_counters(self) -> CgroupCounters:
        cpu = _read_key_values(self._cpu_dir / "cpu.stat") if self._cpu_dir else {}
        usage_ns = _read_int(self._cpuacct_dir / "cpuacct.usage") if self._cpuacct_dir else None
        memory = self._memory_dir
        current = _read_int(memory / "memory.usage_in_bytes") if memory else None
        peak = _read_int(memory / "memory.max_usage_in_bytes") if memory else None
        oom = _read_key_values(memory / "memory.oom_control") if memory else {}
        return CgroupCounters(
            cpu_usage_s=usage_ns / 1e9 if usage_ns is not None else None,
            nr_periods=cpu.get("nr_periods"),
            nr_throttled=cpu.get("nr_throttled"),
            throttled_s=cpu["throttled_time"] / 1e9 if "throttled_time" in cpu else None,
            memory_current_mb=current / _MB if current is not None else None,
            memory_peak_mb=peak / _MB if peak is not None else None,
            oom_kill=oom.get("oom_kill"),
        )


def counters_delta(start: CgroupCounters, end: CgroupCounters) -> dict[str, float | int | None]:
    """
    Lo que pasó entre dos fotos: diferencia de los acumulados, y el valor final
    de la memoria (que no es acumulable). Un campo que falte en cualquiera de
    las dos fotos queda en None.
    """

    def diff(a, b):
        return b - a if a is not None and b is not None else None

    return {
        "cgroup_cpu_s": diff(start.cpu_usage_s, end.cpu_usage_s),
        "cgroup_nr_periods": diff(start.nr_periods, end.nr_periods),
        "cgroup_nr_throttled": diff(start.nr_throttled, end.nr_throttled),
        "cgroup_throttled_s": diff(start.throttled_s, end.throttled_s),
        "cgroup_oom_kill": diff(start.oom_kill, end.oom_kill),
        # Pico de TODO el contenedor desde que arrancó (o desde el último reset),
        # no de esta corrida: sirve para ver el margen frente al límite.
        "cgroup_memory_peak_mb": end.memory_peak_mb,
    }


def effective_cpu_count(limits: CgroupLimits) -> tuple[int, str]:
    """
    Núcleos que el proceso puede usar de verdad, y de dónde sale el dato.

    Se toma el menor entre la cuota del cgroup (redondeada hacia arriba), la
    afinidad de CPU del proceso y RENDER_CPU_COUNT (los núcleos del plan, que
    Render define en cada servicio). Este último cubre el caso de que Render
    limite la CPU por otra vía que no deje cuota visible en el cgroup: sin él,
    la afinidad daría los núcleos del host. Solo si no hay ninguno (Windows) se
    cae a `os.cpu_count()`, que NO sirve para calcular porcentajes del
    presupuesto (ver docs/DECISIONES_TECNICAS.md).
    """
    candidates: list[tuple[int, str]] = []
    if limits.cpu_limit_cores:
        candidates.append((max(1, math.ceil(limits.cpu_limit_cores - 1e-9)), "cgroup_quota"))
    if hasattr(os, "sched_getaffinity"):
        candidates.append((len(os.sched_getaffinity(0)), "cpu_affinity"))
    render_cpus = os.environ.get("RENDER_CPU_COUNT", "").strip()
    if render_cpus.isdigit() and int(render_cpus) > 0:
        candidates.append((int(render_cpus), "render_cpu_count"))
    if candidates:
        return min(candidates, key=lambda item: item[0])
    return os.cpu_count() or 1, "os_cpu_count"
