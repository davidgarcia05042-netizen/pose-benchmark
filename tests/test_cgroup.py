"""
Lectura del cgroup, sobre árboles de archivos falsos con el formato real de
cgroup v2 y v1 (https://docs.kernel.org/admin-guide/cgroup-v2.html). Así se
prueba en Windows y en CI lo que en Render se lee de /sys/fs/cgroup.
"""

import os
from pathlib import Path

import pytest

from benchmark.cgroup import Cgroup, CgroupCounters, CgroupLimits, counters_delta, effective_cpu_count

GIB = 1024**3


def _write(directory: Path, files: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_text(content, encoding="ascii")


def _v2_tree(root: Path, cpu_max="200000 100000", memory_max=str(8 * GIB)) -> Path:
    _write(
        root,
        {
            "cgroup.controllers": "cpu memory",
            "cpu.max": cpu_max,
            "memory.max": memory_max,
            "cpu.stat": "usage_usec 5000000\nuser_usec 4000000\nsystem_usec 1000000\n"
            "nr_periods 100\nnr_throttled 25\nthrottled_usec 750000\n",
            "memory.current": str(512 * 1024 * 1024),
            "memory.peak": str(600 * 1024 * 1024),
            "memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n",
        },
    )
    return root


def _no_proc(tmp_path: Path) -> Path:
    return tmp_path / "proc_self_cgroup_inexistente"


def test_v2_limits_of_a_2c_8g_container(tmp_path: Path):
    cgroup = Cgroup.detect(_v2_tree(tmp_path / "cg"), _no_proc(tmp_path))

    assert cgroup.limits() == CgroupLimits("v2", 2.0, 8192.0)


def test_v2_without_limits_reports_none(tmp_path: Path):
    cgroup = Cgroup.detect(_v2_tree(tmp_path / "cg", cpu_max="max 100000", memory_max="max"), _no_proc(tmp_path))

    assert cgroup.limits() == CgroupLimits("v2", None, None)


def test_v2_counters(tmp_path: Path):
    counters = Cgroup.detect(_v2_tree(tmp_path / "cg"), _no_proc(tmp_path)).counters()

    assert counters.cpu_usage_s == pytest.approx(5.0)
    assert counters.nr_periods == 100
    assert counters.nr_throttled == 25
    assert counters.throttled_s == pytest.approx(0.75)
    assert counters.memory_current_mb == pytest.approx(512)
    assert counters.memory_peak_mb == pytest.approx(600)
    assert counters.oom_kill == 0


def test_v2_follows_the_own_cgroup_path_outside_a_container(tmp_path: Path):
    """En un host (sin namespace de cgroup) los límites están en la subcarpeta propia."""
    root = tmp_path / "cg"
    _write(root, {"cgroup.controllers": "cpu memory", "cpu.max": "max 100000"})
    _v2_tree(root / "system.slice" / "bench.scope", cpu_max="150000 100000")
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/system.slice/bench.scope\n", encoding="ascii")

    assert Cgroup.detect(root, proc).limits().cpu_limit_cores == pytest.approx(1.5)


def test_v2_kernel_without_memory_peak(tmp_path: Path):
    root = _v2_tree(tmp_path / "cg")
    (root / "memory.peak").unlink()  # memory.peak llegó en Linux 5.19

    assert Cgroup.detect(root, _no_proc(tmp_path)).counters().memory_peak_mb is None


def test_v1_limits_and_counters(tmp_path: Path):
    root = tmp_path / "cg"
    _write(
        root / "cpu,cpuacct",
        {
            "cpu.cfs_quota_us": "200000",
            "cpu.cfs_period_us": "100000",
            "cpuacct.usage": "3000000000",
            "cpu.stat": "nr_periods 40\nnr_throttled 4\nthrottled_time 200000000\n",
        },
    )
    _write(
        root / "memory",
        {
            "memory.limit_in_bytes": str(8 * GIB),
            "memory.usage_in_bytes": str(256 * 1024 * 1024),
            "memory.max_usage_in_bytes": str(300 * 1024 * 1024),
            "memory.oom_control": "oom_kill_disable 0\nunder_oom 0\noom_kill 1\n",
        },
    )
    cgroup = Cgroup.detect(root, _no_proc(tmp_path))
    counters = cgroup.counters()

    assert cgroup.limits() == CgroupLimits("v1", 2.0, 8192.0)
    assert counters.cpu_usage_s == pytest.approx(3.0)
    assert counters.nr_throttled == 4
    assert counters.throttled_s == pytest.approx(0.2)
    assert counters.memory_peak_mb == pytest.approx(300)
    assert counters.oom_kill == 1


def test_v1_unlimited_is_a_huge_number_not_a_word(tmp_path: Path):
    root = tmp_path / "cg"
    _write(root / "cpu", {"cpu.cfs_quota_us": "-1", "cpu.cfs_period_us": "100000"})
    _write(root / "memory", {"memory.limit_in_bytes": "9223372036854771712"})

    assert Cgroup.detect(root, _no_proc(tmp_path)).limits() == CgroupLimits("v1", None, None)


def test_no_cgroup_at_all(tmp_path: Path):
    """El portátil con Windows: nada que leer, todo None, sin excepciones."""
    cgroup = Cgroup.detect(tmp_path / "vacio", _no_proc(tmp_path))

    assert cgroup.limits() == CgroupLimits(None, None, None)
    assert cgroup.counters() == CgroupCounters()


def test_counters_delta_subtracts_accumulators_and_keeps_final_memory():
    start = CgroupCounters(cpu_usage_s=1.0, nr_periods=10, nr_throttled=1, throttled_s=0.1, memory_peak_mb=100, oom_kill=0)
    end = CgroupCounters(cpu_usage_s=3.5, nr_periods=30, nr_throttled=6, throttled_s=0.4, memory_peak_mb=250, oom_kill=0)

    delta = counters_delta(start, end)

    assert delta["cgroup_cpu_s"] == pytest.approx(2.5)
    assert delta["cgroup_nr_periods"] == 20
    assert delta["cgroup_nr_throttled"] == 5
    assert delta["cgroup_throttled_s"] == pytest.approx(0.3)
    assert delta["cgroup_memory_peak_mb"] == 250
    assert delta["cgroup_oom_kill"] == 0


def test_counters_delta_with_missing_fields_gives_none():
    delta = counters_delta(CgroupCounters(), CgroupCounters(cpu_usage_s=2.0))

    assert delta["cgroup_cpu_s"] is None


def test_effective_cpu_count_uses_the_quota_not_the_host(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)

    assert effective_cpu_count(CgroupLimits("v2", 2.0, None)) == (2, "cgroup_quota")
    assert effective_cpu_count(CgroupLimits("v2", 1.5, None)) == (2, "cgroup_quota")


def test_effective_cpu_count_takes_the_smaller_of_quota_and_affinity(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4, 5, 6, 7}, raising=False)
    assert effective_cpu_count(CgroupLimits("v2", 2.0, None)) == (2, "cgroup_quota")

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0}, raising=False)
    assert effective_cpu_count(CgroupLimits("v2", 2.0, None)) == (1, "cpu_affinity")


def test_render_cpu_count_covers_a_limit_the_cgroup_does_not_show(monkeypatch):
    """Sin cuota visible, la afinidad daría los núcleos del host; RENDER_CPU_COUNT da los del plan."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    monkeypatch.setenv("RENDER_CPU_COUNT", "2")
    assert effective_cpu_count(CgroupLimits("v2", None, None)) == (2, "render_cpu_count")

    monkeypatch.setenv("RENDER_CPU_COUNT", "no-es-un-numero")
    assert effective_cpu_count(CgroupLimits("v2", None, None)) == (16, "cpu_affinity")


def test_effective_cpu_count_without_any_limit_falls_back_to_os(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 12)

    assert effective_cpu_count(CgroupLimits(None, None, None)) == (12, "os_cpu_count")
