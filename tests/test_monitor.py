"""Monitor de recursos: CPU por tiempo de CPU, pico de RAM exacto o muestreado, deltas del cgroup."""

import time
from pathlib import Path

import pytest

from benchmark.cgroup import CgroupCounters
from benchmark.monitor import ResourceMonitor, read_peak_rss_mb, reset_peak_rss


def _busy(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        sum(i * i for i in range(500))


def test_a_busy_single_thread_uses_about_one_core(tmp_path: Path):
    # Rutas que no existen: fuerza el camino "muestreado" en cualquier SO.
    with ResourceMonitor(interval_s=0.05, proc_status=tmp_path / "x", clear_refs=tmp_path / "no" / "x") as monitor:
        _busy(0.5)
    result = monitor.result

    assert result.wall_s >= 0.5
    assert result.cpu_s > 0
    # Un solo hilo ocupado: cerca de 1 núcleo, nunca 2 (el margen cubre el
    # propio hilo del monitor y la resolución del reloj de CPU en Windows).
    assert 0.5 < result.cpu_cores_used < 1.6
    assert result.samples >= 3
    assert result.cpu_percent_mean is not None
    assert result.ram_peak_source == "sampled"
    assert result.ram_mb_peak == result.ram_mb_peak_sampled >= result.ram_mb_mean > 0


def test_an_idle_block_uses_almost_no_cpu(tmp_path: Path):
    with ResourceMonitor(interval_s=0.05, proc_status=tmp_path / "x", clear_refs=tmp_path / "no" / "x") as monitor:
        time.sleep(0.3)

    assert monitor.result.cpu_cores_used < 0.3


def test_the_exact_peak_is_used_when_it_can_be_reset(tmp_path: Path):
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmHWM:\t 9999999 kB\nVmRSS:\t 1000 kB\n", encoding="ascii")
    clear_refs = tmp_path / "clear_refs"

    with ResourceMonitor(interval_s=0.05, proc_status=status, clear_refs=clear_refs) as monitor:
        time.sleep(0.1)

    assert clear_refs.read_text(encoding="ascii") == "5"  # el reset se pidió
    assert monitor.result.ram_peak_source == "vmhwm"
    assert monitor.result.ram_mb_peak == pytest.approx(9999999 / 1024)


def test_a_block_shorter_than_the_interval_still_reports_cpu_and_ram(tmp_path: Path):
    """Sin muestras de psutil, el tiempo de CPU y el RSS de entrada/salida siguen ahí."""
    with ResourceMonitor(interval_s=10, proc_status=tmp_path / "x", clear_refs=tmp_path / "no" / "x") as monitor:
        _busy(0.05)

    assert monitor.result.samples == 0
    assert monitor.result.cpu_percent_mean is None
    assert monitor.result.cpu_cores_used is not None
    assert monitor.result.ram_mb_peak > 0


def test_cgroup_deltas_are_attached(tmp_path: Path):
    class FakeCgroup:
        def __init__(self):
            self.snapshots = [
                CgroupCounters(cpu_usage_s=10.0, nr_periods=100, nr_throttled=10, throttled_s=1.0),
                CgroupCounters(cpu_usage_s=12.0, nr_periods=120, nr_throttled=15, throttled_s=1.5, memory_peak_mb=700),
            ]

        def counters(self):
            return self.snapshots.pop(0)

    with ResourceMonitor(interval_s=0.05, cgroup=FakeCgroup(), proc_status=tmp_path / "x", clear_refs=tmp_path / "no" / "x") as monitor:
        time.sleep(0.05)
    row = monitor.result.as_row()

    assert row["cgroup_cpu_s"] == pytest.approx(2.0)
    assert row["cgroup_nr_periods"] == 20
    assert row["cgroup_nr_throttled"] == 5
    assert row["cgroup_memory_peak_mb"] == 700
    assert "cgroup" not in row  # aplanado en la fila


def test_read_peak_rss_mb_parses_kb(tmp_path: Path):
    status = tmp_path / "status"
    status.write_text("VmPeak:\t 5000 kB\nVmHWM:\t 2048 kB\n", encoding="ascii")

    assert read_peak_rss_mb(status) == pytest.approx(2.0)
    assert read_peak_rss_mb(tmp_path / "no_existe") is None


def test_reset_peak_rss_reports_failure(tmp_path: Path):
    assert reset_peak_rss(tmp_path / "no" / "clear_refs") is False
