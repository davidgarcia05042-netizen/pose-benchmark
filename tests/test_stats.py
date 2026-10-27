"""Resumen estadístico y juicio de aceptabilidad (docs/criterios_rendimiento.md)."""

import math

import pytest

from benchmark import config, stats


def test_percentile_interpolates_like_numpy():
    assert stats.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert stats.percentile([1, 2, 3, 4], 90) == pytest.approx(3.7)
    assert stats.percentile([7], 99) == 7
    assert stats.percentile([], 50) is None


def test_describe_gives_the_t_student_interval():
    result = stats.describe([10.0, 11.0, 12.0])

    assert result["n"] == 3
    assert result["mean"] == pytest.approx(11.0)
    assert result["median"] == pytest.approx(11.0)
    assert result["std"] == pytest.approx(1.0)
    assert result["cv_pct"] == pytest.approx(100 / 11)
    half = 4.303 / math.sqrt(3)
    assert result["ci95_low"] == pytest.approx(11 - half)
    assert result["ci95_high"] == pytest.approx(11 + half)


def test_describe_of_a_single_value_has_no_interval():
    result = stats.describe([5.0])

    assert result["std"] == 0.0
    assert result["ci95_low"] is None


def test_t95_beyond_the_table():
    assert stats.t95(1) == pytest.approx(12.706)
    assert stats.t95(200) == pytest.approx(1.96)
    with pytest.raises(ValueError):
        stats.t95(0)


def _row(**overrides) -> dict:
    row = {
        "environment": "render",
        "concurrency": "2",
        "model": "mediapipe_heavy",
        "video_id": "2EJ1",
        "phase": "measured",
        "processing_s": "40",
        "job_s": "40.5",
        "model_init_s": "0.5",
        "ms_per_frame": "123",
        "realtime_factor": "3.6",
        "cpu_cores_used": "1.05",
        "ram_mb_peak": "520",
        "detection_rate": "1.0",
        "video_duration_s": "10.9",
    }
    row.update(overrides)
    return row


def test_summarize_ignores_warmup_and_groups_by_environment_concurrency_model_video():
    rows = [
        _row(processing_s="40"),
        _row(processing_s="41"),
        _row(processing_s="42"),
        _row(phase="warmup", processing_s="90"),
        _row(model="mediapipe_lite", processing_s="12"),
        _row(environment="docker", processing_s="30"),
    ]

    summaries = {(s["environment"], s["model"]): s for s in stats.summarize(rows)}

    assert summaries[("render", "mediapipe_heavy")]["n"] == 3
    assert summaries[("render", "mediapipe_heavy")]["processing_s_median"] == pytest.approx(41)
    assert summaries[("render", "mediapipe_lite")]["n"] == 1
    assert summaries[("docker", "mediapipe_heavy")]["n"] == 1


def test_ram_drift_is_how_much_the_starting_ram_grew_between_jobs():
    rows = [_row(ram_start_mb="250"), _row(ram_start_mb="360"), _row(ram_start_mb="440"), _row(phase="warmup", ram_start_mb="100")]

    assert stats.summarize(rows)[0]["ram_drift_mb"] == pytest.approx(190)


def test_max_video_duration_within_the_shutdown_window():
    summary = stats.summarize([_row(), _row(), _row()])[0]

    expected = (config.SERVICE_MAX_SHUTDOWN_DELAY_S - 0.5) / 3.6
    assert summary["max_video_s_within_shutdown"] == pytest.approx(expected)
    assert summary["ram_worker_estimate_mb"] == pytest.approx(520 * config.SERVICE_WORKER_CONCURRENCY)
    assert summary["throughput_videos_per_hour"] == pytest.approx(2 * 3600 / 40.5)


@pytest.mark.parametrize(
    ("job_s_max", "verdict"),
    [(120, stats.ACCEPTABLE), (300, stats.ACCEPTABLE), (301, stats.CONDITIONAL), (900, stats.CONDITIONAL), (901, stats.NOT_ACCEPTABLE)],
)
def test_time_verdict_follows_the_service_limits(job_s_max, verdict):
    assert stats.evaluate({"job_s_max": job_s_max, "n": 3, "concurrency": 2})["verdict_time"] == verdict


@pytest.mark.parametrize(
    ("ram_worker_mb", "verdict"),
    [(1000, stats.ACCEPTABLE), (8192 * 0.8, stats.ACCEPTABLE), (7000, stats.CONDITIONAL), (8192, stats.CONDITIONAL), (8200, stats.NOT_ACCEPTABLE)],
)
def test_ram_verdict_leaves_a_safety_margin(ram_worker_mb, verdict):
    assert stats.evaluate({"ram_worker_estimate_mb": ram_worker_mb, "n": 3, "concurrency": 2})["verdict_ram"] == verdict


def test_a_stable_measurement_is_valid():
    result = stats.evaluate({"n": 3, "processing_s_cv_pct": 1.0, "detection_rate_min": 0.98, "concurrency": 2})

    assert result["measurement_valid"] is True
    assert result["warnings"] == ""


def test_high_variability_few_repetitions_or_no_detection_invalidate_the_measurement():
    result = stats.evaluate({"n": 2, "processing_s_cv_pct": 9.0, "detection_rate_min": 0.1, "concurrency": 2})

    assert result["measurement_valid"] is False
    assert "repeticiones" in result["warnings"]
    assert "CV" in result["warnings"]
    assert "detección" in result["warnings"]


def test_measuring_one_video_at_a_time_is_valid_but_noted():
    result = stats.evaluate({"n": 3, "processing_s_cv_pct": 1.0, "detection_rate_min": 1.0, "concurrency": 1})

    assert result["measurement_valid"] is True
    assert "concurrencia 1" in result["warnings"]


def test_a_video_cut_with_max_frames_is_flagged():
    cut = stats.summarize([_row(frames_processed="90", video_frames_reported="325")] * 3)[0]
    whole = stats.summarize([_row(frames_processed="324", video_frames_reported="325")] * 3)[0]

    assert cut["video_truncated"] is True and "recortado" in cut["warnings"]
    assert whole["video_truncated"] is False and "recortado" not in whole["warnings"]


def test_format_summary_is_one_line():
    summary = stats.summarize([_row(), _row(), _row()])[0]

    assert "\n" not in stats.format_summary(summary)
