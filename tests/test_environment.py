"""Reporte del entorno: etiqueta, versiones contra el servicio, orientación."""

import pytest

from benchmark import config, environment, results


def _snapshot(**overrides) -> dict:
    info = {
        "python_version": "3.12.5",
        "versions": dict(config.SERVICE_PINNED_VERSIONS),
        "cv2_version": config.SERVICE_CV2_VERSION,
        "opencv_distributions": ["opencv-contrib-python==4.11.0.86"],
    }
    info.update(overrides)
    return info


def test_no_mismatches_when_everything_matches_the_service():
    assert environment.version_mismatches(_snapshot()) == []


def test_another_opencv_next_to_the_contrib_is_flagged():
    problems = environment.version_mismatches(
        _snapshot(opencv_distributions=["opencv-contrib-python==4.11.0.86", "opencv-python-headless==4.10.0.84"])
    )

    assert len(problems) == 1 and "opencv-python-headless" in problems[0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"python_version": "3.11.9"},
        {"cv2_version": "4.10.0"},
        {"versions": dict(config.SERVICE_PINNED_VERSIONS, mediapipe="0.10.20")},
        {"versions": dict(config.SERVICE_PINNED_VERSIONS, numpy=None)},
    ],
)
def test_any_version_difference_is_flagged(overrides):
    assert len(environment.version_mismatches(_snapshot(**overrides))) == 1


def test_environment_label(monkeypatch):
    monkeypatch.delenv(environment.ENVIRONMENT_OVERRIDE_VARIABLE, raising=False)
    monkeypatch.setenv("RENDER", "true")
    assert environment.detect_environment_label() == "render"

    monkeypatch.setenv(environment.ENVIRONMENT_OVERRIDE_VARIABLE, "docker_2cpu_8gb")
    assert environment.detect_environment_label() == "docker_2cpu_8gb"


def test_snapshot_and_row_fields():
    info = environment.snapshot()
    row = environment.row_fields(info)

    assert info["budget"] == {"plan": "2c-8g", "cpu_cores": 2, "ram_mb": 8192}
    assert info["cpu_effective"] >= 1
    assert set(row) <= set(results.ROW_COLUMNS)
    assert row["versions_match_service"] == (not info["version_mismatches"])


def test_git_commit_comes_from_render_or_from_the_image(monkeypatch):
    """Con una imagen de Docker Hub, Render no define RENDER_GIT_COMMIT: lo trae la imagen."""
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    monkeypatch.delenv(environment.GIT_COMMIT_VARIABLE, raising=False)
    assert environment.git_commit() is None

    monkeypatch.setenv(environment.GIT_COMMIT_VARIABLE, "5a87fa0")
    assert environment.row_fields(environment.snapshot())["git_commit"] == "5a87fa0"

    monkeypatch.setenv("RENDER_GIT_COMMIT", "abc1234def")
    assert environment.git_commit() == "abc1234def"


def test_snapshot_can_include_the_full_package_list():
    packages = environment.snapshot(include_packages=True)["packages"]

    assert any(entry.lower().startswith("psutil==") for entry in packages)


def test_open_video_applies_the_rotation_in_this_environment():
    """Lo mismo que comprueba el build de la imagen con --check."""
    assert environment.orientation_is_applied() is True


def test_the_report_names_the_budget_and_the_mismatches():
    info = environment.snapshot()
    info["version_mismatches"] = ["numpy 2.0.0 (el servicio usa 1.26.4)"]

    report = environment.format_report(info)

    assert "2c-8g" in report
    assert "numpy 2.0.0" in report


def test_main_check_fails_on_mismatches(monkeypatch, capsys):
    real = environment.snapshot
    monkeypatch.setattr(environment, "snapshot", lambda: dict(real(), version_mismatches=["x"]))

    assert environment.main(["--check"]) == 1
    assert environment.main([]) == 0
