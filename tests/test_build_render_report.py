"""
El informe solo de Render, con corridas sintéticas: se prueba que compara los
modelos entre sí (orden, eficiencia, dos a la vez) y que no necesita al PC.
"""

import pytest

pytest.importorskip("docx")
pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from scripts import build_render_report, build_report  # noqa: E402
from tests.test_build_report import _text, _write_run  # noqa: E402


def test_render_report_ranks_the_models_and_compares_concurrency(tmp_path):
    render = _write_run(tmp_path / "render", "render", {"yolo26n": 104, "mediapipe_lite": 30})
    render_c2 = _write_run(tmp_path / "render_c2", "render", {"yolo26n": 150, "mediapipe_lite": 40})
    out = build_render_report.build(build_report.Run(build_report.RENDER, render),
                                    build_report.Run(build_report.RENDER_C2, render_c2),
                                    tmp_path / "informe_render.docx", None, ["una nota"])

    text = _text(out)
    assert "Estado del informe: final." in text
    fastest_first = text.split("Del más rápido al más lento")[1]
    assert fastest_first.index("MediaPipe Lite") < fastest_first.index("YOLO26-nano")
    assert "MediaPipe Lite es el más eficiente" in text
    assert "5. Dos videos a la vez" in text
    assert "US$ por 1.000 videos" in text
    assert "Nota. una nota" in text
    assert "PC (" not in text  # el PC no entra en este informe
    assert (tmp_path / "informe_render_figuras" / "06_videos_por_hora.png").is_file()


def test_render_report_without_the_concurrency_2_run_says_it_is_pending(tmp_path):
    render = _write_run(tmp_path / "render", "render", {"yolo26n": 104, "mediapipe_lite": 30})
    out = build_render_report.build(build_report.Run(build_report.RENDER, render), None, tmp_path / "i.docx", None)

    assert "Pendiente: la corrida con --concurrency 2" in _text(out)


def test_bold_markers_work_anywhere_in_a_bullet(tmp_path):
    report = build_report.Report()
    report.bullets(["inicio **negrita uno** medio **negrita dos** fin", "sin **cerrar"])

    first, second = report.doc.paragraphs[-2], report.doc.paragraphs[-1]
    assert [(run.text, bool(run.bold)) for run in first.runs] == [
        ("inicio ", False), ("negrita uno", True), (" medio ", False), ("negrita dos", True), (" fin", False)]
    assert second.text == "sin **cerrar"
