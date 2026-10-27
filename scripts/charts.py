"""
Gráficas y formato compartidos por el análisis local (scripts/analyze_results.py
y scripts/build_report.py). No va en la imagen.

Paleta categórica de referencia (modo claro), en orden FIJO: cada modelo
conserva su color aunque un filtro saque a otro de la gráfica. Los números van
con coma decimal: los documentos del trabajo de grado están en español.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

MODEL_COLORS = {
    "yolo26n": "#2a78d6",
    "mediapipe_lite": "#eb6834",
    "mediapipe_full": "#1baf7a",
    "mediapipe_heavy": "#eda100",
}
OTHER_COLOR = "#8a8984"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE_LINE = "#52514e"

MODEL_LABELS = {
    "yolo26n": "YOLO26-nano",
    "mediapipe_lite": "MediaPipe Lite",
    "mediapipe_full": "MediaPipe Full",
    "mediapipe_heavy": "MediaPipe Heavy",
}


def num(value, decimals: int = 1, suffix: str = "") -> str:
    """Número con coma decimal (el documento va en español); 'n/d' si falta."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/d"
    text = f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return text + suffix


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def ordered_models(models) -> list[str]:
    present = set(models)
    return [m for m in MODEL_COLORS if m in present] + sorted(present - set(MODEL_COLORS))


def geometric_mean(values) -> float | None:
    values = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


_COMMA = FuncFormatter(lambda value, _: num(value, 0 if abs(value) >= 10 or value == int(value) else 1))


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.yaxis.set_major_formatter(_COMMA)


def source_bars(models_frame: pd.DataFrame, column: str, title: str, ylabel: str, out_path: Path,
                colors: dict[str, str], decimals: int = 0, spread: tuple[str, str] | None = None,
                reference: tuple[float, str] | None = None) -> Path:
    """Modelos en el eje X; una barra por fuente (entorno), con el color de `colors`. Una sola escala."""
    models = ordered_models(models_frame["model"])
    sources = [s for s in colors if s in set(models_frame["source"])]
    width = 0.8 / max(1, len(sources))
    fig, ax = plt.subplots(figsize=(6.6, 3.6), facecolor=SURFACE)
    for index, source in enumerate(sources):
        subset = models_frame[models_frame["source"] == source].set_index("model")
        positions = [m + (index - (len(sources) - 1) / 2) * width for m in range(len(models))]
        values = [subset[column].get(model) if model in subset.index else None for model in models]
        heights = [v if v is not None and pd.notna(v) else 0 for v in values]
        bars = ax.bar(positions, heights, width=width * 0.9, color=colors[source], label=source,
                      edgecolor=SURFACE, linewidth=1.5)
        if spread:
            low = [subset[spread[0]].get(m) if m in subset.index else None for m in models]
            high = [subset[spread[1]].get(m) if m in subset.index else None for m in models]
            for x, lo, hi in zip(positions, low, high):
                if lo is not None and hi is not None and pd.notna(lo):
                    ax.plot([x, x], [lo, hi], color=TEXT_PRIMARY, linewidth=1)
                    ax.plot([x - width * 0.12, x + width * 0.12], [hi, hi], color=TEXT_PRIMARY, linewidth=1)
                    ax.plot([x - width * 0.12, x + width * 0.12], [lo, lo], color=TEXT_PRIMARY, linewidth=1)
        for bar, value in zip(bars, values):
            if value is not None and pd.notna(value):
                top = bar.get_height()
                if spread:  # la etiqueta va por encima del bigote
                    hi = subset[spread[1]].get(models[list(bars).index(bar)])
                    top = max(top, hi if hi is not None and pd.notna(hi) else top)
                ax.annotate(num(value, decimals), (bar.get_x() + bar.get_width() / 2, top), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=TEXT_SECONDARY)
    if reference:
        ax.axhline(reference[0], color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.annotate(reference[1], (1, reference[0]), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=TEXT_SECONDARY)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([model_label(m) for m in models])
    _style(ax, title, ylabel)
    ax.margins(y=0.12)
    ax.legend(frameon=False, fontsize=8, ncol=len(sources), loc="upper left", bbox_to_anchor=(0, -0.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def model_bars(frame: pd.DataFrame, column: str, title: str, ylabel: str, out_path: Path, decimals: int = 0,
               spread: tuple[str, str] | None = None, reference: tuple[float, str] | None = None) -> Path:
    """
    Una barra por modelo, con su color fijo. `frame` tiene una fila por modelo
    (columna "model"). Sin leyenda: el nombre de cada modelo va en el eje.
    """
    models = ordered_models(frame["model"])
    subset = frame.set_index("model")
    fig, ax = plt.subplots(figsize=(6.6, 3.4), facecolor=SURFACE)
    values = [subset[column].get(model) for model in models]
    bars = ax.bar(range(len(models)), [v if v is not None and pd.notna(v) else 0 for v in values], width=0.6,
                  color=[MODEL_COLORS.get(m, OTHER_COLOR) for m in models], edgecolor=SURFACE, linewidth=1.5)
    for x, (bar, value, model) in enumerate(zip(bars, values, models)):
        top = bar.get_height()
        if spread:
            lo, hi = subset[spread[0]].get(model), subset[spread[1]].get(model)
            if lo is not None and hi is not None and pd.notna(lo) and pd.notna(hi):
                ax.plot([x, x], [lo, hi], color=TEXT_PRIMARY, linewidth=1)
                for y in (lo, hi):
                    ax.plot([x - 0.07, x + 0.07], [y, y], color=TEXT_PRIMARY, linewidth=1)
                top = max(top, hi)
        if value is not None and pd.notna(value):
            ax.annotate(num(value, decimals), (x, top), xytext=(0, 2), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8, color=TEXT_SECONDARY)
    if reference:
        ax.axhline(reference[0], color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.annotate(reference[1], (1, reference[0]), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=TEXT_SECONDARY)
    _style(ax, title, ylabel)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([model_label(m) for m in models])
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def video_bars(summary: pd.DataFrame, column: str, video_order: list[str], title: str, ylabel: str, out_path: Path,
               reference: tuple[float, str] | None = None, decimals: int = 0) -> Path:
    """Videos en el eje X (orden del manifiesto), una barra por modelo con su color fijo."""
    videos = [v for v in video_order if v in set(summary["video_id"])] + sorted(set(summary["video_id"]) - set(video_order))
    models = ordered_models(summary["model"])
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(9.5, 4.0), facecolor=SURFACE)
    for index, model in enumerate(models):
        subset = summary[summary["model"] == model].set_index("video_id")
        positions = [v + (index - (len(models) - 1) / 2) * width for v in range(len(videos))]
        values = [subset[column].get(video) if video in subset.index else None for video in videos]
        bars = ax.bar(positions, [v if v is not None and pd.notna(v) else 0 for v in values], width=width * 0.9,
                      color=MODEL_COLORS.get(model, OTHER_COLOR), label=model_label(model), edgecolor=SURFACE, linewidth=1.5)
        # Valor sobre cada barra: tres colores de la paleta no llegan a 3:1 de
        # contraste con el fondo, así que el dato no puede depender del color.
        for bar, value in zip(bars, values):
            if value is not None and pd.notna(value):
                ax.annotate(num(value, decimals), (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=TEXT_SECONDARY)
    if reference:
        ax.axhline(reference[0], color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.annotate(reference[1], (1, reference[0]), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=TEXT_SECONDARY)
    ax.set_xticks(range(len(videos)))
    ax.set_xticklabels(videos)
    _style(ax, title, ylabel)
    ax.margins(y=0.1)
    ax.legend(frameon=False, fontsize=8, ncol=len(models), loc="upper left", bbox_to_anchor=(0, -0.08))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def ratio_dots(pairs: pd.DataFrame, out_path: Path) -> Path:
    """Cuántas veces más lento es Render por modelo: un punto por video, raya en la media geométrica."""
    models = ordered_models(pairs["model"])
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(models) + 1.4), facecolor=SURFACE)
    for row, model in enumerate(reversed(models)):
        values = pairs[pairs["model"] == model]["ratio_ms"].tolist()
        ax.scatter(values, [row] * len(values), s=42, color=MODEL_COLORS.get(model, OTHER_COLOR),
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        center = geometric_mean(values)
        if center:
            ax.plot([center, center], [row - 0.28, row + 0.28], color=TEXT_PRIMARY, linewidth=2, zorder=4)
            ax.annotate(f"×{num(center, 2)}", (center, row + 0.3), ha="center", va="bottom", fontsize=7.5,
                        color=TEXT_SECONDARY)
    ax.axvline(1.0, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.annotate("igual que el PC", (1.0, -0.6), ha="left", va="center", fontsize=7.5, color=TEXT_SECONDARY,
                xytext=(4, 0), textcoords="offset points")
    # _style pone el formato numérico en el eje Y; los nombres de los modelos
    # van DESPUÉS, o el formato los reemplaza por 0, 1, 2...
    _style(ax, "Tiempo por frame en Render ÷ tiempo en el PC", "")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([model_label(m) for m in reversed(models)])
    ax.set_ylim(-0.9, len(models) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "×" + num(value, 1)))
    ax.set_xlim(left=min(0.9, pairs["ratio_ms"].min() - 0.05), right=pairs["ratio_ms"].max() + 0.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out_path
