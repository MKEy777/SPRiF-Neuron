from pathlib import Path

import matplotlib
from matplotlib.text import Annotation

matplotlib.use("Agg")

from plot_combined_figures import (
    build_mechanism_figure,
    build_temporal_figure,
    export_figure,
    load_source_data,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _panel_labels(fig):
    labels = {text.get_text() for text in fig.texts}
    for ax in fig.axes:
        labels.update(text.get_text() for text in ax.texts)
    return labels


def test_load_source_data_covers_all_four_analysis_directories():
    data = load_source_data(REPO_ROOT)

    assert data["real_trajectory"]["layer1_slow"].shape == (784, 256, 3)
    assert data["controlled_trajectory"]["sprif_x_t"].shape == (900, 64, 3)
    assert len(data["reset_stats"]) == 656
    assert data["impulse"]["GSC_L0_slow_resp"].shape == (300, 100, 3)
    assert data["asrnn"]["ECG_col2_meta"].shape == (5,)
    assert data["source_dirs"] == {
        "trajectory_visualization",
        "trajectory_analysis",
        "reset_analysis",
        "impulse_analysis",
    }


def test_mechanism_figure_has_approved_panels_and_size():
    fig = build_mechanism_figure(load_source_data(REPO_ROOT))

    assert {"a", "b", "c", "d", "e"}.issubset(_panel_labels(fig))
    annotations = [
        child
        for ax in fig.axes
        for child in ax.get_children()
        if isinstance(child, Annotation) and child.arrow_patch is not None
    ]
    assert annotations, "projective-reset panel must contain at least one reset arrow"
    width, height = fig.get_size_inches()
    assert width <= 7.2
    assert height <= 3.8


def test_temporal_figure_has_approved_panels_and_size():
    fig = build_temporal_figure(load_source_data(REPO_ROOT))

    assert {"a", "b", "c"}.issubset(_panel_labels(fig))
    width, height = fig.get_size_inches()
    assert width <= 7.2
    assert height <= 3.8


def test_export_figure_writes_editable_vector_and_raster_files(tmp_path):
    fig = build_mechanism_figure(load_source_data(REPO_ROOT))
    paths = export_figure(fig, tmp_path / "mechanism", dpi=120)

    assert {path.suffix for path in paths} == {".svg", ".pdf", ".png"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert "<text" in (tmp_path / "mechanism.svg").read_text(encoding="utf-8")
