import colorsys
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).with_name("plot_si_dms_composite.py")
SPEC = importlib.util.spec_from_file_location("plot_si_dms_composite", SCRIPT)
PLOT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PLOT)


def complete_run(model="sprif_full", seed=1):
    rows = []
    for delay in (200, 400, 800, 1600, 2500):
        rows.append({"model": model, "seed": seed, "delay_ms": delay,
                     "intervention_count": 0, "intervention_fraction": 0.0,
                     "accuracy": 0.9, "forced_hit_rate": 0.0})
        for count in (1, 2, 4, 8, 16, 32, 40):
            for fraction in (.05, .10, .15, .20, .30, .50):
                rows.append({"model": model, "seed": seed, "delay_ms": delay,
                             "intervention_count": count, "intervention_fraction": fraction,
                             "accuracy": 0.9, "forced_hit_rate": 1.0})
    return rows


def test_primary_slice_keeps_clean_and_only_fifteen_percent_interventions():
    frame = pd.DataFrame(
        [
            {"intervention_count": 0, "intervention_fraction": 0.0},
            {"intervention_count": 8, "intervention_fraction": 0.10},
            {"intervention_count": 8, "intervention_fraction": 0.15},
            {"intervention_count": 8, "intervention_fraction": 0.50},
        ]
    )

    selected = PLOT.select_primary_slice(frame, fraction=0.15)

    assert selected[["intervention_count", "intervention_fraction"]].to_dict("records") == [
        {"intervention_count": 0, "intervention_fraction": 0.0},
        {"intervention_count": 8, "intervention_fraction": 0.15},
    ]


def test_load_data_preserves_intervention_fraction(tmp_path, monkeypatch):
    run_dir = tmp_path / "sprif_full" / "seed_1"
    run_dir.mkdir(parents=True)
    (run_dir / "eval_metrics.json").write_text(
        json.dumps(
            [
                {
                    "model": "sprif_full",
                    "seed": 1,
                    "delay_ms": 200,
                    "intervention_count": 8,
                    "intervention_fraction": 0.15,
                    "accuracy": 0.9,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(PLOT, "RESULTS_DIR", tmp_path)

    loaded = PLOT.load_data()

    assert loaded.loc[0, "intervention_fraction"] == pytest.approx(0.15)


def test_per_k_stats_average_delays_within_seed_before_seed_std():
    frame = pd.DataFrame(
        [
            {"model": "m", "seed": 1, "delay_ms": 200, "intervention_count": 8, "accuracy": 1.0},
            {"model": "m", "seed": 1, "delay_ms": 400, "intervention_count": 8, "accuracy": 0.6},
            {"model": "m", "seed": 2, "delay_ms": 200, "intervention_count": 8, "accuracy": 0.8},
            {"model": "m", "seed": 2, "delay_ms": 400, "intervention_count": 8, "accuracy": 0.8},
            {"model": "m", "seed": 3, "delay_ms": 200, "intervention_count": 8, "accuracy": 0.6},
            {"model": "m", "seed": 3, "delay_ms": 400, "intervention_count": 8, "accuracy": 1.0},
        ]
    )

    means, stds = PLOT.per_k_seed_stats(frame, "m", [8])

    assert means[0] == pytest.approx(0.8)
    assert stds[0] == pytest.approx(0.0)


def test_validate_run_entries_accepts_exact_eval_grid():
    PLOT.validate_run_entries(complete_run(), "sprif_full", 1)


def test_replay_cue2_natural_spike_count_is_stable():
    trace = PLOT.replay_trial()
    cue2_start = (trace["pre"] + trace["cue"] + trace["delay"]) // trace["dt"]

    assert int(trace["spikes"][cue2_start:].sum()) == 203


def test_clean_and_stressed_replays_share_the_exact_same_input():
    clean = PLOT.replay_trial(intervention_count=0)
    stressed = PLOT.replay_trial(intervention_count=8)

    assert np.array_equal(clean["input"], stressed["input"])
    assert clean["first"] == stressed["first"] == 0
    assert clean["second"] == stressed["second"] == 0
    assert clean["target"] == clean["pred"] == 1
    assert stressed["target"] == stressed["pred"] == 1
    assert clean["hits"].sum() == 0
    assert stressed["hits"].sum() == 80


def test_paired_raster_selection_uses_same_non_tonic_units_for_both_conditions():
    clean = PLOT.replay_trial(intervention_count=0)
    stressed = PLOT.replay_trial(intervention_count=8)
    units = PLOT.select_paired_raster_units(clean, stressed, n_units=10,
                                            max_delay_occupancy=0.40)

    assert len(units) == 10
    for trace in (clean, stressed):
        raw_times = PLOT._time_axis(trace)
        display_mask = raw_times >= trace["pre"]
        times = raw_times[display_mask] - trace["pre"]
        natural = ((trace["spikes"] > 0)[display_mask]
                   & ~trace["hits"][display_mask])
        delay_mask = ((times >= trace["cue"])
                      & (times < trace["cue"] + trace["delay"]))
        assert natural[delay_mask][:, units].mean(axis=0).max() <= 0.40


def test_paired_panel_labels_clean_and_stress_and_compares_cue2():
    clean = PLOT.replay_trial(intervention_count=0)
    stressed = PLOT.replay_trial(intervention_count=8)
    fig, axes = plt.subplots(1, 4)

    PLOT.plot_paired_trial(*axes, clean, stressed)

    assert axes[0].get_ylabel() == "Clean"
    assert axes[1].get_ylabel() == "$K=8$"
    assert axes[2].get_title(loc="left") == "Cue-2 natural spikes"
    assert axes[3].get_title(loc="left") == "Cue-2 match readout"
    assert [line.get_label() for line in axes[2].lines] == ["Clean", "$K=8$"]
    assert [line.get_label() for line in axes[3].lines] == ["Clean", "$K=8$"]
    plt.close(fig)


def test_raster_selection_excludes_tonic_delay_units_but_keeps_cue2_activity():
    trace = PLOT.replay_trial()
    units = PLOT.select_raster_units(trace, n_units=10, max_delay_occupancy=0.40)
    raw_times = PLOT._time_axis(trace)
    display_mask = raw_times >= trace["pre"]
    times = raw_times[display_mask] - trace["pre"]
    natural = (trace["spikes"] > 0)[display_mask]
    delay_mask = (times >= trace["cue"]) & (times < trace["cue"] + trace["delay"])
    cue2_mask = times >= trace["cue"] + trace["delay"]

    assert len(units) == 10
    assert natural[delay_mask][:, units].mean(axis=0).max() <= 0.40
    assert np.count_nonzero(natural[cue2_mask][:, units].sum(axis=0)) >= 5


def test_matching_cues_use_the_same_visual_encoding_and_identity_label():
    trace = PLOT.replay_trial()
    fig, (raster_ax, readout_ax, zoom_ax) = plt.subplots(1, 3)

    PLOT.plot_trial(raster_ax, trace, readout_ax, zoom_ax)

    task_strip = [patch for patch in raster_ax.patches
                  if isinstance(patch, Rectangle) and np.isclose(patch.get_y(), 10.2)]
    strip_labels = [text.get_text() for text in raster_ax.texts]
    assert np.allclose(task_strip[0].get_facecolor(), task_strip[2].get_facecolor())
    assert "cue 1 (A)" in strip_labels
    assert "cue 2 (A)" in strip_labels
    assert zoom_ax.get_title(loc="left") == "Cue 2 (A) zoom"
    plt.close(fig)


def test_caption_level_metadata_is_not_drawn_inside_main_panels():
    trace = {
        "spikes": np.zeros((370, 64)),
        "hits": np.zeros((370, 64), dtype=bool),
        "logits": np.zeros((370, 2)),
        "intervention": np.zeros(370, dtype=bool),
        "dt": 10,
        "pre": 100,
        "cue": 1000,
        "delay": 1600,
    }
    trace["spikes"][270:370:10, 0] = 1
    fig, (trial_ax, readout_ax, zoom_ax, accuracy_ax) = plt.subplots(1, 4)
    PLOT.plot_trial(trial_ax, trace, readout_ax, zoom_ax)

    rows = []
    for model in PLOT.LABELS:
        for seed in (1, 2, 3):
            rows.extend(complete_run(model=model, seed=seed))
    PLOT.plot_accuracy(accuracy_ax, PLOT.select_primary_slice(pd.DataFrame(rows)))

    panel_text = " ".join(text.get_text() for ax in fig.axes for text in ax.texts)
    plt.close(fig)
    assert "SPRiF full | seed 1, sample 0" not in panel_text
    assert "delay 1600 ms" not in panel_text
    assert "unseen" not in panel_text
    assert "Accuracy drop at" not in panel_text


def test_trial_detail_plots_use_external_axes_without_overlay_text():
    trace = {
        "spikes": np.zeros((370, 64)),
        "hits": np.zeros((370, 64), dtype=bool),
        "logits": np.zeros((370, 2)),
        "intervention": np.zeros(370, dtype=bool),
        "dt": 10,
        "pre": 100,
        "cue": 1000,
        "delay": 1600,
    }
    trace["spikes"][270:370:10, 0] = 1
    fig, (raster_ax, readout_ax, zoom_ax) = plt.subplots(1, 3)

    PLOT.plot_trial(raster_ax, trace, readout_ax, zoom_ax)

    all_text = " ".join(text.get_text() for ax in fig.axes for text in ax.texts)
    task_strip = [patch for patch in raster_ax.patches
                  if isinstance(patch, Rectangle) and np.isclose(patch.get_y(), 10.2)]
    assert not raster_ax.child_axes
    assert len(task_strip) == 3
    assert task_strip[0].get_x() == pytest.approx(0.0)
    assert readout_ax.get_title(loc="left") == "Readout probability"
    assert zoom_ax.get_title(loc="left") == "Cue 2 (A) zoom"
    legend = readout_ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["match", "non-match"]
    assert 0.45 <= legend.get_frame().get_alpha() <= 0.65
    assert "natural" not in all_text
    assert "forced" not in all_text
    assert "target = prediction" not in all_text
    plt.close(fig)


def test_brf_and_lif_use_distinct_non_neutral_line_colors():
    brf_hsv = colorsys.rgb_to_hsv(*to_rgb(PLOT.COLORS["brf"]))
    lif_hsv = colorsys.rgb_to_hsv(*to_rgb(PLOT.COLORS["lif"]))
    hue_distance = min(abs(brf_hsv[0] - lif_hsv[0]), 1 - abs(brf_hsv[0] - lif_hsv[0]))

    assert brf_hsv[1] >= 0.25
    assert lif_hsv[1] >= 0.25
    assert hue_distance >= 0.15


def test_left_column_vertical_spacing_is_compact():
    assert PLOT.LEFT_COLUMN_HSPACE <= 0.28


def test_model_legend_is_anchored_to_left_column():
    assert PLOT.MODEL_LEGEND_NCOL == 3
    assert PLOT.MODEL_LEGEND_X <= 0.30


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.__setitem__(0, {**rows[0], "model": "lif"}), "model mismatch"),
        (lambda rows: rows.__setitem__(-1, dict(rows[0])), "duplicate evaluation key"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "intervention_count": 3}), "evaluation grid mismatch"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "forced_hit_rate": 0.5}), "forced_hit_rate"),
    ],
)
def test_validate_run_entries_rejects_mislabeled_duplicate_or_invalid_grid(mutate, message):
    rows = complete_run()
    mutate(rows)

    with pytest.raises(RuntimeError, match=message):
        PLOT.validate_run_entries(rows, "sprif_full", 1)
