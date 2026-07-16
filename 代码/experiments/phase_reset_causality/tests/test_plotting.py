import pandas as pd

from phase_causal.plotting import sensitivity_rows


def test_sensitivity_rows_do_not_mix_unrelated_sweeps():
    frame = pd.DataFrame([
        {"model": "sprif", "mode": "fast_reset", "sweep": "main",
         "event_step": 500, "k": 8, "gamma": 1.0},
        {"model": "sprif", "mode": "fast_reset", "sweep": "k_sweep",
         "event_step": 500, "k": 4, "gamma": 1.0},
        {"model": "sprif", "mode": "fast_reset", "sweep": "gamma_sweep",
         "event_step": 500, "k": 8, "gamma": 0.5},
        {"model": "sprif", "mode": "fast_reset", "sweep": "ood_frequency",
         "event_step": 500, "k": 8, "gamma": 1.0},
    ])

    selected = sensitivity_rows(frame, "k_sweep")

    assert set(selected["sweep"]) == {"main", "k_sweep"}
    assert set(selected["k"]) == {4, 8}


def test_frequency_rows_include_seen_and_unseen_sweeps_only():
    frame = pd.DataFrame([
        {"model": "sprif", "mode": "fast_reset", "sweep": "id_frequency"},
        {"model": "sprif", "mode": "slow_reset", "sweep": "ood_frequency"},
        {"model": "sprif", "mode": "fast_reset", "sweep": "main"},
        {"model": "gru", "mode": "fast_reset", "sweep": "id_frequency"},
    ])

    selected = sensitivity_rows(frame, "frequency_all")

    assert set(selected["sweep"]) == {"id_frequency", "ood_frequency"}
    assert set(selected["model"]) == {"sprif"}
