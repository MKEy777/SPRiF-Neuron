from sidms.reporting import summarize


def test_summary_keeps_baselines_separate_and_computes_stress_drop():
    rows = [
        {"model": "sprif_full", "seed": 1, "delay_ms": 100, "intervention_count": 0, "accuracy": .9},
        {"model": "sprif_full", "seed": 1, "delay_ms": 100, "intervention_count": 2, "accuracy": .7},
        {"model": "lif", "seed": 1, "delay_ms": 100, "intervention_count": 0, "accuracy": .8},
        {"model": "lif", "seed": 1, "delay_ms": 100, "intervention_count": 2, "accuracy": .5},
    ]
    summary = summarize(rows)
    full = next(x for x in summary if x["model"] == "sprif_full")
    assert abs(full["stress_drop"] - .2) < 1e-8
    assert full["role"] == "mechanism_ablation"
    assert next(x for x in summary if x["model"] == "lif")["role"] == "external_baseline"
