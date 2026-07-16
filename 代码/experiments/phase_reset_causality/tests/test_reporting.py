from phase_causal.reporting import paired_effect_rows, validate_unique_records


def test_result_records_are_unique_by_experimental_key():
    rows = [
        {"model": "sprif", "seed": 1, "mode": "fast_reset", "k": 8, "gamma": 1.0},
        {"model": "sprif", "seed": 1, "mode": "slow_reset", "k": 8, "gamma": 1.0},
    ]
    validate_unique_records(rows)


def test_duplicate_result_records_are_rejected():
    row = {"model": "sprif", "seed": 1, "mode": "fast_reset", "k": 8, "gamma": 1.0}

    try:
        validate_unique_records([row, dict(row)])
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("duplicate rows must be rejected")


def test_paired_effect_rows_compute_slow_minus_fast_per_seed():
    rows = [
        {"model": "sprif", "seed": 1, "mode": "fast_reset", "k": 8,
         "gamma": 1.0, "event_step": 200, "excess_auc": 1.0},
        {"model": "sprif", "seed": 1, "mode": "slow_reset", "k": 8,
         "gamma": 1.0, "event_step": 200, "excess_auc": 3.5},
    ]

    paired = paired_effect_rows(rows)

    assert paired == [{
        "seed": 1, "k": 8, "gamma": 1.0, "event_step": 200,
        "fast_excess_auc": 1.0, "slow_excess_auc": 3.5,
        "slow_minus_fast_auc": 2.5,
    }]


def test_paired_effect_rows_keep_sweeps_and_datasets_separate():
    rows = []
    for dataset, sweep, offset in (
        ("in_distribution", "main", 0.0),
        ("unseen_frequency", "ood_frequency", 10.0),
    ):
        rows.extend([
            {"model": "sprif", "seed": 1, "dataset": dataset, "sweep": sweep,
             "event_count": 1, "mode": "fast_reset", "k": 8, "gamma": 1.0,
             "event_step": 200, "excess_auc": 1.0 + offset},
            {"model": "sprif", "seed": 1, "dataset": dataset, "sweep": sweep,
             "event_count": 1, "mode": "slow_reset", "k": 8, "gamma": 1.0,
             "event_step": 200, "excess_auc": 2.0 + offset},
        ])

    paired = paired_effect_rows(rows)

    assert len(paired) == 2
    assert {row["sweep"] for row in paired} == {"main", "ood_frequency"}
