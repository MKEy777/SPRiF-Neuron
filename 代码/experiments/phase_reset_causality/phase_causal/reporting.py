from __future__ import annotations


RESULT_KEY = (
    "model", "seed", "dataset", "sweep", "mode", "k", "gamma",
    "event_step", "event_count",
)


def validate_unique_records(rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        key = tuple(row.get(field) for field in RESULT_KEY)
        if key in seen:
            raise ValueError(f"duplicate experimental result key: {key}")
        seen.add(key)


def paired_effect_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        if row.get("model") != "sprif" or row.get("mode") not in {"fast_reset", "slow_reset"}:
            continue
        key = (
            row.get("dataset"), row.get("sweep"), row.get("event_count"),
            row["seed"], row["k"], row["gamma"], row["event_step"],
        )
        grouped.setdefault(key, {})[row["mode"]] = row
    result = []
    for key, modes in sorted(grouped.items(), key=lambda item: repr(item[0])):
        if set(modes) != {"fast_reset", "slow_reset"}:
            continue
        dataset, sweep, event_count, seed, k, gamma, event_step = key
        fast = float(modes["fast_reset"]["excess_auc"])
        slow = float(modes["slow_reset"]["excess_auc"])
        output = {
            "seed": seed,
            "k": k,
            "gamma": gamma,
            "event_step": event_step,
            "fast_excess_auc": fast,
            "slow_excess_auc": slow,
            "slow_minus_fast_auc": slow - fast,
        }
        if dataset is not None:
            output["dataset"] = dataset
        if sweep is not None:
            output["sweep"] = sweep
        if event_count is not None:
            output["event_count"] = event_count
        result.append(output)
    return result
