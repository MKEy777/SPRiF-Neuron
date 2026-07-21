import random
from collections import defaultdict

MECHANISM_MODELS = {"sprif_full", "sprif_merged", "sprif_lambda0"}
REFERENCE_MODEL = "sprif_full"
CHANCE = 0.5

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def _per_seed_metrics(values, intervention_fraction=0.15):
    by_seed = defaultdict(list)
    for r in values:
        by_seed[r["seed"]].append(r)
    max_k = max(r["intervention_count"] for r in values)
    per_seed = {}
    for seed, rows in by_seed.items():
        clean = _mean([r["accuracy"] for r in rows if r["intervention_count"] == 0])
        stress = _mean([r["accuracy"] for r in rows
                        if r["intervention_count"] == max_k
                        and r.get("intervention_fraction", 0.0) == intervention_fraction])
        headroom = clean - CHANCE
        rel = (clean - stress) / headroom if headroom > 1e-6 else float("nan")
        per_seed[seed] = {"clean": clean, "stress": stress,
                          "abs_drop": clean - stress, "rel_drop": rel}
    return per_seed

def _bootstrap_ci(samples, reps=2000, alpha=0.05, seed=0):
    if len(samples) < 2:
        v = samples[0] if samples else float("nan")
        return v, v
    rng = random.Random(seed)
    means = []
    n = len(samples)
    for _ in range(reps):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(resample))
    means.sort()
    lo = means[int((alpha / 2) * reps)]
    hi = means[int((1 - alpha / 2) * reps)]
    return lo, hi

def _bootstrap_pvalue(ref, other, reps=2000, seed=0):
    if len(ref) < 2 or len(other) < 2:
        return float("nan")
    rng = random.Random(seed)
    observed = _mean(ref) - _mean(other)
    pooled = ref + other
    n_ref, n = len(ref), len(pooled)
    count = 0
    for _ in range(reps):
        perm = [pooled[rng.randrange(n)] for _ in range(n)]
        diff = _mean(perm[:n_ref]) - _mean(perm[n_ref:])
        if abs(diff) >= abs(observed):
            count += 1
    return count / reps

def summarize(rows, intervention_fraction=0.15):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    per_model = {m: _per_seed_metrics(v, intervention_fraction) for m, v in grouped.items()}
    ref_drops = [s["abs_drop"] for s in per_model.get(REFERENCE_MODEL, {}).values()]

    output = []
    for model in sorted(grouped):
        seeds = per_model[model]
        clean = [s["clean"] for s in seeds.values()]
        abs_drop = [s["abs_drop"] for s in seeds.values()]
        rel_drop = [s["rel_drop"] for s in seeds.values()]
        clean_lo, clean_hi = _bootstrap_ci(clean, seed=1)
        drop_lo, drop_hi = _bootstrap_ci(abs_drop, seed=2)
        pval = float("nan") if model == REFERENCE_MODEL else _bootstrap_pvalue(ref_drops, abs_drop, seed=3)
        output.append({
            "model": model,
            "role": "mechanism_ablation" if model in MECHANISM_MODELS else "external_baseline",
            "n_seeds": len(seeds),
            "clean_accuracy": _mean(clean),
            "clean_ci_lo": clean_lo, "clean_ci_hi": clean_hi,
            "max_stress_accuracy": _mean([s["stress"] for s in seeds.values()]),
            "abs_drop": _mean(abs_drop),
            "abs_drop_ci_lo": drop_lo, "abs_drop_ci_hi": drop_hi,
            "rel_drop": _mean([r for r in rel_drop if r == r]),
            "p_vs_sprif_full": pval,
        })
    return output

def summarize_curve(rows):
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    output = []
    for model in sorted(by_model):
        mrows = by_model[model]
        clean_by_seed = defaultdict(list)
        for r in mrows:
            if r["intervention_count"] == 0:
                clean_by_seed[r["seed"]].append(r["accuracy"])
        clean_seed = {s: _mean(v) for s, v in clean_by_seed.items()}
        clean_mean = _mean(list(clean_seed.values()))

        cl = list(clean_seed.values())
        lo, hi = _bootstrap_ci(cl, seed=1)
        output.append({"model": model, "intervention_fraction": 0.0,
                       "accuracy": clean_mean, "acc_ci_lo": lo, "acc_ci_hi": hi,
                       "abs_drop": 0.0, "n_seeds": len(cl)})
        fracs = sorted({r.get("intervention_fraction", 0.0) for r in mrows
                        if r["intervention_count"] != 0})
        max_k = max(r["intervention_count"] for r in mrows)
        for frac in fracs:
            by_seed = defaultdict(list)
            for r in mrows:
                if r["intervention_count"] == max_k and r.get("intervention_fraction", 0.0) == frac:
                    by_seed[r["seed"]].append(r["accuracy"])
            acc_seed = [_mean(v) for v in by_seed.values()]
            drop_seed = [clean_seed.get(s, clean_mean) - _mean(v) for s, v in by_seed.items()]
            a_lo, a_hi = _bootstrap_ci(acc_seed, seed=2)
            output.append({"model": model, "intervention_fraction": frac,
                           "accuracy": _mean(acc_seed), "acc_ci_lo": a_lo, "acc_ci_hi": a_hi,
                           "abs_drop": _mean(drop_seed), "n_seeds": len(acc_seed)})
    return output

