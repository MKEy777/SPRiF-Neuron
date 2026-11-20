import random
from collections import defaultdict

MECHANISM_MODELS = {"memory_reset5", "real_modes3"}
LEGACY_MODELS = {"sprif_full", "sprif_merged", "sprif_lambda0"}
REFERENCE_MODEL = "sprif_scalar"
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

def _legacy_summarize(rows, intervention_fraction=0.15):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    per_model = {m: _per_seed_metrics(v, intervention_fraction) for m, v in grouped.items()}
    # Legacy-only files retain their original reference and result identity.
    reference = REFERENCE_MODEL if REFERENCE_MODEL in per_model else "sprif_full"
    ref_drops = [s["abs_drop"] for s in per_model.get(reference, {}).values()]

    output = []
    for model in sorted(grouped):
        seeds = per_model[model]
        clean = [s["clean"] for s in seeds.values()]
        abs_drop = [s["abs_drop"] for s in seeds.values()]
        rel_drop = [s["rel_drop"] for s in seeds.values()]
        clean_lo, clean_hi = _bootstrap_ci(clean, seed=1)
        drop_lo, drop_hi = _bootstrap_ci(abs_drop, seed=2)
        pval = float("nan") if model == reference else _bootstrap_pvalue(ref_drops, abs_drop, seed=3)
        role = ("primary_model" if model == REFERENCE_MODEL else
                "legacy_model" if model in LEGACY_MODELS else
                "mechanism_ablation" if model in MECHANISM_MODELS else "external_baseline")
        output.append({
            "model": model,
            "role": role,
            "reference_model": reference,
            "n_seeds": len(seeds),
            "clean_accuracy": _mean(clean),
            "clean_ci_lo": clean_lo, "clean_ci_hi": clean_hi,
            "max_stress_accuracy": _mean([s["stress"] for s in seeds.values()]),
            "abs_drop": _mean(abs_drop),
            "abs_drop_ci_lo": drop_lo, "abs_drop_ci_hi": drop_hi,
            "rel_drop": _mean([r for r in rel_drop if r == r]),
            "p_vs_" + reference: pval,
        })
    return output

def _legacy_summarize_curve(rows):
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



def _modern_groups(rows):
    import math
    groups = defaultdict(list)
    seen = set()
    for r in rows:
        if r.get('training') not in ('Clean','Intervention'):
            raise ValueError('Every modern record must have training=Clean or Intervention')
        if r.get('smoke') is not False:
            raise ValueError('Formal summary requires explicit smoke=false; exclude smoke or unknown runs')
        value = r.get('accuracy')
        if not isinstance(value,(float,int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Accuracy must be finite and between 0 and 1')
        k=(r['training'],r['model'],r['seed'],r['delay_ms'],r['intervention_count'],r.get('intervention_fraction',0))
        if k in seen: raise ValueError(f'Duplicate evaluation: {k}')
        seen.add(k); groups[(r['training'],r['model'])].append(r)
    return groups


def summarize(rows, intervention_fraction=0.15):
    """TNNLS: per-condition 5-seed sample SD with seed-matched clean-minus-stress."""
    if not rows: raise ValueError('No evaluation records')
    if not any('training' in r for r in rows): return _legacy_summarize(rows,intervention_fraction)
    from statistics import mean, stdev
    import math
    output=[]; delays={200,400,800,1600,2500}; expected_seeds={0,1,2,42,1111}
    for (training,model), group in sorted(_modern_groups(rows).items()):
        seeds={r['seed'] for r in group}
        if seeds != expected_seeds: raise ValueError(f'{training}/{model}: expected five formal seeds')
        selected=[r for r in group if r['intervention_count']==0 or (r['intervention_count']==40 and math.isclose(r.get('intervention_fraction',0),intervention_fraction,abs_tol=1e-9))]
        lookup={}
        for r in selected:
            key=(r['seed'],r['delay_ms'],r['intervention_count'])
            if key in lookup: raise ValueError(f'Duplicate primary evaluation {training}/{model}/{key}')
            lookup[key]=100*r['accuracy']
        expected={(seed,d,k) for seed in seeds for d in delays for k in (0,40)}
        if set(lookup)!=expected: raise ValueError(f'{training}/{model}: missing or unexpected primary delay/K records')
        result={'training':training,'model':model,'n_seeds':len(seeds)}
        for label, ds in [('five_delay',sorted(delays)),('2500ms',[2500])]:
            clean=[mean(lookup[(seed,d,0)] for d in ds) for seed in sorted(seeds)]
            stress=[mean(lookup[(seed,d,40)] for d in ds) for seed in sorted(seeds)]
            delta=[a-b for a,b in zip(clean,stress)]
            for name,values in [('clean',clean),('stress',stress),('delta40',delta)]:
                result[f'{label}_{name}_mean']=mean(values)
                result[f'{label}_{name}_sd']=stdev(values)
        output.append(result)
    return output


def summarize_curve(rows):
    """TNNLS fraction curves are never mixed across training conditions."""
    if not rows: raise ValueError('No evaluation records')
    if not any('training' in r for r in rows): return _legacy_summarize_curve(rows)
    from statistics import mean,stdev
    output=[]
    for (training,model),group in sorted(_modern_groups(rows).items()):
        seeds={r['seed'] for r in group}; delays={200,400,800,1600,2500}
        if seeds != {0,1,2,42,1111}: raise ValueError(f'{training}/{model}: expected five formal seeds')
        fractions=sorted({r['intervention_fraction'] for r in group if r['intervention_count']==40})
        for fraction in [0.0]+fractions:
            k=0 if fraction==0 else 40
            selected=[r for r in group if r['intervention_count']==k and (k==0 or r['intervention_fraction']==fraction)]
            lookup={(r['seed'],r['delay_ms']):r['accuracy'] for r in selected}
            if len(lookup)!=len(selected) or set(lookup)!={(seed,d) for seed in seeds for d in delays}:
                raise ValueError(f'{training}/{model}: incomplete or duplicate fraction {fraction}')
            vals=[mean(lookup[(seed,d)] for d in delays) for seed in sorted(seeds)]
            output.append({'training':training,'model':model,'intervention_fraction':fraction,'n_seeds':len(vals),'accuracy_mean':mean(vals),'accuracy_sd':stdev(vals)})
    return output
