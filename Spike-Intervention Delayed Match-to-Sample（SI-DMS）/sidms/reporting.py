from collections import defaultdict


MECHANISM_MODELS = {"sprif_full", "sprif_merged", "sprif_lambda0"}


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows: grouped[row["model"]].append(row)
    output = []
    for model, values in sorted(grouped.items()):
        clean = [r["accuracy"] for r in values if r["intervention_count"] == 0]
        stressed = [r["accuracy"] for r in values if r["intervention_count"] == max(
            x["intervention_count"] for x in values)]
        output.append({"model": model,
                       "role": "mechanism_ablation" if model in MECHANISM_MODELS else "external_baseline",
                       "clean_accuracy": sum(clean) / len(clean),
                       "max_stress_accuracy": sum(stressed) / len(stressed),
                       "stress_drop": sum(clean) / len(clean) - sum(stressed) / len(stressed)})
    return output
