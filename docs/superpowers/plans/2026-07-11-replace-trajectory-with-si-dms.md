# Replace Trajectory Visualization with SI-DMS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace only the obsolete `trajectory_visualization` evidence package in `experiment-design-20260606` with the SI-DMS mechanism experiment while preserving every other paper experiment.

**Architecture:** Archive the old trajectory assets under `legacy/` with an explicit evidence warning. Add a self-contained SI-DMS experiment plan, result-fill templates, and figure manifest, then update the project-level evidence index and positioning claim so the paper points to SI-DMS rather than unsupported reset arrows.

**Tech Stack:** Markdown, JSON result templates, repository-relative artifact paths.

## Global Constraints

- Preserve main benchmark, ablation, impulse, reset, frequency-selectivity, noise-robustness, and sequence-noise artifacts.
- Never invent SI-DMS results; unknown values remain `TBD`.
- Mechanism ablations are only `sprif_full`, `sprif_merged`, and `sprif_lambda0`.
- LIF, ASRNN, and BRF are external baselines, not mechanism ablations.
- A trajectory with zero spikes cannot support pre/post reset-arrow evidence.

---

### Task 1: Archive obsolete trajectory visualization

**Files:**
- Move: `experiment-design-20260606/results/figures/trajectory_visualization/`
- Create: `experiment-design-20260606/legacy/trajectory_visualization/README.md`

- [x] Verify both source and destination resolve inside the repository.
- [x] Move the directory without modifying unrelated figure folders.
- [x] Add a warning that archived assets are provenance-only and may not support reset claims.

### Task 2: Add the SI-DMS evidence package

**Files:**
- Create: `experiment-design-20260606/si-dms-experiment-plan.md`
- Create: `experiment-design-20260606/results/si_dms/README.md`
- Create: `experiment-design-20260606/results/si_dms/result_template.json`
- Create: `experiment-design-20260606/results/figures/si_dms/MANIFEST.md`

- [x] Define task timing, intervention semantics, models, metrics, and integrity gates.
- [x] Map the two claims to paired comparisons.
- [x] Add only `TBD` result cells.

### Task 3: Update cross-document references

**Files:**
- Modify: `experiment-design-20260606/positioning-update.md`
- Modify: `experiment-design-20260606/results/README.md`
- Modify: `experiment-design-20260606/results/figures/MANIFEST.md`

- [x] Replace the old trajectory-visualization defense for C2 with SI-DMS.
- [x] Add SI-DMS to the results index and figure manifest.
- [x] Keep all unrelated experiments listed.

### Task 4: Verify evidence isolation

- [x] Search active files for `trajectory_visualization`, `trajectory_data_phi`, and reset-arrow claims.
- [x] Confirm any remaining occurrence is under `legacy/` or explicitly describes archival status.
- [x] Parse the JSON template and confirm every result field is `TBD`.
- [x] Confirm unrelated result artifacts remain present.
