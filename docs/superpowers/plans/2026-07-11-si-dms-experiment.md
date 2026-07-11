# SI-DMS Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, tested SI-DMS experiment package under `Spike-Intervention Delayed Match-to-Sample（SI-DMS）/`.

**Architecture:** A self-contained Python package generates synthetic DMS batches online, applies deterministic internal spike interventions, exposes six models through one step interface, and shares one training/evaluation pipeline. Results are serialized to JSON/CSV and converted into aggregate tables and figures.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, Matplotlib, PyYAML, pytest.

## Global Constraints

- Mechanism ablations are only `sprif_full`, `sprif_merged`, and `sprif_lambda0`.
- LIF, ASRNN, and BRF are external baselines, not SPRiF ablations.
- No direct slow-state readout.
- Controlled interventions occur after intrinsic flow and before thresholding.
- Existing source directories and scripts remain untouched.

---

### Task 1: Data and Intervention Contract

**Files:** Create `sidms/config.py`, `sidms/data.py`, and `tests/test_data.py`.

- [ ] Write failing tests for cue timing, labels, delay length, label-independent masks, selected-neuron count, and deterministic generation.
- [ ] Run focused pytest and verify failure due to missing modules.
- [ ] Implement typed configuration and online DMS batch generation.
- [ ] Re-run focused tests and verify pass.

### Task 2: Unified Neuron Models

**Files:** Create `sidms/surrogates.py`, `sidms/cells.py`, `sidms/models.py`, and `tests/test_cells.py`.

- [ ] Write failing tests for all model output shapes, forced-spike hits, SPRiF reset identity, slow-state insulation, lambda-zero reset, and merged-state reset.
- [ ] Verify tests fail before production code exists.
- [ ] Implement LIF, ASRNN, BRF, SPRiF full, lambda0, merged, and shared leaky readout.
- [ ] Verify all cell/model tests pass and task loss gives nonzero gradient to full SPRiF lambda.

### Task 3: Training and Evaluation

**Files:** Create `sidms/engine.py`, `train.py`, `evaluate.py`, and `tests/test_engine.py`.

- [ ] Write failing tests for one optimizer step, checkpoint round-trip, deterministic grid evaluation, and metric schema.
- [ ] Implement training, validation, checkpointing, JSONL history, grid evaluation, and trace export.
- [ ] Run tests and a CPU smoke training command.

### Task 4: Full Experiment Orchestration and Reporting

**Files:** Create `run_all.py`, `aggregate.py`, `plot_results.py`, `config/default.yaml`, and `tests/test_reporting.py`.

- [ ] Test command construction, aggregation formulas, stress-AUC, and figure creation using fixture results.
- [ ] Implement multi-model/multi-seed orchestration, CSV aggregation, heatmaps, curves, and mechanism summary plots.
- [ ] Verify reporting tests pass.

### Task 5: Documentation and Final Verification

**Files:** Create `README.md`, `requirements.txt`, `.gitignore`, and `SOURCE_NOTES.md`.

- [ ] Document provenance, equations, configuration, full and smoke commands, output schema, and interpretation limits.
- [ ] Run the complete pytest suite.
- [ ] Run one end-to-end smoke experiment for `sprif_full`, `sprif_merged`, and `sprif_lambda0`.
- [ ] Confirm all expected checkpoints, JSON, CSV, trace, and PNG files exist and contain no placeholder results.
