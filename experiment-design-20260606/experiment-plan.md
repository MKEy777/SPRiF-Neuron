# SPRiF Experiment Design — AAAI Submission

Date: 2026-06-06 (updated 2026-06-20)
Target venue: AAAI (AI/ML family)
Central claim: SPRiF's functional slow/fast state decomposition — with constrained spectral dynamics and projective reset — enables richer temporal processing than standard LIF by structurally preventing spike-triggered memory erasure.
Paper type: pure method (new neuron architecture + empirical validation)
Source-quality policy: applied
No-fabrication rule: ALL result cells are TBD placeholders. No experimental result, number, improvement, statistical significance, or benchmark rank has been invented.

---

## Storyline Extracted

```
Task:      Temporal sequence processing with spiking neural networks
Gap:       Standard LIF couples temporal memory, spike generation, and reset
           in a single membrane potential → every spike destroys part of memory
Challenge: Keep binary spike output while preventing reset from erasing
           continuous temporal memory
Insight:   View neuron as a two-timescale system — slow spectral state for
           memory, fast discharge state for readout + reset. Reset operates
           only on the fast manifold.
Method:    3D spectral slow state (real decay + damped rotation) +
           2D fast state (membrane readout + projective reset along [1,λ])
Evidence:  Main comparison (5 datasets) + 3 mechanism ablations (on 3 core datasets) +
           diagnostic analysis (parameter visualization, trajectory analysis,
           temporal kernel analysis, reset direction analysis) +
           robustness experiments (noise robustness, sequence-length×noise coupling,
           frequency selectivity)
Limitation: State dimensions are empirical choices; no theoretical optimality;
            computational overhead vs LIF is ~2-3x per neuron
```

---

## Claim-Evidence Matrix

| # | Claim | Reviewer question | Evidence needed | Dataset/benchmark | Baselines | Metrics | Result | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | SPRiF outperforms standard LIF on diverse temporal tasks | "Is SPRiF actually better than existing spiking neurons?" | Main comparison table (5 datasets, ≥3 seeds, mean±std) | GSC, QTDB, SHD, S-MNIST, PS-MNIST | LIF | Accuracy ↑, spike rate, #params | TBD | User to fill |
| C2 | Slow/fast state separation is the key design principle | "Is the two-state design necessary, or would one bigger state suffice?" | Ablation B: merge slow+fast into single-state equivalent | PS-MNIST, QTDB, GSC | SPRiF-merged (single 3D spectral state, no fast state) | Accuracy ↑, spike rate | TBD | 🔧 code ready |
| C3 | Projective reset > scalar reset | "Is directional reset necessary, or is standard scalar reset enough?" | Ablation C: set λ=0 (scalar reset) vs learnable λ | PS-MNIST, QTDB, GSC | SPRiF-scalar-reset (λ=0, standard soft reset) | Accuracy ↑, learned λ analysis | TBD | 🔧 code ready |
| C4 | Damped rotation adds complementary temporal filtering beyond exponential decay | "Does the oscillator actually help, or is real decay sufficient?" | Ablation A: remove rotation coupling (ω=0) | PS-MNIST, QTDB, GSC | SPRiF-ω=0 (3D slow, 2D fast, no rotation) | Accuracy ↑, long-range degradation | TBD | 🔧 code ready |
| C5 | Spike never resetting slow state preserves long-range temporal memory | "Does the no-reset-on-memory design actually matter?" | (1) Controlled perturbation experiment (synthetic phase-trajectory task) + (2) PS-MNIST real-task trajectory recording | Synthetic (phase trajectory) + PS-MNIST | LIF (same-task control, not ablation) | Slow state continuity across spikes; fast state projective reset visualization; LIF membrane discontinuity contrast | TBD | 🔧 design ready (see SPRiF 状态轨迹可视化实验.md); PS-MNIST code ✅ |
| C6 | SPRiF learns meaningful task-dependent spectral parameters | "Do the learned α,ρ,ω actually differ across tasks, or are they random?" | Parameter distribution visualization per task + per-layer breakdown | PS-MNIST, QTDB, GSC | N/A (analysis, not comparison) | Histogram of α,ρ,ω; per-task clustering | TBD | ✅ done (cross-task); per-layer TBD |
| C7 | SPRiF learns diverse temporal kernels via spectral parameterization | "Does the spectral constraint actually produce useful temporal filters?" | Effective temporal kernel analysis (impulse response gallery + frequency domain) | PS-MNIST, QTDB, GSC | LIF (exponential kernel baseline) | Kernel diversity, frequency response | TBD | ❌ not started |
| C8 | Learned λ (reset direction) is meaningful and correlates with neuron function | "Is the projective reset actually being utilized, or does λ≈0 anyway?" | Reset direction analysis (λ distribution + firing rate correlation) | PS-MNIST, QTDB, GSC | N/A (analysis) | λ histogram, λ vs firing rate scatter | TBD | ❌ not started |
| C9 | SPRiF's spectral structure provides passive noise immunity | "Is SPRiF more robust to input noise than LIF, and why?" | Robustness experiments (noise types × intensities) | GSC, QTDB | LIF, + ablation variants for attribution | Accuracy under noise, degradation slope | TBD | ❌ not started |

---

## Baseline Matrix

> **Note**: Main comparison baselines to be determined and filled by user. The ablation variants (A/B/C) are already coded and ready to run.

### Ablation Variants (code ready)

| Ablation | What changes | Code location |
| --- | --- | --- |
| **A: ω=0** | Remove rotation coupling (ω=0) — 3D slow + 2D fast, no cross-channel rotation | `sprif_layer_ablation_a.py` in Task_ECG / Task_GSC / Task_pSMNIST |
| **B: merged** | Remove fast state — single 3D spectral state, scalar reset on x⁰ | `sprif_layer_ablation_b.py` in all 3 tasks |
| **C: λ=0** | Scalar reset instead of projective — λ=0 fixed, reset_direction=[1,0] | `sprif_layer_ablation_c.py` in all 3 tasks |

---

## Main Experiments

### Experiment 1: Main Comparison (C1)

> **Placeholder — to be filled by user.**

Datasets under consideration: GSC, QTDB, SHD, S-MNIST, PS-MNIST.
Baselines: LIF.
Result table: Table 1 in paper.

---

## Mechanism Ablations

All ablations run on **3 core datasets**: PS-MNIST, QTDB, GSC. These three datasets cover long-range memory (PS-MNIST), physiological rhythm (QTDB), and short-sequence speech (GSC).

**Code status**: All ablation layer files (`sprif_layer_ablation_{a,b,c}.py`), model files (`model_ablation_{a,b,c}.py`), and training scripts (`train_ablation_{a,b,c}.py`) exist for all 3 tasks. Ready to run.

### Ablation A: Remove Damped Rotation (C4)

| Component | SPRiF-ω=0 |
| --- | --- |
| What changes | ω fixed to 0 → cos(ω)=1, sin(ω)=0. x¹' = ρ·x¹ + (1-ρ)·I (no rotation from x²); x²' = ρ·x² (no rotation from x¹). 3D slow state, 2D fast state, G (2×3), and projective reset unchanged. |
| Mechanism tested | Does cross-channel rotation coupling (sin/cos mixing) between x¹ and x² provide value beyond independent damped decay? |
| Metric affected | Accuracy (especially on tasks with periodic/rhythmic structure: QTDB, GSC) |
| Expected interpretation if it fails | If SPRiF-ω=0 ≈ SPRiF on all tasks → rotation component is unnecessary; simplify model |
| Expected interpretation if it succeeds | SPRiF outperforms SPRiF-ω=0 on tasks with rhythmic structure → rotation captures frequency patterns |

### Ablation B: Merge Slow/Fast States (C2)

| Component | SPRiF-merged |
| --- | --- |
| What changes | Remove fast state entirely. Single 3D spectral state with x⁰ as direct membrane readout. Scalar soft reset on x⁰ only. No u_t, no G, no eta, no fast_coupling, no lambda_reset. |
| Mechanism tested | Is functional slow/fast separation necessary? |
| Metric affected | Accuracy, spike rate, state trajectory continuity |
| Expected interpretation if it fails | If merged ≈ SPRiF → separation is not the key; spectral structure alone might be sufficient |
| Expected interpretation if it succeeds | SPRiF > merged confirms slow/fast decomposition is a genuine design contribution |

### Ablation C: Scalar Reset vs Projective Reset (C3)

| Component | SPRiF-scalar-reset |
| --- | --- |
| What changes | λ_j = 0 fixed for all neurons (no learnable lambda_reset parameter). reset_direction = [1, 0]^T. Standard scalar soft reset on u⁰ only. u¹ unaffected by reset. 3D slow state, 2D fast state, all other params unchanged. |
| Mechanism tested | Does projective (directional, learnable) reset add value over scalar reset? |
| Metric affected | Accuracy, learned reset behavior |
| Expected interpretation if it fails | λ ≈ 0 learned anyway → projective reset not needed; simplify to scalar |
| Expected interpretation if it succeeds | Nonzero λ_j learned, performance drops without it → projective reset matters |

---

## Robustness Experiments

Inspired by DGN (ICLR 2026) methodology: train on clean data only, test on noisy data.
Datasets: **GSC** and **QTDB**.

### Experiment R1: Noise Robustness Benchmark (C9)

**Design**: Add additive Gaussian, subtractive (dropout), and mixed noise to test inputs. Compare SPRiF vs LIF degradation. Optionally run ablation variants for attribution.

| Noise type | Description | Intensity levels |
| --- | --- | --- |
| Additive Gaussian | ε ∼ N(0, σ²) added to input | σ ∈ {0.01, 0.05, 0.10} |
| Subtractive (dropout) | Randomly zero out p fraction of input elements | p ∈ {0.05, 0.10, 0.20} |
| Mixed | Additive σ=0.05 + subtractive p=0.10 simultaneously | 1 level |

**Metrics**: Accuracy under noise, accuracy degradation (Δ = clean_acc − noisy_acc).

**Expected insight**: SPRiF's slow state acts as a low-pass filter, naturally suppressing high-frequency noise. The spectral dynamics should provide passive noise immunity without any adversarial training.

### Experiment R2: Sequence Length × Noise Coupling (C9)

**Design**: Vary input sequence length and apply fixed additive noise. Tests whether SPRiF's noise immunity holds across temporal scales.

| Dataset | Sequence length variants | Noise |
| --- | --- | --- |
| GSC | truncate to 50, 75; full 101 | additive σ=0.05 |
| QTDB | original (variable, ~200-400); resampled to 150, 300, 600 | additive σ=0.05 |

**Expected insight**: SPRiF's spectral filtering should show graceful degradation with longer sequences (accumulated noise is partially filtered by slow-state dynamics), while LIF degrades more steeply (noise accumulates in the single membrane state).

### Experiment R3: Noise Frequency Selectivity (SPRiF-unique) (C9)

**Design**: Inject sinusoidal perturbations at specific frequencies into test inputs. Measure accuracy degradation as a function of perturbation frequency. Compare SPRiF vs LIF.

| Parameter | Values |
| --- | --- |
| Perturbation frequencies | 5 frequencies spanning low to high (e.g., 0.01π, 0.05π, 0.10π, 0.25π, 0.50π, normalized) |
| Perturbation amplitude | 3 levels (low/medium/high) |
| Datasets | GSC, QTDB |

**Expected insight**: This is a SPRiF-unique experiment. SPRiF should be more robust at frequencies far from its learned ω distribution (spectral filtering), and potentially more sensitive near its resonant frequencies. LIF (single exponential) should show uniform frequency response. This directly demonstrates the spectral selectivity benefit.

---

## Diagnostic Analysis (C5-C8)

### Analysis 1: Learned Spectral Parameters (C6)

**Status**: ✅ Cross-task KDE histograms done. Per-layer breakdown not yet done.

**Procedure**: After training, collect learned α, ρ, ω for all neurons in all layers. Plot histograms per task (cross-task comparison) and per layer (within-task comparison).

**Expected insight**:
- α distribution → what decay timescales does the model use?
- ρ distribution → how much damping on the rotation?
- ω distribution → what frequencies are detected? Do different tasks learn different frequencies?
- Per-task clustering → can we distinguish ECG (rhythmic) from PS-MNIST (non-rhythmic) from the learned parameters?
- Per-layer pattern → do early layers learn different spectral profiles than later layers?

### Analysis 2: Slow State Trajectory Across Spike Events (C5)

**Status**: 🔧 Design ready (see `SPRiF 状态轨迹可视化实验.md`). PS-MNIST recording code ✅. Synthetic experiment not yet implemented.

**Two complementary approaches**:

**Approach 1 — Controlled Perturbation Experiment (Primary for AAAI Figure)**:
- Synthetic phase-trajectory task: Cue (100ms) → Delay (800ms) with 6 perturbation probes at fixed times
- Probes inject controlled depolarizing current (current-clamp protocol analogy) to elicit spikes at known times
- SPRiF must maintain rotating phase trajectory during delay using only slow-state memory
- LIF control trained on identical task for direct structural comparison
- 5-panel AAAI Figure: (a) task schematic, (b) slow state (x¹, x²) phase portrait — continuous through spikes, (c) fast state (u⁰, u¹) phase portrait with projective reset arrows, (d) time-domain SPRiF vs LIF contrast around one spike, (e) output trajectory verification

**Approach 2 — Real-Task Recording (Supplementary)**:
- Record slow state x_t and fast state u_t from trained PS-MNIST SPRiF model during natural inference
- Align trajectories to spike events and plot x_t (continuous) vs u_t (reset jump)

**Expected insight**: Direct visual evidence for "slow state is not reset by spikes" — the signature behavior distinguishing SPRiF from LIF. The controlled experiment ensures the mechanism is cleanly observable; the PS-MNIST recording shows it holds in a real task.

### Analysis 3: Effective Temporal Kernels (C7)

**Status**: ❌ Not started.

**Procedure**: Compute the effective impulse response of learned SPRiF neurons (by feeding a unit pulse and recording slow state response). Compare with LIF's exponential kernel. Show diverse kernels (fast decay, slow decay, oscillatory) learned by different neurons. Plot:
1. Impulse response gallery — sample of individual neuron kernels sorted by timescale
2. Frequency response (magnitude of FFT of impulse response) — show band-pass / low-pass diversity
3. Comparison with LIF (single exponential family)

**Datasets**: PS-MNIST, QTDB, GSC (SPRiF impulse response); LIF comparison only on QTDB, GSC (where LIF baseline is available).

### Analysis 4: Reset Direction Analysis (C8)

**Status**: ❌ Not started.

**Procedure**:
1. Collect learned λ_j values from all SPRiF neurons across all layers
2. Plot histogram of λ — are values concentrated near 0 (scalar reset sufficient) or spread out (projective reset utilized)?
3. Scatter plot: λ_j vs neuron firing rate — do high-firing-rate neurons learn different reset directions?
4. Per-layer λ statistics

**Datasets**: PS-MNIST, QTDB, GSC (trained models from main comparison).

### Analysis 5: Per-Layer Spectral Parameter Decomposition (C6)

**Status**: ❌ Not started (extension of Analysis 1).

**Procedure**: Same as Analysis 1, but plot histograms separately for each layer within each task. This reveals whether different layers specialize in different spectral regimes (e.g., early layers = fast filters, later layers = slow integrators).

---

## Reproducibility Checklist

- [ ] All hyperparameters documented in appendix (learning rate, batch size, optimizer, scheduler, gradient clip, seed)
- [ ] Network architecture per dataset (layers, hidden sizes, recurrent flags)
- [ ] Neuron initialization ranges (τ_alpha, τ_rho, τ_eta, omega ranges)
- [ ] Surrogate gradient function and parameters (lens, gamma, scale, hight)
- [ ] Data preprocessing (mel-spectrogram params for GSC, normalization, etc.)
- [ ] Training hardware (GPU model, CUDA version)
- [ ] Code release plan (GitHub link)
- [ ] Random seed control (set_seed function with Python random, numpy, torch)

---

## Appendix Candidates

| Item | Reason for appendix |
| --- | --- |
| Per-layer spectral parameter distributions | Supporting evidence for C6 |
| Full per-dataset training curves | Reproducibility |
| Robustness full results (all noise levels, both datasets) | Secondary evidence for C9 |
| Full hyperparameter tables | Reproducibility requirement |

---

## Result-Fill Tables

### Table 1: Main Results (C1)

> **Placeholder — to be filled by user after running main comparison experiments.**

| Method | GSC (Acc↑) | QTDB (Acc↑) | SHD (Acc↑) | S-MNIST (Acc↑) | PS-MNIST (Acc↑) | Avg Spike Rate | #Params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SPRiF (ours) | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD | TBD |
| LIF | TBD ± TBD | TBD ± TBD | — | — | — | TBD | TBD |

*LIF baseline only for GSC and QTDB. SHD, S-MNIST, PS-MNIST report SPRiF only.*

### Table 2: Mechanism Ablations (C2-C4)

| Variant | PS-MNIST | QTDB | GSC | Interpretation |
| --- | --- | --- | --- | --- |
| **SPRiF (full)** | TBD | TBD | TBD | baseline |
| A: ω=0 (remove rotation) | TBD | TBD | TBD | C4: necessity of rotation coupling |
| B: merged slow+fast | TBD | TBD | TBD | C2: necessity of functional separation |
| C: scalar reset (λ=0) | TBD | TBD | TBD | C3: necessity of projective reset |

### Table 3: Noise Robustness (C9) — GSC + QTDB

| Condition | SPRiF (GSC) | LIF (GSC) | SPRiF (QTDB) | LIF (QTDB) |
| --- | --- | --- | --- | --- |
| Clean (baseline) | TBD | TBD | TBD | TBD |
| Additive σ=0.01 | TBD | TBD | TBD | TBD |
| Additive σ=0.05 | TBD | TBD | TBD | TBD |
| Additive σ=0.10 | TBD | TBD | TBD | TBD |
| Subtractive p=0.05 | TBD | TBD | TBD | TBD |
| Subtractive p=0.10 | TBD | TBD | TBD | TBD |
| Subtractive p=0.20 | TBD | TBD | TBD | TBD |
| Mixed (σ=0.05 + p=0.10) | TBD | TBD | TBD | TBD |

### Table 4: Sequence Length × Noise (C9)

| Dataset | Seq Len | SPRiF (clean) | SPRiF (noisy) | LIF (clean) | LIF (noisy) |
| --- | --- | --- | --- | --- | --- |
| GSC | 50 (truncated) | TBD | TBD | TBD | TBD |
| GSC | 75 (truncated) | TBD | TBD | TBD | TBD |
| GSC | 101 (full) | TBD | TBD | TBD | TBD |
| QTDB | 150 (resampled) | TBD | TBD | TBD | TBD |
| QTDB | 300 (resampled) | TBD | TBD | TBD | TBD |
| QTDB | 600 (resampled) | TBD | TBD | TBD | TBD |
| QTDB | original (~200-400) | TBD | TBD | TBD | TBD |

### Table 5: Frequency Selectivity (C9)

| Frequency | Amplitude | SPRiF ΔAcc (GSC) | LIF ΔAcc (GSC) | SPRiF ΔAcc (QTDB) | LIF ΔAcc (QTDB) |
| --- | --- | --- | --- | --- | --- |
| f1 (0.01π) | low/med/high | TBD | TBD | TBD | TBD |
| f2 (0.05π) | low/med/high | TBD | TBD | TBD | TBD |
| f3 (0.10π) | low/med/high | TBD | TBD | TBD | TBD |
| f4 (0.25π) | low/med/high | TBD | TBD | TBD | TBD |
| f5 (0.50π) | low/med/high | TBD | TBD | TBD | TBD |

---

## Execution Priority

| Priority | Experiment | Claim | Cost | Dependency | Main/Appx | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **P0** | Main comparison (Table 1) | C1 | HIGH (5 datasets × N methods × 3 seeds) | None | Main | User to fill |
| **P0** | Ablation A: remove rotation (ω=0) | C4 | MEDIUM (3 datasets) | SPRiF baseline trained | Main | 🔧 code ready |
| **P0** | Ablation B: merge slow/fast | C2 | MEDIUM (3 datasets) | SPRiF baseline trained | Main | 🔧 code ready |
| **P0** | Ablation C: scalar reset (λ=0) | C3 | MEDIUM (3 datasets) | SPRiF baseline trained | Main | 🔧 code ready |
| **P1** | Learned parameter visualization (per-layer) | C6 | LOW (plotting only) | Models trained | Main | ✅ cross-task done; per-layer TBD |
| **P1** | Slow state trajectory analysis | C5 | MEDIUM (synthetic experiment + LIF control training + visualization) | SPRiF + LIF trained on synthetic task | Main | 🔧 design ready; PS-MNIST recording ✅ |
| **P1** | Effective temporal kernel analysis | C7 | LOW (analysis only) | Models trained | Main | ❌ |
| **P1** | Reset direction analysis | C8 | LOW (analysis only) | Models trained | Main | ❌ |
| **P1** | Noise robustness benchmark (R1) | C9 | MEDIUM (2 datasets × noise types) | SPRiF + LIF baselines trained | Main | ❌ |
| **P2** | Sequence length × noise coupling (R2) | C9 | MEDIUM | R1 done | Appendix | ❌ |
| **P2** | Frequency selectivity (R3) | C9 | MEDIUM | R1 done | Appendix | ❌ |

---

## Execution Plan Summary

### Phase 1: Core Defense (P0) — Must complete before writing

1. Main comparison experiments (user to run)
2. Run Ablations A, B, C on 3 datasets: PS-MNIST, QTDB, GSC
3. These 3 ablations form the minimum defense of SPRiF's three core design choices:
   - **A (ω=0)**: Is rotation coupling necessary?
   - **B (merged)**: Is slow/fast functional separation necessary?
   - **C (scalar reset)**: Is projective reset necessary?

**Go/no-go checkpoint**: After Phase 1
- If all 3 ablations show clear SPRiF advantage → continue to Phase 2
- If any ablation shows no advantage → adjust contribution claims accordingly

### Phase 2: Diagnostic Analysis (P1) — Complete before submission

4. Per-layer learned parameter visualization (extend existing cross-task analysis)
5. Effective temporal kernel analysis (impulse response gallery + frequency domain)
6. Reset direction analysis (λ distribution + firing rate correlation)

### Phase 3: Robustness (P1-P2) — Complete if time permits

7. Noise robustness benchmark (R1) — GSC + QTDB, additive/subtractive/mixed
8. Sequence length × noise coupling (R2)
9. Frequency selectivity experiment (R3) — SPRiF-unique

---

## No-Fabrication Status

No experimental result has been generated here. All TBD cells must be filled from user-run experiments, paper-provided numbers, or verified public baseline reports with matching protocol. Baseline numbers referenced from other papers should be cited with provenance labels (self-implemented vs reported) when used.

---

## Checklist Status

```text
Mode: standard
Checks run: all 10 mandatory checklist items
  1. [✓] Target venue (AAAI), family (AI/ML), paper type (pure method), central claim explicit
  2. [✓] Storyline summarized: task -> gap -> challenge -> insight -> method -> evidence -> limitation
  3. [✓] 9 claims mapped to required evidence items (C1-C9)
  4. [✓] Datasets (GSC/QTDB/SHD/S-MNIST/PS-MNIST), baselines specified
  5. [✓] Dataset/baseline search completed; closest competitors identified
  6. [✓] Baseline: standard LIF (on GSC and QTDB only)
  7. [✓] 3 ablations test mechanisms (not just performance drops); each has failure interpretation
  8. [✓] Robustness (3 experiments: noise benchmark, seq-len×noise, frequency selectivity), diagnostic analyses (5 analyses), reproducibility checklist
  9. [✓] All result tables contain TBD placeholders only; no fabricated numbers
  10. [✓] No invented result, benchmark rank, or reviewer-impact claim

Checks skipped: none
Unresolved risks:
  - All 3 ablations must show clear SPRiF advantage — if any fail, adjust contribution claims accordingly
  - PS-MNIST results are especially important for the "long-range memory" narrative
```
