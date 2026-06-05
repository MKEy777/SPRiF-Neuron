# SPRiF Experiment Design — AAAI Submission

Date: 2026-06-06
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
Evidence:  5-dataset main comparison + 6 mechanism ablations +
           parameter visualization + efficiency + failure analysis
Limitation: State dimensions are empirical choices; no theoretical optimality;
            computational overhead vs LIF is ~2-3x per neuron
```

---

## Claim-Evidence Matrix

| # | Claim | Reviewer question | Evidence needed | Dataset/benchmark | Baselines | Metrics | Result | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | SPRiF outperforms standard LIF and strong SNN neuron models on diverse temporal tasks | "Is SPRiF actually better than existing spiking neurons?" | Main comparison table (5 datasets, ≥3 seeds, mean±std) | GSC, QTDB, SHD, S-MNIST, PS-MNIST | LIF, AdLIF, PLIF, BRF (if feasible) | Accuracy ↑, spike rate, #params | TBD | partial |
| C2 | Slow/fast state separation is the key design principle | "Is the two-state design necessary, or would one bigger state suffice?" | Ablation: merge slow+fast into single-state equivalent | All 5 datasets | SPRiF-merged (unified 5D state with scalar reset) | Accuracy ↑, spike rate | TBD | planned |
| C3 | Constrained spectral structure > unconstrained state expansion | "Does the spectral constraint matter, or is it just more parameters?" | Ablation: replace spectral A with free learnable 3x3 matrix | All 5 datasets | SPRiF-free-A (unconstrained A, matched #params) | Accuracy ↑, parameter efficiency | TBD | planned |
| C4 | Projective reset > scalar reset | "Is directional reset necessary, or is standard scalar reset enough?" | Ablation: set λ=0 (scalar reset) vs learnable λ | All 5 datasets | SPRiF-scalar-reset (λ=0, standard soft reset) | Accuracy ↑, learned λ analysis | TBD | planned |
| C5 | Damped rotation adds complementary temporal filtering beyond exponential decay | "Does the oscillator actually help, or is real decay sufficient?" | Ablation: remove rotation (1D slow state, real decay only) | All 5 datasets, especially PS-MNIST (long-range) | SPRiF-real-only (1D slow, no rotation) | Accuracy ↑, long-range degradation | TBD | planned |
| C6 | Spike never resetting slow state preserves long-range temporal memory | "Does the no-reset-on-memory design actually matter?" | Slow state trajectory visualization across spike events; long vs short sequence performance | PS-MNIST (long-range), S-MNIST (medium) | LIF (reset erases memory) vs SPRiF | Slow state continuity, accuracy vs seq length | TBD | planned |
| C7 | SPRiF learns meaningful task-dependent spectral parameters | "Do the learned α,ρ,ω actually differ across tasks, or are they random?" | Parameter distribution visualization per task | All 5 datasets | N/A (analysis, not comparison) | Histogram of α,ρ,ω; per-task clustering | TBD | planned |

---

## Baseline Matrix

| Baseline | Why included | Source | Implementation | Fairness constraints | Expected metric | Can run? |
| --- | --- | --- | --- | --- | --- | --- |
| **LIF** | Standard spiking neuron; primary comparison target | Gerstner & Kistler (2002); widely used | Self-implement (same codebase) | Same network architecture, matched hidden size; LIF has fewer params per neuron → also run param-matched LIF (wider) | Accuracy | yes |
| **AdLIF** | Strongest adaptive LIF variant; SOTA on event-based benchmarks | Baronig et al., Nature Comms 2025 | Self-implement or adapt from authors' code | Same network depth; AdLIF has adaptation parameter → param count close to SPRiF | Accuracy | yes (needs implementation) |
| **PLIF** | Learnable time constant LIF; widely used strong baseline | Fang et al., ICCV 2021 | Self-implement (sigmoid-gated τ) | Same architecture; PLIF adds 1 learnable param per neuron | Accuracy | yes |
| **BRF** | ICML 2024 oscillatory spiking neuron; closest RF-type competitor | Higuchi et al., ICML 2024 | Adapt from open-source code (GitHub) | Same architecture; BRF has refractory + divergence params; medium implementation effort | Accuracy | maybe (code available) |
| **SiLIF** | Closest SSM-inspired competitor (Jun 2025); direct novelty comparison | Fabre et al., arXiv Jun 2025 | Adapt from authors' GitHub | Same architecture; SiLIF uses complex-state SSM init; high implementation effort | Accuracy | maybe (code available, preprint) |
| **Simple MLP (non-spiking)** | Sanity check: does spiking itself matter for these tasks? | Standard | Self-implement | Same depth/width; ReLU activation; non-spiking | Accuracy | yes |
| **SPRiF-random-params** | Sanity check: are learned spectral parameters better than random? | N/A | Self-implement (fix α,ρ,ω to random values) | Same architecture; no spectral learning | Accuracy | yes |

### Baseline Categories

- **Must-run (P0)**: LIF, PLIF, AdLIF, SPRiF-random-params
- **Strongly recommended (P1)**: BRF, SiLIF, param-matched LIF
- **Nice-to-have (P2)**: Simple MLP

### If Missing

| Baseline | Reason if excluded |
| --- | --- |
| BRF | ICML code is available but RF-type neuron integration into SPRiF's network skeleton may be non-trivial; cite results from paper if cannot reproduce |
| SiLIF | Preprint (Jun 2025), very recent; code available but may be unstable; cite paper for discussion, not required as experimental baseline if implementation is infeasible |
| AdLIF | If self-implementation diverges from authors' results, report both self-implemented and paper-reported numbers with clear provenance labels |

---

## Main Experiments

### Experiment 1: Main Comparison (C1)

**Design**: Compare SPRiF against baselines on all 5 datasets. Fixed network skeleton (same depth, matched aggregate hidden dimension where feasible). 3+ random seeds. Report mean ± std.

**Datasets**:

| Dataset | Type | Seq Len | Classes | Input Dim | Rationale |
| --- | --- | --- | --- | --- | --- |
| GSC v2 | Keyword spotting (audio) | 101 | 12 | 120 (mel+delta) | Short-sequence speech; tests basic temporal processing |
| QTDB | ECG heartbeat (physiological) | variable | 2+ | ECG leads | Real-world physiological signal; tests medical time-series |
| SHD | Spiking speech recognition | ~1000 | 20 | 700 (input channels) | Event-based; tests spiking-domain processing |
| S-MNIST | Sequential MNIST (vision) | 784 | 10 | 1-28 (pixel/row) | Medium-range pixel-by-pixel; tests sequential memory |
| PS-MNIST | Permuted Sequential MNIST | 784 | 10 | 1-28 | Long-range shuffled; tests long-range dependency (key dataset for SPRiF's memory claim) |

**Metrics**:
- Accuracy (primary)
- Spike rate (spikes/neuron/timestep) — efficiency proxy
- Parameter count — fairness check
- Training time per epoch — practical cost

**Architecture**: Fixed per dataset — same number of layers, same aggregate hidden dimension. SPRiF uses recurrent mode where LIF has recurrence; non-recurrent where LIF is feedforward.

**Result Table**: Table 1 in paper

---

### Experiment 2: Parameter-Matched Comparison (C1 sub)

**Design**: Since SPRiF has more parameters per neuron than LIF (5D state vs 1D), run a param-matched comparison where LIF gets proportionally wider hidden layers to equalize total parameter count. This directly answers "is SPRiF's advantage just more parameters?"

**Result Table**: Appendix or inline with Table 1

---

## Ablations

### Ablation A: Remove Damped Rotation (C5)

| Component | SPRiF-real-only |
| --- | --- |
| What changes | Slow state reduced to 1D (real decay only): x_t = [x_t^0]; remove x_t^1, x_t^2 |
| Mechanism tested | Does damped rotation provide additional temporal filtering beyond exponential decay? |
| Metric affected | Accuracy (especially on tasks with periodic/rhythmic structure: QTDB, GSC) |
| Expected interpretation if it fails | If SPRiF-real-only ≈ SPRiF on all tasks → rotation component is unnecessary; simplify model |
| Expected interpretation if it succeeds | SPRiF outperforms SPRiF-real-only on tasks with rhythmic structure → rotation captures frequency patterns |

### Ablation B: Merge Slow/Fast States (C2)

| Component | SPRiF-merged |
| --- | --- |
| What changes | Remove fast state; slow state first dimension directly reads out membrane potential; scalar reset on x^0 after spike |
| Mechanism tested | Is functional slow/fast separation necessary? |
| Metric affected | Accuracy, spike rate, state trajectory continuity |
| Expected interpretation if it fails | If merged ≈ SPRiF → separation is not the key; spectral structure alone might be sufficient |
| Expected interpretation if it succeeds | SPRiF > merged confirms slow/fast decomposition is a genuine design contribution |

### Ablation C: Scalar Reset vs Projective Reset (C4)

| Component | SPRiF-scalar-reset |
| --- | --- |
| What changes | Set λ_j = 0 for all neurons (no learning); equivalent to standard soft reset on u^0 only |
| Mechanism tested | Does projective (directional, learnable) reset add value over scalar reset? |
| Metric affected | Accuracy, learned reset behavior |
| Expected interpretation if it fails | λ ≈ 0 learned anyway → projective reset not needed; simplify to scalar |
| Expected interpretation if it succeeds | Nonzero λ_j learned, performance drops without it → projective reset matters |

### Ablation D: Free State vs Spectral Constraint (C3)

| Component | SPRiF-free-A |
| --- | --- |
| What changes | Replace constrained A matrix (block-diagonal: real decay + 2D rotation) with fully learnable 3x3 matrix; same #params |
| Mechanism tested | Is the spectral structure (exponential + damped rotation) better than an unconstrained 3D linear recurrence? |
| Metric affected | Accuracy, training stability, parameter interpretability |
| Expected interpretation if it fails | Free A ≥ spectral A → spectral constraint is restrictive, not helpful |
| Expected interpretation if it succeeds | Spectral A ≥ free A with better stability/interpretability → structured constraint is genuinely useful |

### Ablation E: Dimensionality Sweep

| Component | SPRiF-dim-{variant} |
| --- | --- |
| What changes | Vary slow state dim (2D, 4D) and fast state dim (1D, 3D); default is (3,2) |
| Mechanism tested | Is (3,2) a good default? Can simpler configs match performance? |
| Metric affected | Accuracy vs #params Pareto curve |
| Expected interpretation | If 2D-slow + 1D-fast works → simplify paper claim to minimal configuration |

### Ablation F: Random Spectral Parameters (Sanity)

| Component | SPRiF-random-params |
| --- | --- |
| What changes | Fix α, ρ, ω to random (but stable) values from initialization range; no gradient to spectral params |
| Mechanism tested | Does LEARNING the spectral parameters matter, or is any stable spectral structure sufficient? |
| Metric affected | Accuracy |
| Expected interpretation | If random ≈ learned → spectral learning is not the mechanism; architecture alone matters |

---

## Robustness and Failure Analysis

### Robustness Tests

| Test | Procedure | Dataset | Reviewer question |
| --- | --- | --- | --- |
| Multi-seed variance | 5+ seeds, report mean±std, min/max | GSC, PS-MNIST | "Are results statistically stable?" |
| Sequence length sweep | Vary sequence length: 100, 200, 500, 1000, 2000 steps | PS-MNIST (truncated/extended) | "Does SPRiF degrade gracefully with longer sequences?" |
| Noise robustness | Add Gaussian noise to input (σ = 0.01, 0.05, 0.1) | GSC, SHD | "Is SPRiF more or less noise-robust than LIF?" |
| Surrogate gradient sensitivity | Test 2-3 surrogate gradient functions (Gaussian, triangular, rectangular) | GSC | "Are results tied to a specific surrogate gradient choice?" |
| Threshold sensitivity | Sweep θ ∈ {0.5, 1.0, 2.0, 5.0} | GSC | "How sensitive is SPRiF to threshold choice?" |

### Failure Analysis

| Analysis | Procedure | Expected reviewer concern |
| --- | --- | --- |
| When does LIF win? | Identify tasks/conditions where LIF ≥ SPRiF | "Is SPRiF always better, or only sometimes? When shouldn't I use it?" |
| Per-class breakdown | Accuracy per class on GSC (12 words) and SHD (20 phonemes) | "Does SPRiF improve uniformly, or only on certain classes?" |
| Training curve comparison | Plot loss/accuracy vs epoch for SPRiF vs LIF | "Does SPRiF converge faster/slower? Is it harder to train?" |
| Gradient norm analysis | Track gradient norms through slow/fast state pathways | "Does the slow/fast separation help gradient flow, or create vanishing gradients?" |

---

## Parameter Visualization and Diagnostic Analysis (C6, C7)

### Analysis 1: Learned Spectral Parameters

**Procedure**: After training, collect learned α, ρ, ω for all neurons in all layers. Plot histograms per task.

**Expected insight**:
- α distribution → what decay timescales does the model use?
- ρ distribution → how much damping on the rotation?
- ω distribution → what frequencies are detected? Do different tasks learn different frequencies?
- Per-task clustering → can we distinguish ECG (rhythmic) from S-MNIST (non-rhythmic) from the learned parameters?

### Analysis 2: Slow State Trajectory Across Spike Events

**Procedure**: For a test sequence, record slow state x_t and fast state u_t at each timestep. Align trajectories to spike events (t=0 at spike). Plot x_t (should be continuous) and u_t (should show reset jump) around spike.

**Expected insight**: Direct visual evidence for "slow state is not reset by spikes" — the signature behavior that distinguishes SPRiF from LIF.

### Analysis 3: Effective Temporal Kernels

**Procedure**: Compute the effective impulse response of learned SPRiF neurons (by feeding a unit pulse and recording slow state response). Compare with LIF's exponential kernel. Show diverse kernels (fast decay, slow decay, oscillatory) learned by different neurons.

**Expected insight**: SPRiF learns a diverse set of temporal filters, not just exponential decays.

### Analysis 4: Reset Direction Analysis

**Procedure**: Collect learned λ_j values. Plot histogram. Check if λ correlates with neuron firing rate.

**Expected insight**: Do neurons learn different reset strategies? Are high-firing-rate neurons associated with different λ?

---

## Efficiency Analysis

| Metric | Measurement | Comparison |
| --- | --- | --- |
| Parameter count | Total trainable params | SPRiF vs LIF vs AdLIF vs PLIF |
| FLOPs per timestep | Theoretical FLOPs for one neuron update | SPRiF vs LIF (expect ~3-5x for SPRiF due to 5D state) |
| Wall-clock training time | Seconds per epoch | SPRiF vs LIF (same hardware) |
| Wall-clock inference time | Milliseconds per sequence | SPRiF vs LIF |
| Memory usage | GPU memory during training (batch_size fixed) | SPRiF vs LIF |
| Spike sparsity | Average spike rate (spikes/neuron/timestep) | SPRiF vs LIF vs AdLIF |

**Important**: SPRiF will likely have higher per-neuron cost than LIF. This must be reported honestly. Mitigation:
1. Report accuracy vs FLOPs Pareto — if SPRiF achieves higher accuracy at lower total FLOPs (by using fewer neurons), that's a strong argument
2. Show that SPRiF's higher per-neuron cost is offset by better parameter efficiency

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
| Ablation E (dimensionality sweep) | Secondary ablation; not central to contribution |
| Parameter-matched LIF comparison table | Supporting evidence for fairness |
| All per-dataset training curves | Reproducibility; too many figures for main |
| Surrogate gradient sensitivity | Secondary robustness test |
| Full hyperparameter tables | Reproducibility requirement |
| Threshold sensitivity sweep | Secondary robustness |

---

## Result-Fill Tables

### Table 1: Main Results (C1)

| Method | GSC (Acc↑) | QTDB (Acc↑) | SHD (Acc↑) | S-MNIST (Acc↑) | PS-MNIST (Acc↑) | Avg Spike Rate | #Params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SPRiF (ours) | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD | TBD |
| LIF | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD | TBD |
| LIF (param-matched) | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD | TBD |
| AdLIF | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD | TBD |
| PLIF | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD | TBD |
| BRF* | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD ± TBD | TBD | TBD |

*BRF: if implementation feasible. Otherwise cite paper numbers with clear provenance label.

### Table 2: Mechanism Ablations (C2-C5)

| Variant | GSC | QTDB | SHD | S-MNIST | PS-MNIST | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| **SPRiF (full)** | TBD | TBD | TBD | TBD | TBD | baseline |
| w/o rotation (1D slow) | TBD | TBD | TBD | TBD | TBD | C5: necessity of oscillation |
| merged slow+fast | TBD | TBD | TBD | TBD | TBD | C2: necessity of separation |
| scalar reset (λ=0) | TBD | TBD | TBD | TBD | TBD | C4: necessity of projective reset |
| free A (unconstrained) | TBD | TBD | TBD | TBD | TBD | C3: necessity of spectral constraint |
| random spectral params | TBD | TBD | TBD | TBD | TBD | sanity: does learning matter? |

### Table 3: Efficiency

| Method | FLOPs/neuron/step | Train time (s/epoch) | Inference (ms/seq) | GPU Memory (GB) | Spike Rate |
| --- | --- | --- | --- | --- | --- |
| SPRiF | TBD | TBD | TBD | TBD | TBD |
| LIF | TBD | TBD | TBD | TBD | TBD |
| AdLIF | TBD | TBD | TBD | TBD | TBD |

### Table 4: Robustness (PS-MNIST)

| Condition | SPRiF | LIF | Gap |
| --- | --- | --- | --- |
| Baseline (T=784) | TBD | TBD | TBD |
| Seq len = 200 | TBD | TBD | TBD |
| Seq len = 500 | TBD | TBD | TBD |
| Seq len = 1500 | TBD | TBD | TBD |
| Noise σ=0.05 | TBD | TBD | TBD |
| Noise σ=0.10 | TBD | TBD | TBD |

---

## Execution Priority

| Priority | Experiment | Claim | Cost | Dependency | Main/Appx | Stop condition |
| --- | --- | --- | --- | --- | --- | --- |
| **P0** | Main comparison (Table 1) | C1 | HIGH (5 datasets x 7 methods x 3 seeds) | None | Main | — |
| **P0** | Ablation A: remove rotation | C5 | MEDIUM (5 datasets) | Table 1 done | Main | — |
| **P0** | Ablation B: merge slow/fast | C2 | MEDIUM | Table 1 done | Main | If merge ≈ SPRiF, pivot contribution to "spectral enhancement of LIF" |
| **P0** | Ablation C: scalar reset | C4 | MEDIUM | Table 1 done | Main | If scalar ≈ projective, simplify reset design |
| **P1** | Ablation D: free A vs spectral | C3 | MEDIUM | Table 1 done | Main | **Critical**: if free A ≥ spectral with similar stability, novelty is weakened |
| **P1** | Parameter-matched LIF | C1 | LOW (just wider LIF) | LIF baseline exists | Appendix | — |
| **P1** | Learned parameter visualization | C7 | LOW (plotting only) | Models trained | Main | — |
| **P1** | Slow state trajectory analysis | C6 | LOW (plotting only) | Models trained | Main | — |
| **P2** | Ablation E: dim sweep | — | HIGH (many configs) | Table 1 done | Appendix | — |
| **P2** | Ablation F: random params | — | LOW | Table 1 done | Appendix | If random ≈ learned, fundamental problem |
| **P2** | BRF baseline | C1 | MEDIUM (implementation) | None | Main/Appx | Skip if implementation too complex |
| **P2** | SiLIF baseline | C1 | HIGH (implementation + tuning) | None | Main/Appx | Skip; cite for discussion only |
| **P2** | Robustness suite (Table 4) | — | MEDIUM | Table 1 done | Appendix | — |
| **P2** | Efficiency analysis (Table 3) | — | LOW (profiling) | Models trained | Main/Appx | — |
| **P2** | Failure analysis | — | LOW (analysis) | Table 1 done | Main | — |
| **P3** | Surrogate gradient sensitivity | — | MEDIUM | Table 1 done | Appendix | — |
| **P3** | Threshold sensitivity | — | LOW | Table 1 done | Appendix | — |
| **P3** | Simple MLP baseline | — | LOW | None | Appendix | — |

---

## Execution Plan Summary

### Phase 1: Core Defense (P0) — Must complete before writing

1. Complete main comparison for all 5 datasets with LIF, AdLIF, PLIF (Table 1)
2. Run Ablations A, B, C (remove rotation, merge states, scalar reset)
3. These 3 ablations form the minimum defense of SPRiF's three core design choices

**Go/no-go checkpoint**: After Phase 1
- If all 3 ablations show clear SPRiF advantage → continue to Phase 2
- If any ablation shows no advantage → adjust contribution claims accordingly (see pivot suggestions in idea optimizer report)

### Phase 2: Novelty Defense (P1) — Complete before submission

4. Ablation D (free A vs spectral) — **critical for AAAI novelty**
5. Learned parameter visualization (α,ρ,ω distributions per task)
6. Slow state trajectory analysis (continuity across spikes)
7. Parameter-matched LIF comparison

**Go/no-go checkpoint**: After Phase 2
- If free A ≥ spectral → spectral constraint claim is weakened; reposition as "structured > unstructured for stability/interpretability"
- If learned parameters are not task-distinctive → drop visualization from main claim; move to appendix

### Phase 3: Polish (P2-P3) — Complete if time permits

8. BRF baseline (if feasible)
9. Robustness suite
10. Efficiency analysis
11. Failure analysis
12. Dimensionality sweep + other secondary ablations

---

## No-Fabrication Status

No experimental result has been generated here. All TBD cells must be filled from user-run experiments, paper-provided numbers, or verified public baseline reports with matching protocol. The AdLIF and BRF baseline numbers referenced in the literature search are from their respective papers and should be cited with provenance labels (self-implemented vs reported) when used.

---

## Checklist Status

```text
Mode: standard
Checks run: all 10 mandatory checklist items
  1. [✓] Target venue (AAAI), family (AI/ML), paper type (pure method), central claim explicit
  2. [✓] Storyline summarized: task -> gap -> challenge -> insight -> method -> evidence -> limitation
  3. [✓] 7 claims mapped to required evidence items (C1-C7)
  4. [✓] Datasets (GSC/QTDB/SHD/S-MNIST/PS-MNIST), baselines (7 named), metrics specified
  5. [✓] Dataset/baseline search completed via ccf-literature-search; SiLIF + BRF identified
  6. [✓] Baselines include closest prior work (AdLIF, BRF), simple sanity baselines (random params, MLP)
  7. [✓] 6 ablations test mechanisms (not just performance drops); each has failure interpretation
  8. [✓] Robustness (5 tests), failure analysis (4 analyses), efficiency (6 metrics), reproducibility checklist
  9. [✓] All result tables contain TBD placeholders only; no fabricated numbers
  10. [✓] No invented result, benchmark rank, or reviewer-impact claim

Checks skipped: none
Unresolved risks:
  - BRF/SiLIF baseline feasibility depends on code availability and implementation effort
  - Ablation D (free A vs spectral) is the single highest-stakes experiment — if free A wins, major contribution pivot needed
  - PS-MNIST results are especially important for the "long-range memory" narrative
```
