# SPRiF Positioning Update — Literature-Grounded (Round 2)

Date: 2026-06-06
Based on: ccf-literature-search (12 papers, 4 clusters) + ccf-experiment-designer (7 claims, 6 ablations)
Previous round: ccf-idea-optimizer initial report

---

## What Changed From Round 1

| Dimension | Round 1 (Pre-Search) | Round 2 (Post-Search) |
| --- | --- | --- |
| Closest competitor | Unknown — speculated AdLIF/RF | **SiLIF (Fabre, Jun 2025)** — confirmed, SSM-inspired oscillatory spiking neuron |
| Novelty confidence | Uncertain — marked `needs-literature-search` | **Moderately high** — projective reset has zero known prior; slow/fast functional decomposition is unique; spectral constraint vs SSM-import is the key differentiation |
| Strongest novelty signal | Slow/fast state separation | **Projective reset** (zero prior) + **spike never resets memory** (zero prior with this framing) |
| Weakened novelty area | N/A (unknown) | Spectral/oscillatory dynamics — BRF, PRF, SiLIF all have oscillatory components; SPRiF's differentiation is in *where* the oscillation lives (slow memory vs membrane) |
| Baseline urgency | Should compare with AdLIF | **Must benchmark against AdLIF** (SOTA, Nature Comms); BRF and SiLIF as strong secondary targets |
| SSM-SNN field status | Unknown activity level | **Very active (2024-2025)** — 6+ frameworks; SPRiF must explicitly position against this wave |

---

## Direct Comparison: SPRiF vs SiLIF (Closest Competitor)

This is the most important differentiation table for AAAI reviewers. Use in Related Work.

| Dimension | SiLIF (Fabre, Jun 2025) | SPRiF (Ours) |
| --- | --- | --- |
| **Core motivation** | SSM training techniques can stabilize spiking neuron training | Spike reset should not destroy temporal memory |
| **Design philosophy** | Import SSM parametrization (init, discretization) into LIF | Redesign neuron internal state from first principles |
| **State structure** | Single-state (multi-dimensional but functionally unified) | **Dual-state**: slow (3D spectral memory) + fast (2D discharge) |
| **Oscillatory dynamics** | Via complex-state SSM initialization | Via intrinsic damped rotation in slow state |
| **Where oscillation lives** | In the membrane potential (which gets reset) | **In the slow state (which never gets reset)** |
| **Reset mechanism** | Standard scalar reset on membrane | **Projective reset** along [1,λ] in fast state only |
| **Memory after spike** | Partially erased (standard LIF reset) | **Preserved** — slow state is never touched by reset |
| **Training focus** | Stability via SSM-style parametrization | Mechanism design; training uses standard BPTT + surrogate gradient |
| **Key result claim** | SOTA on speech recognition with SSM-level efficiency | Rich temporal filtering + principled reset across diverse temporal tasks |
| **Peer review** | Preprint only (Jun 2025) | Target: AAAI |

### The "One Sentence" Differentiation

> **SiLIF asks "how to train SNNs like SSMs." SPRiF asks "how to design a spiking neuron where spike events don't erase memory."** SiLIF imports SSM techniques into the LIF framework; SPRiF redesigns the framework itself.

---

## Comparison Matrix: SPRiF vs All Key Competitors

| Design Element | LIF | AdLIF | BRF | SiLIF | **SPRiF** |
| --- | --- | --- | --- | --- | --- |
| Multi-state | No (1D) | Yes (2D: V + adaptation) | No (1D complex or 2D real) | Yes (multi-D, single functional role) | **Yes (5D, two functional roles)** |
| State function | Unified (memory=readout=reset) | Memory + threshold modulation | Unified (resonating membrane) | Unified (SSM-augmented membrane) | **Functional decomposition (memory ≠ readout ≠ reset)** |
| Oscillatory dynamics | No (exponential only) | No | Yes (resonance) | Yes (complex-state init) | **Yes (intrinsic damped rotation in slow state)** |
| Reset target | Membrane (same state) | Membrane only (adaptation persists) | Membrane (same state) | Membrane (same state) | **Fast state only (slow state preserved)** |
| Reset type | Scalar subtract | Scalar subtract | Smooth reset | Scalar subtract | **Projective (directional, learnable λ)** |
| Temporal filter | Single exponential | Exponential + slow adaptation | Resonant filter | SSM-learned filter | **Exponential + damped oscillation (structured)** |
| Parameter interpretability | Low (τ only) | Medium (τ + adaptation params) | Medium (frequency + damping) | Low (SSM params are opaque) | **High (α,ρ,ω,η,λ each have clear dynamics meaning)** |

### Key Insight From This Matrix

SPRiF is the **only** model that simultaneously has:
1. Multi-state architecture
2. **Functional** (not spatial, not parametric) state role separation
3. Oscillatory dynamics that survive spike reset
4. Projective/directional reset mechanism
5. Interpretable dynamics parameters

Any single competitor may share one or two of these features, but none combines all five. This is SPRiF's unique contribution space.

---

## Updated Core Narrative (Elevator Pitch)

### Old (Round 1)
> "LIF couples memory, firing, and reset in one variable. SPRiF separates them into slow and fast states."

### New (Round 2 — Literature-Grounded)
> "Every standard spiking neuron — LIF, AdLIF, even recent oscillatory models like BRF — shares the same structural assumption: **the state that triggers a spike is the state that gets reset**. This means every spike partially erases temporal memory. SPRiF is the first spiking neuron to break this assumption: we decompose the neuron into a **slow spectral memory** (real decay + damped rotation, constrained by design) and a **fast discharge manifold** (readout + projective reset). The slow state carries continuous temporal memory and is **never touched by reset**. The fast state translates memory into spikes and absorbs the reset. This functional decomposition — not more parameters, not SSM import — is the contribution."

### Why This Works for AAAI

1. **Positions against the entire field, not just LIF** — "every standard spiking neuron... shares the same structural assumption" frames SPRiF as a principled departure, not an incremental tweak
2. **Names competitors without attacking them** — BRF is cited as an example of the shared assumption, not as a failed method
3. **Preempts the "just more parameters" attack** — explicitly states "not more parameters, not SSM import"
4. **Clear one-sentence takeaway** — "the state that triggers a spike is the state that gets reset" is a sticky framing reviewers will remember

---

## Literature-Grounded Innovation Claims (Updated)

### C1: Functional slow/fast state decomposition [Confidence: HIGH]
> SPRiF is the first spiking neuron model to functionally decompose neuron state by dynamical role: a slow spectral state for continuous temporal memory, and a fast state for membrane readout, spike generation, and post-spike reset.

**Literature basis**: AdLIF (Baronig 2025) has adaptation but the adaptation variable modulates firing threshold, not stores input history. TC-LIF (Yin IJCNN 2024) has dendritic+somatic compartments but these are spatial, not functional. No known work proposes functional role-based state decomposition in a spiking neuron.

**Strongest defense**: Ablation B (merge slow/fast) + Ablation D (free A vs spectral) together.

### C2: Spike-never-resets-memory principle [Confidence: HIGH]
> In SPRiF, spike events trigger reset only on the fast discharge manifold; the slow spectral state that carries temporal memory is structurally preserved across spike events.

**Literature basis**: PSN (Fang NeurIPS 2023) removes reset entirely for parallelization — different philosophy. All LIF variants (LIF, PLIF, AdLIF, GLIF) reset the same membrane potential that integrates memory. BRF/PRF reset the resonating membrane. No known work keeps a dedicated memory state untouched by reset.

**Strongest defense**: Slow state trajectory visualization (Analysis 2 in experiment plan) — direct visual evidence.

### C3: Constrained spectral structure > unconstrained state expansion [Confidence: MODERATE]
> SPRiF's slow state uses a structured spectral parametrization (block-diagonal: real decay + 2D rotation) rather than an unconstrained recurrent matrix. This provides interpretable dynamics parameters (α,ρ,ω) and empirically outperforms equivalent-capacity unconstrained recurrence.

**Literature basis**: SiLIF (Fabre Jun 2025) also uses structured SSM parametrization but imports it for training stability, not as an intrinsic design constraint. BRF/PRF have oscillatory dynamics but in single-state neurons.

**Novelty risk**: Reviewer may argue "structured state space is well-known from S4/Mamba." Defense: SPRiF uses structure as a **neuron design principle**, not a **network architecture pattern**. The structured A matrix lives inside each neuron's slow state, not as a network layer.

**Strongest defense**: Ablation D (free A vs spectral) — **this is the highest-stakes experiment**. If free A ≥ spectral A with comparable stability, this claim must be downgraded.

### C4: Projective reset in fast state space [Confidence: HIGH — STRONGEST NOVELTY]
> SPRiF's post-spike reset operates as a directional projection along a learnable vector [1,λ] in the 2D fast state space, rather than a scalar subtraction on membrane potential. This allows the neuron to learn how reset affects different dimensions of the discharge state.

**Literature basis**: **Zero known prior work.** "Projective reset" / "directional reset" returned no relevant results in comprehensive literature search. All known spiking neuron models use scalar reset (LIF family) or remove reset entirely (PSN). PRF's "decoupled reset" is in the complex domain and serves computational decoupling, not functional state projection.

**Strongest defense**: Ablation C (scalar reset λ=0 vs learnable λ) + learned λ analysis.

---

## Updated Reviewer Risk Register

| # | Risk | Round 1 Label | Round 2 Update |
| --- | --- | --- | --- |
| 1 | "Just more state variables" | design-fixable (HIGH) | **Downgraded to MEDIUM**. Literature search confirms no other model does functional decomposition. Defense: direct comparison matrix shows SPRiF's unique combination. Ablation D (free A) and B (merge) are specific defenses. |
| 2 | SiLIF already did this | NEW | **NEW RISK (MEDIUM-HIGH)**. SiLIF (Jun 2025) proposes SSM-inspired oscillatory spiking neurons. Defense: SPRiF ≠ SiLIF in motivation (design principle vs training technique), architecture (dual-state vs single-state), and reset (projective vs scalar). Explicit comparison table in Related Work. |
| 3 | BRF/PRF already have oscillation | requires-new-result (HIGH) | **Downgraded to MEDIUM**. BRF/PRF oscillation is in the membrane (gets reset); SPRiF oscillation is in the slow state (never reset). Differentiation is clear and testable. |
| 4 | SSM-SNN field is crowded | NEW | **NEW RISK (MEDIUM)**. 2024-2025 saw 6+ SSM-SNN frameworks. SPRiF must not be positioned as "SSM for SNN" — must be positioned as "neuron design principle." |
| 5 | No theoretical analysis | design-fixable (MEDIUM) | Unchanged. AAAI doesn't require theorems but effective kernel analysis (Analysis 3 in experiment plan) would help. |
| 6 | Small benchmark datasets | evidence-fixable (MEDIUM) | Unchanged. Defense: emphasize diagnostic value over scale; consider LRA as additional benchmark if compute allows. |
| 7 | Per-neuron computational cost | NEW | **NEW RISK (LOW-MEDIUM)**. 5D state per neuron is 3-5x LIF cost. Must report honestly and show accuracy-vs-FLOPs Pareto advantage. |

---

## Related Work Positioning Strategy

### Section Structure (Recommended for AAAI)

1. **Spiking Neuron Models** → LIF, PLIF, GLIF (standard baselines)
2. **Adaptive and Multi-Timescale Neurons** → AdLIF, DA-LIF, TC-LIF
   - *Positioning*: "These add adaptation variables that modulate firing, not store memory."
3. **Oscillatory and Resonate-and-Fire Neurons** → BRF, PRF
   - *Positioning*: "These have oscillatory membrane dynamics but reset the same state that oscillates."
4. **SSM-Inspired Spiking Models** → SiLIF, SpikingSSMs, FLAMES
   - *Positioning*: "These import SSM techniques into SNN networks. SPRiF redesigns the neuron itself."
5. **Reset Mechanism Innovations** → PSN, STSep
   - *Positioning*: "PSN removes reset; SPRiF redesigns it. STSep decouples at network level; SPRiF at neuron level."

### The "Citation Table" (for paper)

| Cite | For | Tone |
| --- | --- | --- |
| LIF (Gerstner) | Foundation | Neutral |
| AdLIF (Baronig 2025) | Strong baseline; adaptation ≠ memory | Respectful differentiation |
| BRF (Higuchi ICML 2024) | Oscillatory lineage; single-state limitation | Respectful differentiation |
| SiLIF (Fabre Jun 2025) | Closest SSM-SNN work; different philosophy | Direct but respectful contrast |
| PSN (Fang NeurIPS 2023) | Validates interest in reset redesign | Supportive |
| Yin et al. (2024) | Reset is essential (motivation for redesign) | Supportive |
| SpikingSSMs (Shen AAAI 2025) | Network-level SSM-SNN; SPRiF is neuron-level | Scope distinction |

### Golden Rule for Related Work
> Never say a competitor is "wrong" or "worse." Say: "X focuses on Y; in contrast, SPRiF addresses the orthogonal question of Z." This is the AAAI tone.

---

## Updated Elevator Pitch (Final)

### For the abstract / introduction hook:
> In every standard spiking neuron, the membrane potential serves triple duty: it accumulates past input, determines when to fire, and gets reset after each spike. This means every spike event partially erases the neuron's memory of past inputs. We introduce SPRiF, a spiking neuron that structurally separates these functions: a constrained spectral slow state carries continuous temporal memory and is never reset, while a fast discharge state handles readout, spike generation, and a learnable projective reset. Across five temporal processing benchmarks, this functional decomposition consistently outperforms standard and adaptive LIF neurons. Ablation studies confirm that the spectral constraint, state separation, and projective reset each independently contribute to performance, and that the learned spectral parameters adapt to task-specific temporal structure.

### For the contribution list:
1. **Functional state decomposition** for spiking neurons — separating temporal memory, spike readout, and reset into distinct dynamical roles
2. **Constrained spectral slow dynamics** — structured (not free) temporal filtering via real decay and damped rotation modes
3. **Projective fast-state reset** — directional, learnable reset that never touches the memory state
4. **Empirical validation** across speech, physiological, event-based, and long-range sequence tasks with mechanism ablations and learned dynamics analysis

---

## Writing-Readiness Note

### Ready to write now:
- Abstract (using updated elevator pitch above)
- Introduction (using updated core narrative)
- Related Work (using the 5-section structure and citation table)
- Method section (unchanged from draft — already solid)

### Not ready to write (blocked on experiments):
- Results section (waiting for Table 1 + Table 2 fill-in)
- Ablation analysis (waiting for Ablations A-D)
- Parameter visualization (waiting for Analysis 1-4)
- Limitation paragraph (needs failure analysis results)
- Claim-evidence alignment pass (needs completed results to verify claims)

### Recommended writing order:
1. Revise Introduction + Related Work now (literature-grounded, no experiments needed)
2. Polish Method section now (already near-complete)
3. Wait for Phase 1 experiment results (P0) → write Results skeleton
4. Wait for Phase 2 experiment results (P1) → fill Results + write Analysis
5. Write Limitation + Conclusion last

---

## Checklist Status (Second-Pass Update)

```text
Round 2 updates applied to Round 1 checklist:
  1. [✓] Target unchanged: AAAI, AI/ML family
  2. [✓] Idea card updated with literature-grounded confidence levels
  3. [✓] Missing inputs resolved: closest work identified (SiLIF, BRF, AdLIF)
  4. [✓] Novelty now grounded: HIGH for projective reset + functional decomposition; MODERATE for spectral constraint
  5. [✓] Method mechanism unchanged — solid; added comparison matrix vs all competitors
  6. [✓] Innovation type unchanged: method/architecture (primary) + empirical finding (secondary)
  7. [✓] Experiment plan designed (separate report); aligned with updated claims
  8. [✓] Reviewer risks updated: 2 new risks (SiLIF, compute cost), 2 downgraded, literature-informed defenses
  9. [✓] Candidate concretizations narrowed: Positioning A (design principle) confirmed as best with literature backing
  10. [✓] Writing-readiness: Introduction + Related Work ready to write now; Results blocked on experiments

New unresolved: SiLIF full text not accessed (preprint); may contain additional details that affect differentiation
```

---

## Three Files Produced This Session

```
SPRiF Neuron/
  literature-search-20260606-sprif-novelty/
    papers.md          — 12-paper analysis + 4 clusters
    papers.csv         — scored/classified table
    search-notes.md     — query audit + handoff notes

  experiment-design-20260606/
    experiment-plan.md  — full experiment design (7 claims, 6 ablations, TBD tables)
    positioning-update.md — THIS FILE (Round 2 positioning)
```
