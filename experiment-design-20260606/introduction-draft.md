# Introduction (Draft for SPRiF Paper — AAAI)

> **Writing plan**
> - Target venue: AAAI (AI/ML family)
> - Section role: create reviewer curiosity and make the contribution feel necessary
> - Paragraph roles: (1) task & value → (2) landscape & progress → (3) gap & root reason → (4) insight & method → (5) contributions & evidence preview
> - Core narrative: "the state that triggers a spike is the state that gets reset" → functional decomposition → projective reset → empirical validation
> - Idea scope: preserved (no method, claim, or experiment changes)

---

## 1 Introduction

Spiking neural networks (SNNs) process temporal information through discrete spike events, offering an event-driven computational paradigm with applications in speech recognition, physiological signal analysis, and event-based recognition. Unlike continuous-activation networks, the representational capacity of an SNN depends not only on its inter-layer connectivity but critically on the internal dynamics of individual neurons: how each neuron accumulates input history, decides when to fire, and reorganizes its state after each spike event. For temporal tasks where the relevant information is distributed across multiple timescales—from milliseconds in speech phonemes to hundreds of timesteps in long permuted sequences—the design of neuron-level dynamics directly constrains what temporal patterns the network can retain and exploit.

The leaky integrate-and-fire (LIF) neuron remains the dominant computational unit in SNNs due to its simplicity and compatibility with surrogate-gradient training. A rich lineage of extensions has substantially enriched the LIF dynamics: adaptive threshold models introduce slow auxiliary variables that modulate firing behavior \cite{adlif, dalif, biophysical_adaptation}; resonant and oscillatory neurons add damped subthreshold oscillations for frequency-selective processing \cite{brf, prf, drf}; multi-timescale designs equip neurons or synapses with heterogeneous decay rates \cite{bimodal, hetsyn, mtc, ltgate}; and recent reset mechanism innovations redesign the post-spike state update to reduce information loss \cite{inflor, arlif, rplif, psn}. Despite this diversity, all existing spiking neurons share a structural assumption: **the state variable that triggers a spike is the same state variable that gets reset**. In standard LIF, the membrane potential simultaneously accumulates past input, determines spike timing, and absorbs the reset—meaning every spike event partially erases the neuron's temporal memory. Even neurons that add oscillatory dynamics (e.g., resonate-and-fire variants \cite{brf, drf}) or adaptation variables (e.g., SE-adLIF \cite{adlif}) still reset the membrane potential that serves as the primary readout. Recent reset innovations—whether they soften the reset \cite{inflor}, make it adaptive \cite{arlif}, model it via threshold dynamics \cite{rplif}, or eliminate it entirely for parallelization \cite{psn}—all operate on, or reason about, a single state that simultaneously stores memory and generates spikes.

This structural coupling motivates the central question of this work: **must the spike-triggering state and the memory-bearing state be the same variable?** From a temporal modeling perspective, a spike is a discrete event signaling that the current state has reached a threshold; it necessitates a correction of the discharge-related state, but it does not logically require that all accumulated temporal memory be simultaneously weakened or erased. If the memory-carrying state could be structurally insulated from spike-triggered reset, the neuron would retain a richer record of input history across spike events—potentially enabling more faithful temporal processing without increasing network size or abandoning binary spike communication.

We introduce **SPRiF** (Spectral Projective Reset Integrate-and-Fire Neuron), a spiking neuron that addresses this question through *functional state decomposition*. SPRiF separates the neuron into two dynamically distinct components: a **slow spectral memory state** and a **fast discharge state**. The slow state integrates input through a constrained spectral structure—comprising a real exponential decay mode and a two-dimensional damped rotation mode—providing structured temporal filtering that extends beyond the single-exponential LIF memory trace. By design, the slow state is **never reset** by spike events, preserving continuous temporal memory across the spike train. The fast state receives projections from the slow state, produces the membrane readout, generates binary spikes via threshold crossing, and absorbs a novel *projective reset* along a learnable direction $[1, \lambda]^\top$ in its two-dimensional state space. SPRiF is fully compatible with standard SNN training: it uses backpropagation through time (BPTT) with surrogate gradients, requiring no specialized optimization. This design realizes what we term the *spike-never-resets-memory* principle: spike events trigger corrections only on the fast discharge manifold, while the slow spectral memory evolves continuously and undisturbed.

We emphasize that SPRiF's functional decomposition is qualitatively different from existing multi-timescale neuron designs. Prior multi-timescale models assign different decay rates or time constants to parallel processing channels—a strategy we term *rate decomposition*—but all channels serve the same dynamical role (memory accumulation and readout) and are subject to the same reset. SPRiF instead assigns different *dynamical functions* to its two states, enforcing the separation structurally through the spectral constraint on the slow state and the projective reset on the fast state, rather than through heterogeneous time constants alone.

We evaluate SPRiF across five temporal processing benchmarks spanning sequential image classification (S-MNIST, PS-MNIST), physiological signal analysis (QTDB ECG), speech recognition (Google Speech Commands), and event-based digit recognition (Spiking Heidelberg Digits). SPRiF achieves the best accuracy on three benchmarks (S-MNIST: 99.28\%, PS-MNIST: 95.86\%, QTDB: 88.43\%) and ranks second on the remaining two (GSC: 94.55\% vs. 94.84\% best; SHD: 91.52\% vs. 91.70\% best), matching or outperforming 11 established baselines—including GLIF, SE-adLIF, BRF, TC-LIF, and DGN—while using substantially fewer parameters (e.g., 0.067M on S-MNIST vs. 0.15M for the nearest baseline). Mechanism ablations across three core datasets confirm that each design element contributes materially: removing the damped rotation coupling degrades accuracy by up to 3.5 points, merging the slow and fast states causes a 2.4–5.0 point drop, and replacing the projective reset with a scalar reset reduces accuracy by 0.3–1.0 points. Analysis of learned spectral parameters reveals that SPRiF automatically adapts its internal timescales and oscillation frequencies to the temporal structure of each task.

**Contributions.** (1) We propose *functional state decomposition* for spiking neurons—separating temporal memory storage, spike readout, and post-spike reset into dynamically distinct state components, a principle that is orthogonal to existing rate-decomposition and single-state-reset approaches. (2) We introduce a *constrained spectral slow state* with real exponential decay and damped rotation modes, providing interpretable, structured temporal filtering beyond the single-exponential LIF trace. (3) We propose *projective fast-state reset*, a learnable directional reset mechanism that operates exclusively on the discharge state and leaves the memory state untouched. (4) We provide comprehensive empirical validation across five diverse temporal benchmarks with eleven baselines, three mechanism ablations, and learned parameter analysis. Code will be released upon acceptance.

---

## Paragraph Role Audit

| # | Role | Content |
|---|------|---------|
| P1 | Task & value | SNNs for temporal processing; neuron dynamics determine representational capacity |
| P2 | Landscape & progress | LIF + 4 extension families (adaptive, oscillatory, multi-timescale, reset); shared triple-duty assumption |
| P3 | Gap & root reason | State coupling: memory = readout = reset → every spike erases memory; no existing work questions this structurally |
| P4 | Insight & method | Functional decomposition → slow spectral memory (never reset) + fast discharge (projective reset); spike-never-resets-memory |
| P5 | Differentiation | Functional decomposition vs rate decomposition (explicit, preempts "just multi-timescale LIF" objection) |
| P6 | Evidence preview | 5 benchmarks, 11 baselines, key numbers, ablation deltas, parameter analysis |
| P7 | Contributions | 4 bullets: functional decomposition, spectral slow state, projective reset, empirical validation |

## Claim-Evidence Map

| Claim | Where stated | Evidence source | Status |
|-------|-------------|-----------------|--------|
| Functional decomposition is novel | P4, P5 | Literature search (no known prior) | supported |
| Slow state never reset | P4 | Method definition + trajectory analysis (C5, planned) | supported by design |
| Projective reset is novel | P4, P7 | Literature search (zero known prior) | supported |
| 3/5 benchmarks best; 2/5 competitive (#2) | P6 | Table 1 (filled) | supported |
| Each ablation contributes independently | P6 | Table 2 (filled) | supported |
| Parameters adapt to task structure | P6 | Parameter visualization (C6, cross-task done) | supported |

## Checklist Status

```text
Mode: Standard (full section draft)
Target venue: AAAI (AI/ML family)
Idea scope: PRESERVED — no method, claim, or experiment changes made

1. [✓] Target venue explicit: AAAI, AI/ML family
2. [✓] Available materials: Related Work draft, positioning-update.md, experiment-plan.md,
         Table 1 results (filled), Table 2 ablation results (filled), literature search (Round 1 + 2)
3. [✓] Global story defined: LIF triple-duty → structural coupling → functional decomposition →
         spectral memory + projective reset → empirical validation across 5 benchmarks
4. [✓] Section role: create curiosity; make contribution feel necessary; prepare reader for Method
5. [✓] Major claims mapped:
         C1 (functional decomposition) → P4, P5, P7
         C2 (spike-never-resets-memory) → P3, P4
         C3 (spectral constraint) → P4, P7
         C4 (projective reset) → P4, P7
6. [✓] Venue-fit risks checked: AAAI expects strong baselines (11 cited), clear novelty, honest positioning
7. [✓] Idea scope: preserved, no changes
8. [✓] Sibling modules: ccf-literature-search completed (Round 2); ccf-writing-skills active
9. [N/A] Score-lifting loop: section-level draft, not whole-paper
10. [✓] Remaining risks labeled below
```

## Score-Risk Diagnosis

| # | Question | Risk Level | Mitigation |
|---|----------|------------|------------|
| R1 | "Isn't this just adding more state variables?" | HIGH | P5 explicitly distinguishes functional decomposition from rate decomposition and from unconstrained state expansion |
| R2 | "BRF/D-RF also have oscillation — what's new?" | MEDIUM | P2 acknowledges oscillatory lineage; P4 clarifies oscillation lives in never-reset slow state, not membrane |
| R3 | "Local Timescale Gates / Bimodal also do slow+fast" | HIGH | P5 directly addresses this with functional vs rate decomposition distinction |
| R4 | "5D state per neuron — is the cost justified?" | LOW-MEDIUM | P6 reports parameter counts showing SPRiF uses comparable or fewer params than baselines |
| R5 | "Only classification benchmarks — where is regression/generation?" | LOW | Accepted limitation; Introduction frames scope as temporal processing benchmarks |
| R6 | "The introduction doesn't mention any SSM-SNN work" | LOW | SiLIF + SpikingSSMs cited in Related Work §2.3; Introduction focuses on neuron-level landscape |
