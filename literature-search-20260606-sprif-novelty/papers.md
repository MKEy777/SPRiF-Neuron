# Literature Search: SPRiF Neuron Novelty Grounding

Date: 2026-06-06
Search purpose: Novelty grounding for SPRiF (Spectral Projective Reset Integrate-and-Fire Neuron) — identifying closest prior work in adaptive/oscillatory/multi-compartment spiking neurons and structured state-space SNN models
Target venue/family: AAAI (AI/ML family)
Source-quality policy: applied (MDPI excluded, primary sources preferred)

## Summary

- Closest-work clusters: (1) Resonate-and-Fire / oscillatory spiking neurons, (2) Adaptive multi-timescale LIF variants, (3) Structured SSM-SNN hybrids, (4) Multi-compartment / state-decoupled SNNs
- Strongest baselines: AdLIF (Baronig 2025), BRF (Higuchi ICML 2024), SiLIF (Fabre Jun 2025), SpikingSSMs (Shen AAAI 2025)
- Benchmark/dataset candidates: Long Range Arena (LRA), SHD, GSC, PS-MNIST — already in SPRiF evaluation suite
- Novelty risks: SiLIF (Jun 2025) is the closest competitor — also brings structured SSM dynamics to spiking neurons; BRF/PRF have oscillatory dynamics but in single-state neurons without slow/fast decomposition
- Recommended next action: Explicitly differentiate SPRiF from SiLIF and BRF/PRF in Related Work; emphasize functional (not spatial) slow/fast decomposition and the "spike never resets memory" principle as the unique contribution

## Paper Table

| # | Title | Year | Venue/source | Link | Type | Insight | Completeness | Numeric evidence | Overall | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Structured State Space Model Dynamics and Parametrization for Spiking Neural Networks (SiLIF) | 2025 | arXiv (Jun 2025) | https://arxiv.org/abs/2506.06374 | pure method | 4 | 3 | 4 | Risk | Closest competitor: bridges SSM to SNN, proposes complex-state oscillatory neuron; SPRiF must differentiate |
| 2 | Balanced Resonate-and-Fire Neurons (BRF) | 2024 | ICML 2024 | https://arxiv.org/abs/2402.14603 | pure method | 4 | 4 | 4 | A | Resonating membrane dynamics with refractory + divergence boundary; single-state, no slow/fast split |
| 3 | PRF: Parallel Resonate and Fire Neuron for Long Sequence Learning in SNNs | 2024 | arXiv (Oct 2024) | https://arxiv.org/abs/2410.03530 | pure method | 4 | 3 | 4 | A | Complex-domain oscillatory neuron; decoupled reset for parallel training; computational not functional decoupling |
| 4 | Advancing Spatio-Temporal Processing in SNNs through Adaptation (AdLIF) | 2025 | Nature Communications | https://arxiv.org/abs/2408.07517 | pure method | 3 | 5 | 5 | A | Adaptation mechanism augments LIF; SOTA on event-based benchmarks; adaptation is not memory storage |
| 5 | SpikingSSMs: Learning Long Sequences with Sparse and Parallel Spiking State Space Models | 2025 | AAAI 2025 | https://arxiv.org/abs/2408.14909 | pure method | 4 | 4 | 4 | A | Dendritic hierarchy + SSM blocks; parallel training via surrogate network; 90% sparsity on LRA |
| 6 | FLAMES: A Hybrid Spiking-State Space Model for Adaptive Memory Retention | 2025 | arXiv (Apr 2025) | https://arxiv.org/abs/2504.01257 | pure method | 4 | 3 | 3 | B | Spike-Aware HiPPO for rate-adaptive memory; memory adaptation via spike rate, not structural separation |
| 7 | Understanding the Functional Roles of Modelling Components in Spiking Neural Networks | 2024 | Neuromorphic Computing and Engineering | https://arxiv.org/abs/2403.16674 | empirical finding | 3 | 4 | 4 | B | Ablation study of leakage/reset/recurrence; reset deemed essential; important background for motivation |
| 8 | Efficient Online Learning for Networks of Two-Compartment Spiking Neurons (TC-LIF) | 2024 | IJCNN 2024 | https://arxiv.org/abs/2402.15969 | pure method | 3 | 3 | 3 | B | Dendritic + somatic compartments; spatial (not functional) compartmentalization |
| 9 | DA-LIF: Dual Adaptive Leaky Integrate-and-Fire Model for Deep SNNs | 2025 | IEEE (2025) | https://ieeexplore.ieee.org/document/10888909 | pure method | 3 | 3 | 3 | B | Dual spatial + temporal adaptation; adaptation mechanism, not functional state separation |
| 10 | Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies (PSN) | 2023 | NeurIPS 2023 | https://openreview.net/forum?id=4q5ZYP0ynu | pure method | 4 | 4 | 4 | B | Eliminates reset entirely for parallelization; opposite approach to SPRiF (redesigns vs removes reset) |
| 11 | Unleashing Temporal Capacity of SNNs through Spatiotemporal Separation (STSep) | 2025 | arXiv (Dec 2025) | https://arxiv.org/abs/2512.05472 | pure method | 3 | 3 | 4 | C | Network-level spatial/temporal decoupling; not neuron-level; SOTA on video benchmarks |
| 12 | CRIMF: Channelwise Regional Integrate and Multiple Firing Neuron | 2024 | IEEE (2024) | https://ieeexplore.ieee.org/document/11159293 | pure method | 3 | 3 | 3 | C | Regional current + multiple firing; focuses on gradient and memory issues, not functional decomposition |

## Clusters

### Cluster 1: Oscillatory / Resonate-and-Fire Neurons (BRF, PRF)

- Representative papers: BRF (Higuchi ICML 2024), PRF (Huang Oct 2024)
- What this cluster already solves: Oscillatory membrane dynamics for frequency-sensitive temporal processing; stable training of RF neurons; parallelized RF training
- Remaining gap: Both BRF and PRF operate with single-state neurons — the membrane potential is still the only state variable. Neither separates temporal memory from spike generation/reset. The oscillatory dynamics are in the same state that gets reset.
- How it affects SPRiF: SPRiF's oscillatory component (damped rotation in slow state) is similar in spirit to RF's resonance, BUT: (a) SPRiF's oscillation is in the slow state that never gets reset, (b) SPRiF combines oscillation with exponential decay in a multi-modal slow state, (c) SPRiF adds projective reset in a separate fast state. Differentiation point: "RF neurons have resonating membrane; SPRiF has resonating memory + non-resonating discharge."

### Cluster 2: Adaptive LIF Variants (AdLIF, DA-LIF, TC-LIF)

- Representative papers: AdLIF (Baronig Nature Comms 2025), DA-LIF (Zhang IEEE 2025), TC-LIF (Yin IJCNN 2024)
- What this cluster already solves: Multi-timescale processing via adaptation currents, dual time constants, or multi-compartment structures
- Remaining gap: Adaptation variables modulate firing behavior (threshold, excitability), not temporal memory storage. TC-LIF compartments are spatial (dendrite/soma), not functional (memory/discharge/reset). None addresses the core SPRiF problem: "spike reset destroys temporal memory."
- How it affects SPRiF: These are the most likely baselines that reviewers will compare against. SPRiF must explicitly state: "AdLIF's adaptation variable is not a memory state — it modulates firing, not stores input history. SPRiF's slow state explicitly stores and filters temporal input history."

### Cluster 3: SSM-SNN Hybrids (SiLIF, SpikingSSMs, FLAMES, SPikE-SSM)

- Representative papers: SiLIF (Fabre Jun 2025), SpikingSSMs (Shen AAAI 2025), FLAMES (Chakraborty Apr 2025), SPikE-SSM (Zhong Oct 2024)
- What this cluster already solves: Integrating structured state-space model techniques (HiPPO, diagonal SSM, NPLR) into spiking neural networks for long-range sequence modeling
- Remaining gap: These methods import SSM architectures into SNN networks (e.g., SSM blocks as network layers). They do not redesign the neuron-level dynamics to structurally separate memory from reset. SPRiF operates at the neuron level, not the network level.
- How it affects SPRiF: SiLIF is the closest competitor — it also proposes complex-state spiking neurons with oscillatory regimes inspired by SSMs. SPRiF's differentiation: (a) SiLIF focuses on training stability via SSM parametrization; SPRiF focuses on functional state decomposition, (b) SiLIF's neuron is a single-state model; SPRiF has explicit slow/fast separation, (c) SPRiF's projective reset has no analog in SiLIF.

### Cluster 4: State Decoupling / Reset Innovation (PSN, STSep, CRIMF)

- Representative papers: PSN (Fang NeurIPS 2023), STSep (Dong Dec 2025), CRIMF (2024)
- What this cluster already solves: Removing reset for parallelization (PSN), network-level spatiotemporal separation (STSep), enhanced memory via regional currents (CRIMF)
- Remaining gap: PSN removes reset entirely (different philosophy). STSep operates at architecture level. CRIMF adds more state variables without functional role separation. None proposes: "keep reset, but apply it only to a dedicated discharge state, while preserving a separate memory state."
- How it affects SPRiF: These papers validate the general direction of "rethinking reset in SNNs" — which helps SPRiF's motivation. SPRiF's unique angle is redesigning reset (projective + fast-state-only) rather than removing or ignoring it.

## Benchmark And Dataset Candidates

| Name | Link | Task | Metrics | Baselines | Fit | Risks |
| --- | --- | --- | --- | --- | --- | --- |
| Long Range Arena (LRA) | https://github.com/google-research/long-range-arena | Long-sequence classification | Accuracy | S4, Transformer, SSM variants | High (would strengthen SPRiF's long-range claim) | Compute cost; may need architecture adaptation |
| SHD | https://zenodo.org/record/ | Spiking speech recognition | Accuracy | LIF, AdLIF, BRF | Already in SPRiF eval | — |
| GSC v2 | https://arxiv.org/abs/1804.03209 | Keyword spotting | Accuracy | Various SNN/SRNN | Already in SPRiF eval | — |
| PS-MNIST | — | Long-range pixel permutation | Accuracy | LIF, TC-LIF | Already in SPRiF eval | — |

## Citation And Positioning Cautions

- Claims that need direct citation: BRF (ICML 2024), SiLIF (Jun 2025), AdLIF (Nature Comms 2025), SpikingSSMs (AAAI 2025), PSN (NeurIPS 2023)
- Papers that may weaken novelty: SiLIF — reviewer may argue "structured oscillatory spiking neurons already proposed"; defense: SiLIF is a single-state model without functional slow/fast decomposition or projective reset
- Papers that only support background: Yin et al. (2024) — ablation study of LIF components; cite as motivation for why reset matters, not as prior art
- Important to cite but distinguish: BRF and PRF — acknowledge oscillatory neuron lineage but emphasize SPRiF's multi-state functional decomposition is orthogonal
