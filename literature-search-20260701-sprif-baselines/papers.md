# Literature Search: SPRiF Baseline-Inspired Directions (Round 2)

Date: 2026-07-01
Search purpose: Based on SPRiF's Table 1 baseline papers, discover recent (2022-2026) work in non-SSM directions that could provide new ideas, citation targets, or positioning context for SPRiF.
Target venue/family: AAAI (AI/ML family)
Source-quality policy: applied (MDPI excluded; preprints flagged; peer-reviewed preferred)

## Summary

- **Closest-work clusters:** 5 clusters identified — Reset Innovation, Multi-Timescale Neuron Design, Resonate-and-Fire Extensions, Gradient/Information Flow, Gating & Robustness
- **Strongest new baselines to consider:** D-RF (NeurIPS 2025), Local Timescale Gates (arXiv 2025), Bimodal Time Scales (Neural Networks 2026), MTC (NICE 2026)
- **Benchmark/dataset candidates:** Mackey-Glass (time-series regression), continual learning benchmarks
- **Novelty risks:** D-RF combines dendrites + RF (different from SPRiF's functional decomposition); Local Timescale Gates uses dual time-constants with gating (similar spirit to slow/fast but at different abstraction level); "Revisiting Reset" questions whether reset is necessary at all (philosophical contrast with SPRiF's "redesigned reset")
- **Recommended next action:** Use findings to strengthen Related Work positioning; consider adding D-RF or HetSyn as comparison baselines; leverage reset-innovation cluster to sharpen SPRiF's projective-reset novelty claim

## Paper Table

| # | Title | Year | Venue/source | Link | Type | Insight | Completeness | Numeric evidence | Overall | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Dendritic Resonate-and-Fire Neuron for Effective and Efficient Long Sequence Modeling | 2025 | NeurIPS 2025 | https://arxiv.org/abs/2509.17186 | pure method | 4 | 4 | 4 | A | Multi-dendrite RF neuron with frequency-selective filters + dynamic soma threshold; combines dendritic computation with oscillatory dynamics; important comparison target |
| 2 | Revisiting Reset Mechanisms in SNNs for Sequential Modeling | 2025 | arXiv (Apr 2025) | https://arxiv.org/abs/2504.17751 | pure method | 4 | 3 | 3 | A | Questions whether reset is strictly necessary for sparse spiking; proposes fixed-refractory-period architecture; provides theoretical contrast to SPRiF's "redesign reset" philosophy |
| 3 | Local Timescale Gates for Timescale-Robust Continual Spiking Neural Networks | 2025 | arXiv (Oct 2025) | https://arxiv.org/abs/2510.12843 | pure method | 4 | 3 | 4 | A | Dual time-constant dynamics (fast/slow parallel leaky integrators) with learnable blending gate per neuron; closest functional analog to SPRiF's slow/fast decomposition but without spectral structure or projective reset |
| 4 | AR-LIF: Adaptive Reset Leaky Integrate-and-Fire Neuron | 2025 | arXiv (Jul 2025) | https://arxiv.org/abs/2507.20746 | pure method | 3 | 3 | 4 | B | Adaptive reset connecting stimulus, response, and reset phase dynamically; argues hard reset causes information loss and soft reset treats all neurons uniformly; validates SPRiF's directional reset direction |
| 5 | Incorporating the Refractory Period into SNNs through Spike-Triggered Threshold Dynamics (RPLIF) | 2025 | ACM MM 2025 | https://arxiv.org/abs/2509.17769 | pure method | 3 | 4 | 4 | B | Models refractory period via dynamic threshold elevation post-spike instead of membrane reset; alternative to SPRiF's approach of "what happens after a spike" |
| 6 | Recurrent SNNs with Bimodal Neuronal Time Scales | 2026 | Neural Networks | https://www.sciencedirect.com/science/article/abs/pii/S0893608026002923 | pure method | 4 | 4 | 4 | A | First bimodal-timescale architecture for spiking RNNs; explicitly designs two distinct timescales; strong experimental validation; important comparison for SPRiF's slow/fast claim |
| 7 | HetSyn: Versatile Timescale Integration via Heterogeneous Synapses | 2025 | NeurIPS 2025 | https://neurips.cc/virtual/2025/poster/117410 | pure method | 4 | 4 | 4 | A | Moves timescale computation from neuron membrane to individual synapses; per-synapse decay rates; biologically observed learned parameters; strong multi-timescale baseline |
| 8 | Multi-Timescale Conductance Spiking Networks (MTC) | 2026 | IEEE NICE 2026 | https://arxiv.org/abs/2605.11835 | pure method | 4 | 3 | 4 | B | Fast/slow/ultra-slow conductance channels with direct BPTT; produces tonic/phasic/bursting; conductance-based (biological) approach vs SPRiF's spectral approach; outperforms AdLIF on Mackey-Glass |
| 9 | CLIF: Complementary Leaky Integrate-and-Fire Neuron | 2024 | ICML 2024 | https://proceedings.mlr.press/v235/huang24n.html | pure method | 3 | 4 | 4 | B | Adds complementary gradient pathways for temporal gradient flow; addresses gradient vanishing rather than temporal dynamics; relevant background for information-flow perspective |
| 10 | SPLR: Spiking Network for Learning Long-Range Relations | 2024 | NeurIPS 2024 | https://openreview.net/forum?id=2Ez4dhU3NG | pure method | 4 | 4 | 4 | B | Spike-Aware HiPPO + dendrite-inspired components for long-range dependencies; SSM-adjacent but uses HiPPO adaptation not SSM state space; cite for long-range context |
| 11 | Enhancing the Robustness of SNNs with Stochastic Gating Mechanisms | 2024 | AAAI 2024 | https://ojs.aaai.org/index.php/AAAI/article/view/27804 | pure method | 3 | 4 | 4 | B | Stochastic gating for layer-by-layer spike communication; bio-inspired gating for adversarial robustness; relevant to SPRiF's robustness claims (C9) |
| 12 | Neuromorphic Computing Paradigms Enhance Robustness through SNNs | 2025 | Nature Communications | https://www.nature.com/articles/s41467-025-65197-x | pure method | 3 | 5 | 5 | A | Exploits temporal processing for robustness; Nature Comms = high credibility; strong experimental coverage; must-cite for robustness/temporal processing positioning |
| 13 | Operational Manifolds in Spiking Neural Networks | 2026 | Frontiers in Neuroscience | https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1755119/full | empirical finding | 3 | 3 | 3 | C | Defines operational manifolds in hyperparameter space; compares reset vs carry policies; useful diagnostic framework but different research question |
| 14 | InfLoR-SNN: Reducing Information Loss for SNNs | 2022 | ECCV 2022 | https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710036.pdf | pure method | 3 | 4 | 4 | B | Soft Reset + Membrane Potential Redistribution; foundational work showing hard reset loses information; SPRiF's projective reset can be positioned as a structural solution to the same problem |

## Clusters

### Cluster 1: Reset Mechanism Innovation

- **Representative papers:** Revisiting Reset (#2), AR-LIF (#4), RPLIF (#5), InfLoR-SNN (#14)
- **What this cluster already solves:** Establishes that standard hard/soft reset loses information; proposes alternatives (adaptive reset, threshold dynamics, fixed refractory period, soft reset)
- **Remaining gap:** All existing reset innovations operate on a **single state** (membrane potential). None propose functional decomposition where the reset target is structurally separate from the memory store. AR-LIF adapts the reset value but still resets the membrane. RPLIF elevates threshold instead of resetting membrane. InfLoR-SNN redistributes membrane potential. None achieve "spike never resets memory."
- **How it affects SPRiF:** This cluster **validates and strengthens** SPRiF's projective reset novelty. SPRiF can be positioned as: "While prior work redesigns what happens to the membrane after a spike (adaptive value, threshold elevation, soft redistribution), SPRiF is the first to structurally prevent the spike from erasing temporal memory by decoupling the reset target from the memory store." The Revisiting Reset paper provides a useful philosophical contrast (questions necessity of reset vs SPRiF's redesign of reset).

### Cluster 2: Multi-Timescale Neuron Design

- **Representative papers:** Local Timescale Gates (#3), Bimodal Time Scales (#6), HetSyn (#7), MTC (#8)
- **What this cluster already solves:** Multiple timescales within a single neuron or synapse improve temporal processing; dual time-constants with gating; per-synapse decay rates; conductance-based multi-timescale
- **Remaining gap:** Multi-timescale in this cluster is **homogeneous in function** — the slow and fast components serve the same role (integrating input at different speeds). None propose **functional decomposition** where slow = memory (never reset) and fast = discharge (gets reset). Local Timescale Gates uses parallel integrators but both receive input and both participate in spike generation. HetSyn moves timescales to synapses (network-level effect). Bimodal Time Scales has two time constants but doesn't separate memory from reset. MTC uses conductance channels but all contribute to a single membrane potential.
- **How it affects SPRiF:** This cluster provides **strong comparison baselines** and validates the multi-timescale direction. SPRiF can differentiate: "While multi-timescale approaches assign different decay rates to parallel components, SPRiF assigns different **functional roles** — the slow state stores continuous memory (never reset), while the fast state handles membrane readout and spike generation (projective reset)." The key distinction is functional decomposition vs rate decomposition.

### Cluster 3: Resonate-and-Fire Extensions

- **Representative papers:** D-RF (#1)
- **What this cluster already solves:** RF neurons with dendritic structure; frequency-selective dendritic filters; dynamic soma threshold for preventing over-excitation; effective long sequence modeling
- **Remaining gap:** D-RF uses dendritic branches for frequency decomposition but maintains a **single-state** oscillatory neuron — the oscillation still lives in the same state that gets reset. No functional separation between oscillatory memory and discharge. The dynamic soma threshold is a firing-rate regulation mechanism, not a memory protection mechanism.
- **How it affects SPRiF:** D-RF is the **closest competitor** in the RF family. SPRiF differentiates on two axes: (1) oscillation lives in slow state (never reset) vs D-RF's oscillation in dendritic branches (still subject to reset effects through the soma), and (2) SPRiF's projective reset provides structural memory protection while D-RF's dynamic threshold provides rate-based protection. SPRiF can cite D-RF as a complementary approach: dendritic frequency decomposition vs spectral functional decomposition.

### Cluster 4: Gradient and Information Flow

- **Representative papers:** CLIF (#9), SPLR (#10)
- **What this cluster already solves:** Temporal gradient vanishing addressed via complementary pathways (CLIF); long-range dependencies via Spike-Aware HiPPO (SPLR)
- **Remaining gap:** CLIF addresses gradient flow but not temporal dynamics. SPLR addresses long-range dependencies at the network architecture level, not the neuron level.
- **How it affects SPRiF:** Background citations. CLIF supports the general theme of "information loss in SNNs." SPLR provides network-level long-range context. Neither competes with SPRiF's neuron-level innovation.

### Cluster 5: Gating, Robustness, and Diagnostic Frameworks

- **Representative papers:** Stochastic Gating (#11), Neuromorphic Robustness (#12), Operational Manifolds (#13)
- **What this cluster already solves:** Stochastic gating enhances adversarial robustness; temporal processing enhances general robustness; operational manifolds provide a diagnostic framework
- **Remaining gap:** Gating mechanisms operate at the network/spike-communication level, not the neuron-internal state level. Operational manifolds provide diagnostics but not a new neuron design.
- **How it affects SPRiF:** Supports SPRiF's robustness claims (C9). The Neuromorphic Computing Paradigms paper (Nature Comms) is a must-cite for temporal processing positioning. Operational manifolds could be used as a diagnostic tool in SPRiF's analysis.

## Benchmark And Dataset Candidates

| Name | Link | Task | Metrics | Baselines | Fit | Risks |
| --- | --- | --- | --- | --- | --- | --- |
| Mackey-Glass time series | Referenced in MTC paper (arXiv:2605.11835) | Chaotic time-series regression | RMSE / MSE | LIF, AdLIF, MTC | Medium — SPRiF's spectral structure may suit chaotic dynamics | Regression task differs from SPRiF's classification benchmarks |
| Continual learning benchmarks | Referenced in LT-Gate paper | Sequential task learning with interference | Forward transfer, backward transfer | LT-Gate, replay methods | Low — different research question from SPRiF's main claims |

## Citation And Positioning Cautions

### Claims that need direct citation:
- **Reset innovation landscape:** Must cite InfLoR-SNN (#14), AR-LIF (#4), RPLIF (#5), and Revisiting Reset (#2) when claiming projective reset novelty
- **Multi-timescale neuron design:** Must cite Local Timescale Gates (#3), Bimodal Time Scales (#6), HetSyn (#7) when claiming slow/fast functional decomposition novelty
- **Resonate-and-Fire family:** Must cite D-RF (#1) alongside BRF and PRF from the original search
- **Temporal processing robustness:** Must cite Neuromorphic Computing Paradigms (#12)

### Papers that may weaken novelty if not properly distinguished:
- **Local Timescale Gates (#3):** Dual time-constants with gating — if not distinguished on functional decomposition, could appear to subsume SPRiF's slow/fast idea. **Defense:** LT-Gate uses parallel integrators for rate diversity; SPRiF uses spectral decomposition for functional diversity (memory vs discharge vs reset).
- **D-RF (#1):** Combines dendrites + RF for long sequences — could appear to be a more comprehensive RF extension. **Defense:** D-RF's dendritic decomposition is frequency-based (what frequencies to extract); SPRiF's spectral decomposition is functional (what role each state plays).
- **Bimodal Time Scales (#6):** Explicit bimodal timescale design — could overlap with SPRiF's slow/fast. **Defense:** Bimodal = two decay rates for the same function; SPRiF slow/fast = two functional roles with spectral structure.

### Papers that only support background:
- CLIF (#9), SPLR (#10), Stochastic Gating (#11), Operational Manifolds (#13) — useful context but no direct novelty competition
