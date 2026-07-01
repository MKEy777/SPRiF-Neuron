# Search Notes

## Safe Queries Used

1. `spiking neuron model beyond LIF new design 2024 2025 NeurIPS ICLR ICML`
2. `multi-timescale spiking neural network slow fast dynamics 2024 2025`
3. `reset mechanism spiking neural network innovation directional reset 2024 2025`
4. `oscillatory resonant spiking neuron resonate-and-fire 2024 2025`
5. `dendritic computation spiking neural network heterogeneity 2024 2025 Nature`
6. `long-range dependency spiking neural network temporal memory 2024 2025 ICLR NeurIPS`
7. `spike frequency adaptation neural computation memory preservation 2024 2025`
8. `spectral parameterization spiking neuron eigenvalue decomposition dynamics 2024 2025`
9. `gating mechanism spiking neuron memory protection spike-triggered 2024 2025`
10. `SPLR spiking neural network long-range Spike-Aware HiPPO NeurIPS 2024`
11. `"Revisiting Reset Mechanisms" spiking neural network arxiv 2504.17751`
12. `"operational manifolds" spiking neural networks Frontiers 2026`
13. `"stochastic gating" spiking neural network robustness AAAI 2024`
14. `"refractory period" "spike-triggered threshold" spiking neural network 2024 2025`
15. `"bimodal" neuronal time constant spiking neural network recurrent 2025`
16. `dendritic resonate-and-fire neuron NeurIPS 2025 long-range dependencies`
17. `conductance-based spiking neural network multi-timescale 2024 2025 gradient trainable`
18. `spiking neural network ensemble diverse neuron types 2024 2025 ICLR NeurIPS`
19. `"soft reset" "membrane potential redistribution" spiking neural network information loss 2024 2025`
20. `"neuromorphic computing paradigms enhance robustness" Nature Communications 2025 temporal processing SNN`

## Sources Checked

- arXiv (primary for preprints and recent papers)
- OpenReview (ICLR 2024/2025, NeurIPS 2024 submissions)
- NeurIPS 2024/2025 proceedings (papers.nips.cc, neurips.cc/virtual)
- ICML 2024/2025 proceedings (proceedings.mlr.press, icml.cc/virtual)
- AAAI 2024 proceedings (ojs.aaai.org)
- ACM MM 2025 (dl.acm.org)
- Nature Communications (nature.com)
- Neural Networks journal (sciencedirect.com)
- Frontiers in Neuroscience (frontiersin.org)
- IEEE NICE 2026 (arxiv.org)
- ECCV 2022 (ecva.net)
- Semantic Scholar (for citation counts and verification)
- DBLP (for author/venue verification)
- Google Scholar (for citation tracking)
- GitHub: TheBrainLab/Awesome-Spiking-Neural-Networks (curated SNN paper list)
- GitHub: AXYZdong/awesome-snn-conference-paper (conference paper list)
- GitHub: SpikingChen/SNN-Daily-Arxiv (daily arXiv tracker)

## Excluded Sources

- Policy-excluded: All MDPI venues (MDPI journals excluded by shared source-quality policy)
- SSM-related: SiLIF (arXiv 2506.06374), SpikingSSMs (AAAI 2025), FLAMES (arXiv 2504.01257), P-SpikeSSM (ICLR 2025) — all excluded per user request
- Low-signal: Papers with only snippet-level evidence and no accessible full text
- Hardware-only: Resonate-and-Fire Photonic-Electronic (arXiv 2510.14515) — hardware implementation focus, not relevant to SPRiF's algorithmic contribution
- ANN-SNN conversion: RMP-SNN — focused on ANN-to-SNN conversion, not native SNN design
- Continual learning focus: Context Gating in SNNs (ScienceDirect 2025), Astrocyte-gated multi-timescale plasticity (Frontiers 2026) — different research question

## Unknowns

- **Revisiting Reset Mechanisms (arXiv 2504.17751):** Full text not deeply accessed; specific theoretical claims about reset necessity need verification from PDF
- **Bimodal Neuronal Time Scales (Neural Networks 2026):** Abstract only accessed via ScienceDirect; full mechanism details unavailable
- **HetSyn (NeurIPS 2025):** Full experimental coverage not verified from the poster page
- **D-RF (NeurIPS 2025):** PDF accessed but compressed binary; key details extracted from arXiv abstract page
- **DGN (ICLR 2026):** Already in SPRiF baselines; no additional papers found from same group
- **MTC (IEEE NICE 2026):** Very recent; full experimental comparison details not verified

## Handoff Notes

### For writing:
- **Related Work section should add 3 new subsections:**
  1. "Reset Mechanism Innovation" — cite InfLoR-SNN, AR-LIF, RPLIF, Revisiting Reset; position SPRiF's projective reset as the first structural separation of reset target from memory store
  2. "Multi-Timescale Neuron Design" — cite Local Timescale Gates, Bimodal Time Scales, HetSyn, MTC; position SPRiF's functional decomposition (memory ≠ discharge ≠ reset) as distinct from rate decomposition (slow decay ≠ fast decay)
  3. Update "Resonate-and-Fire" subsection — add D-RF as the latest RF extension; position SPRiF's spectral functional decomposition vs D-RF's dendritic frequency decomposition
- **Elevator pitch refinement:** The distinction between "functional decomposition" (SPRiF) and "rate decomposition" (all multi-timescale work) should be emphasized more prominently

### For idea optimization:
- **New angle:** The "Revisiting Reset" paper questions whether reset is necessary at all — SPRiF could strengthen its motivation by arguing "reset IS necessary, but it should never touch memory" as a middle-ground position
- **Potential new experiment:** Compare SPRiF against Local Timescale Gates or Bimodal Time Scales on the same benchmarks to directly validate functional decomposition vs rate decomposition
- **MTC's Mackey-Glass benchmark:** Consider adding Mackey-Glass chaotic time-series regression to demonstrate SPRiF's spectral structure on continuous dynamics prediction (not just classification)

### For experiment design:
- **New baselines to consider:** D-RF (NeurIPS 2025) and HetSyn (NeurIPS 2025) are strong recent baselines from top venues
- **Robustness experiments:** Neuromorphic Computing Paradigms (Nature Comms 2025) and Stochastic Gating (AAAI 2024) suggest that temporal processing robustness is an active evaluation dimension — aligns with SPRiF's planned C9 robustness claims

### For review:
- **Missing related work risk:** If D-RF, Local Timescale Gates, Bimodal Time Scales, or Revisiting Reset are not cited, reviewers may flag incomplete literature coverage
- **Novelty risk:** Local Timescale Gates' dual time-constant design is the closest functional analog to SPRiF's slow/fast decomposition — must be clearly distinguished on "functional role" vs "decay rate"
