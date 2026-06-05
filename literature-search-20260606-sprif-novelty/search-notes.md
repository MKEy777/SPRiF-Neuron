# Search Notes

## Safe Queries Used

- "adaptive LIF neuron spiking neural network multi-timescale 2022 2023 2024 2025"
- "resonate and fire neuron spiking neural network oscillatory dynamics 2022 2023 2024"
- "structured state space spiking neuron slow fast dynamics 2023 2024 2025"
- "projective reset directional reset spiking neuron membrane potential"
- "multi-compartment spiking neuron LIF SNN dendrite soma 2023 2024"
- "learnable time constant spiking neuron PLIF GLIF parametric LIF 2022 2023 2024"
- "spiking neural network temporal memory state decoupling membrane reset 2023 2024 2025"
- "spiking neuron state space structured dynamics spectral rotation damped oscillation 2024 2025"
- "DH-LIF dendritic heterogeneity spiking neuron SNN 2024 Nature Communications"
- "GLIF gated leaky integrate-and-fire neuron NeurIPS 2022 Yao unified spiking neural network"
- "learnable membrane time constant spiking neuron 2023 2024 survey review"
- "spiking neural network state space model oscillatory dynamics spectral parametrization 2024 2025 arxiv"
- "damped oscillation spiking neuron model structured dynamics spectral 2024"
- "PRF Parallel Resonate and Fire neuron long sequence spiking 2024 arxiv"
- "SSM spiking neuron structured state space model 2024 2025 AAAI NeurIPS"

All queries used public keywords only (method names, venue names, task descriptions). No private draft text, results, or unpublished material from the SPRiF manuscript was included in search queries.

## Sources Checked

- arXiv.org — primary paper pages for abstracts and metadata
- Semantic Scholar — discovery and citation graph
- IEEE Xplore — DA-LIF, CRIMF verification
- OpenReview — NeurIPS 2023 PSN paper, ICLR 2025 PRF submission
- AAAI OJS — SpikingSSMs AAAI 2025 verification
- Nature Communications — AdLIF publication verification
- GitHub — BRF open-source code repository

## Excluded Sources

- Policy-excluded sources (MDPI domains): none encountered in relevant results
- Low-quality sources: generic blog posts, non-peer-reviewed preprints from unknown groups without code or evidence were screened out during candidate review
- Untraceable PDFs: none encountered
- Search snippets used only: 0 papers — all 12 final candidates were verified through primary paper pages (arXiv abstract pages or publisher pages)

## Unknowns

- Papers not accessible: SiLIF (Jun 2025) — preprint only, no peer review yet; this is the closest competitor and its venue status is uncertain. Full text not accessed; claims based on abstract.
- Venue status not verified: PRF (Oct 2024) — ICLR 2025 submission status not confirmed from OpenReview page at search time
- Missing benchmark details: SPikE-SSM (Oct 2024) — full benchmark tables not accessed; classified as background only
- Potential missed papers: The SSM-SNN intersection is very active (2024-2025); additional preprints may appear before SPRiF submission. Recommend re-running search closer to submission.
- Conference proceedings: CVPR 2025, ICML 2025 proceedings were still being published at search time; may contain relevant new SNN papers
- "Projective reset" / "directional reset" as a term: returned zero relevant results — this terminology appears unique to SPRiF, supporting novelty of the reset mechanism

## Handoff Notes

- For writing: Cluster 1 (BRF/PRF) and Cluster 3 (SiLIF/SpikingSSMs) are the two groups that must be explicitly discussed in Related Work. AdLIF is the strongest experimental baseline and should be included for comparison. A direct comparison table (SPRiF vs SiLIF vs BRF vs AdLIF) would help reviewers.
- For idea optimization: SiLIF (Jun 2025) is the most urgent differentiation target. SPRiF's unique claim should emphasize "functional slow/fast state decomposition at the neuron level" vs SiLIF's "SSM parametrization for training stability." The projective reset mechanism has no direct analog in any known work — this is the strongest novelty signal.
- For experiment design: Include BRF and AdLIF as baselines if feasible. Consider adding LRA benchmark for long-range capability evidence (many SSM-SNN papers use it). Parameter-matched comparisons are essential given SPRiF's multi-state design (5D total per neuron vs 1D for LIF).
- For review: A reviewer familiar with the SSM-SNN field may ask "how is SPRiF different from SiLIF or SpikingSSMs?" — prepare a direct comparison table in the paper. A reviewer familiar with computational neuroscience may ask about multi-compartment models and slow/fast decomposition — explicitly address the spatial-vs-functional distinction (SPRiF's compartments are functional roles, not physical dendrite/soma).
