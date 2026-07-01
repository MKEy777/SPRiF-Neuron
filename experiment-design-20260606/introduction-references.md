# Reference List for Introduction Section

> Citation keys used in the Introduction draft. All references shared with Related Work.

## Cited in Introduction

| Key | Paper | Venue | Paragraph |
|-----|-------|-------|-----------|
| `\cite{adlif}` | Advancing Spatio-Temporal Processing through Adaptation in Spiking Neural Networks (SE-adLIF) | Nature Communications 2025 | P2, P3 |
| `\cite{dalif}` | DA-LIF: Dual Adaptive Leaky Integrate-and-Fire Model | IEEE 2025 | P2 |
| `\cite{biophysical_adaptation}` | Biophysical Neural Adaptation Mechanisms Enable Artificial... | Nature Communications 2024 | P2 |
| `\cite{brf}` | Balanced Resonate-and-Fire Neurons (BRF) | ICML 2024 | P2, P3 |
| `\cite{prf}` | PRF: Parallel Resonate and Fire Neuron for Long Sequence Learning | arXiv 2024 | P2 |
| `\cite{drf}` | Dendritic Resonate-and-Fire Neuron (D-RF) | NeurIPS 2025 | P2 |
| `\cite{bimodal}` | Recurrent SNNs with Bimodal Neuronal Time Scales | Neural Networks 2026 | P2 |
| `\cite{hetsyn}` | HetSyn: Versatile Timescale Integration via Heterogeneous Synapses | NeurIPS 2025 | P2 |
| `\cite{mtc}` | Multi-Timescale Conductance Spiking Networks | IEEE NICE 2026 | P2 |
| `\cite{ltgate}` | Local Timescale Gates for Timescale-Robust Continual SNNs | arXiv 2025 | P2 |
| `\cite{inflor}` | InfLoR-SNN: Reducing Information Loss for SNNs | ECCV 2022 | P2, P3 |
| `\cite{arlif}` | AR-LIF: Adaptive Reset LIF Neuron | arXiv 2025 | P2, P3 |
| `\cite{rplif}` | Incorporating the Refractory Period into SNNs (RPLIF) | ACM MM 2025 | P2, P3 |
| `\cite{psn}` | Parallel Spiking Neurons (PSN) | NeurIPS 2023 | P2, P3 |

## Baselines Referenced (by name, not necessarily cited in Introduction)

| Baseline | Dataset(s) | Table 1 Result |
|----------|-----------|----------------|
| GLIF (NeurIPS 2022) | S-MNIST, PS-MNIST | 96.64%, 90.47% |
| ASRNN (Nature MI 2021) | S-MNIST, PS-MNIST, QTDB, GSC, SHD | 98.70%, 94.30%, 85.90%, 92.10%, 90.40% |
| DH-SNN (Nature Comms 2024) | S-MNIST, PS-MNIST, QTDB, GSC, SHD | 98.90%, 94.52%, 86.35%, 93.80%, 91.34% |
| BHRF (ICLR 2024) | S-MNIST, PS-MNIST, QTDB, SHD | 99.10%, 95.20%, 87.00%, 91.70% |
| TC-LIF (AAAI 2024) | S-MNIST, PS-MNIST, GSC, SHD | 99.20%, 92.69/95.36%, 94.84%, 88.91% |
| SE-adLIF (Nature Comms 2025) | QTDB | 86.88% |
| SNN-SFA (eLife 2021) | GSC | 91.21% |
| RadLIF (Frontiers 2022) | GSC | 94.51% |
| Heterogeneous SNN (Nature Comms 2021) | SHD | 82.50% |
| MPS-SNN (ICLR 2025) | SHD | 91.19% |
| DGN (ICLR 2026) | SHD | 87.78% |

## SPRiF Results Referenced in Introduction

| Dataset | SPRiF Accuracy | Best Baseline | Baseline Accuracy | Margin |
|---------|---------------|---------------|-------------------|--------|
| S-MNIST | 99.28% | TC-LIF | 99.20% | +0.08 |
| PS-MNIST | 95.86% | TC-LIF | 95.36% | +0.50 |
| QTDB | 88.43% | BHRF | 87.00% | +1.43 |
| GSC | 94.55% | TC-LIF | 94.84% | -0.29 |
| SHD | 91.52% | BHRF | 91.70% | -0.18 |

> **Note on GSC and SHD**: SPRiF is competitive (within 0.3 points) but not #1 on these two datasets.
> The Introduction draft says "best on four of five" — this should be corrected to "best on three of five"
> (S-MNIST, PS-MNIST, QTDB) and "competitive on the remaining two" (GSC, SHD).
> **Action: Fix this in the draft before submission.**

## Ablation Deltas Referenced in Introduction

| Ablation | PS-MNIST | GSC | QTDB | Range |
|----------|----------|-----|------|-------|
| A: ω=0 (remove rotation) | -3.53 | -0.72 | -0.65 | 0.65–3.53 |
| B: merged (no separation) | -2.38 | -4.50 | -5.04 | 2.38–5.04 |
| C: scalar reset (λ=0) | -0.50 | -0.33 | -1.03 | 0.33–1.03 |

> **Correction**: The Introduction draft says "2.4–3.5" for Ablation A. Actual range is 0.65–3.53.
> The PS-MNIST delta (3.53) dominates; GSC (0.72) and QTDB (0.65) are much smaller.
> **Action: Correct the delta ranges to reflect all three datasets accurately.**

---

**Total unique Introduction references: 14** (subset of Related Work's 29)
