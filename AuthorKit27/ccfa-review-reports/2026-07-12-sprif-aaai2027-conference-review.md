# Conference Review — SPRiF: Spectral Projective Reset Integrate-and-Fire Neuron

- **Venue / year / track:** AAAI 2027 (submission mode, anonymous). Assumed regular oral/ poster track, CCF-A CS conference criteria.
- **Manuscript scope:** Main paper `SPRiF_AAAI2027.tex` (399 lines) + technical appendix `SPRiF_AAAI2027_supp.tex`. Reviewed both.
- **Review date:** 2026-07-12
- **Review mode:** standard (full scientific review, multi-reviewer panel, AC synthesis)
- **Inputs available:** local TeX source + compiled PDF present; bib file `sprif2027.bib`; figures present. No source code (released upon acceptance).

---

## 1. Desk Checks

| Check | Result |
|---|---|
| Topic fit (AAAI AI / neural computation) | Pass — spiking neural networks, temporal learning. |
| Anonymity / policy | Pass — `\author{Anonymous Submission}`, empty affiliations, no identifying paths. |
| Length / page limit | Pass — main paper ~8 pp, within AAAI 2027 content limit; supplementary separate. |
| Minimum quality / reviewability | Pass — complete method, experiments, ablations, analyses. |
| Ethics / dual-use | Pass — no ethical red flags; standard ML benchmarks. |
| Hidden manipulation / prompt injection | Pass — none. |
| Compilation | Pass — both `.tex` compile (exit 0, no undefined refs/cites). |

No desk-rejection risk.

---

## 2. Paper Summary

**Problem.** Conventional LIF-family spiking neurons couple temporal integration, spike generation, and post-spike reset in one membrane state; each spike perturbs the very trace that must preserve context.

**Proposed principle — functional state decomposition.** Separate (a) a *reset-free slow spectral memory state* from (b) a *fast discharge state* that reads out, spikes, and receives a learnable *projective reset*.

**Model — SPRiF neuron.** Slow state `x ∈ R³`: one real exponential mode + one damped 2-D rotational pair (eigenvalues `α`, `ρe^{±iω}`). Fast state `u ∈ R²`: projection `Gx` plus leaks, membrane `v=u⁰`, spike `z=H(v-θ)`, then projective reset `u ← ũ − zθ[1, λⱼ]ᵀ`. Reset acts only on fast state; `Δx_reset = 0`.

**Evidence package.** Five temporal benchmarks (S-MNIST, PS-MNIST, QTDB, GSC, SHD); three mechanism ablations; a controlled Spike-Intervention Delayed-Match-to-Sample (SI-DMS) task; impulse-response / temporal-kernel analysis. 5-seed main results, 3-seed SI-DMS.

**Stated contributions.** (1) functional state decomposition principle; (2) SPRiF instantiation (constrained spectral slow state + projective reset); (3) empirical evaluation across 5 benchmarks + ablations + analyses.

---

## 3. Related Work & Novelty Positioning (public-safe search performed)

The paper cites a reasonable set of neuron-level prior art (LIF extensions, BRF/PRF/DRF resonate, TC-LIF, DH-SNN, ASRNN, adaptive/learnable reset) and frames functional state decomposition as "orthogonal to rate decomposition and single-state reset."

**Gap found (needs addressing before camera-ready):**
- **Two-compartment memory/discharge split is anticipated.** LSTM-LIF (Zhang et al., arXiv:2307.07231, 2023) explicitly separates a *dendritic (long-term memory)* compartment from a *somatic (short-term / discharge)* compartment, and notes memory traces are "not corrupted" by somatic reset. This is conceptually very close to SPRiF's slow/fast separation and is **not cited**.
- **Explicit "memory module + spiking module" decomposition.** Zhang (2025), "Revisiting Reset Mechanisms in SNNs for Sequential Modeling" (arXiv:2504.17751), directly decomposes an SNN into a memory module and a spiking module and argues reset should be decoupled — squarely relevant to the paper's central thesis, **not cited**.
- **Decoupled/learnable reset & SSM-spiking.** PRF (Huang et al., 2024, arXiv:2410.03530) and SpikingSSM / SpikeSSM / Karilanova et al. (2025, "Low-Bit Data Processing Using Multiple-Output Spiking Neurons", arXiv:2508.06292) also treat reset as a learnable, decoupled operator and bridge SSMs and spiking neurons. The spectral 3-D slow state overlaps with complex-valued/SSM neuron lines (TC-LIF already cited; SpikingSSM family not).
- **Adaptive/learnable reset.** AR-LIF, RPLIF, PSN (several cited) — the paper's learned `λ` is in a crowded sub-area; positioning vs. "learnable reset direction" prior art is adequate but could be tighter.

**Novelty assessment.** SPRiF's *specific* instantiation — a constrained 3-D *spectral* slow state (real + rotational, sigmoid-bounded, stable) with a *learned projective reset direction* `λⱼ` and a clean structural reset-separation proof — is genuinely distinct from LSTM-LIF (which keeps reset on both compartments) and from the SSM-decomposition line (which typically drops reset or uses fixed refractory). However, the **principle-level novelty claim is overstated**: separating memory from discharge is not new; the contribution is the *spectral + projective-reset* instantiation and the controlled analysis. Reframing is required.

---

## 4. Claim–Evidence Audit (major claims)

| Claim (location) | Evidence | Verdict |
|---|---|---|
| "Highest mean accuracy among compared methods on three tasks" (abs/intro/concl) | Tab. 1: S-MNIST 99.28 > TC-LIF 99.20; PS-MNIST 95.86 > TC-LIF 95.36; QTDB 88.43 > BRF 87.00. | **Supported.** GSC/SHD clearly "competitive" (2nd). OK. |
| "Parameter count lower than strongest compared entry on four datasets, comparable on QTDB" (intro) | SPRiF 0.00177M vs BRF 0.00173M on QTDB → slightly *higher*, so "comparable" holds; lower on other 4. | **Supported** (caveat already stated). |
| "Merging memory and discharge causes the largest degradation" / "merged-state variant showed the largest degradation" (abs, L322, concl) | Tab. 2: PS-MNIST drops — ω=0: **3.53**, Merged: 2.38, λ=0: 0.50. On PS-MNIST the *largest* drop is rotation removal, **not** merge. | **CONTRADICTED by own table.** Major. |
| "ablations associate slow–fast separation with the largest performance differences" (abs) | Same as above; false on PS-MNIST. | **Overclaim.** Major. |
| "structural separation, rather than λ alone, was important" (SI-DMS) | SPRiF-merged 51.7% (chance) vs SPRiF-full 92.8%; λ=0 only 0.6 pp drop. | **Supported** (within-family controlled comparison is strong). |
| Impulse-response kernel advantage (τ 272/138 vs ASRNN 34/46) | Fig. temporal_kernels; authors state "descriptive, not controlled," 2 tasks only. | **Supported as descriptive.** |
| Cross-paper comparison is "descriptive, not significance test" | Stated L287, L292. | **Honest.** Good. |

**The ablation overclaim is the single most damaging writing/claim issue**: the abstract and conclusion assert merged-state is the dominant degradation factor, but Tab. 2 shows rotation removal (`ω=0`) hurts *more* on PS-MNIST. Either the sentence must be qualified ("on 2 of 3 datasets / excluding the long permuted sequence where rotation dominates") or the conclusion reversed to "state separation and spectral rotation are both major, with rotation dominant on PS-MNIST."

---

## 5. Strengths

- **Clear, well-motivated principle** with a clean structural argument (`Δx_reset = 0`, unrolled slow recurrence free of reset).
- **Strong controlled evidence for the core thesis**: the SPRiF-*merged* variant (same family, collapsed state) drops to chance on SI-DMS, while separated variants stay ~92% — a rare *within-model* ablation that actually isolates the claimed mechanism.
- **Honest about cross-paper comparison limits** (descriptive, not significance-tested) and about impulse-response being diagnostic not performance.
- **Broad evaluation** across 5 modalities; reproducibility details in appendix are thorough (seeds, grids, integrity audit).
- Math is sound and the appendixed formal spec is consistent with the main paper.

## 6. Weaknesses (tied to evidence)

1. **Novelty framing overstated** (§3): memory/discharge separation anticipated by LSTM-LIF and the memory/spiking-module decomposition (Zhang 2025); not cited/positioned.
2. **Ablation claim contradicts Table 2 on PS-MNIST** (§4): "merged = largest degradation" is false there.
3. **SI-DMS external baselines at chance** (LIF 50%, ASRNN 52%): the "separation protects memory" argument leans heavily on the within-family merged comparison (which is fine), but the *external* baseline panel is weak and could be read as cherry-picked; BRF (70→65) is the only informative external baseline.
4. **Only 3 seeds for SI-DMS** and `K=8` unseen at train — adequate but thin for the central mechanistic claim.
5. **Rotational-mode benefit is modest** (0.33–3.53 pp) and largest only on the long permuted sequence; the paper sometimes presents rotation as a general gain.
6. **Limitations section is thin**: lists scale + neuromorphic only; omits (a) the cross-paper efficiency caveat, (b) SI-DMS being synthetic, (c) the ablation nuance above.

---

## 7. Reviewer Panel

**R1 — Method / soundness (positive-leaning).** Math is clean; reset-separation proof correct; parameterization guarantees stability. Concern: novelty of *principle* vs. instantiation. Score tendency: 6/10.

**R2 — Evidence / experiment (mixed).** Breadth good; cross-paper caveat honest; SI-DMS within-family comparison is a real strength, but external baselines at chance and only 3 seeds weaken it. Ablation table contradicts the abstract. Score tendency: 5/10.

**R3 — Novelty / positioning (negative-leaning).** Functional state decomposition as a *principle* is anticipated by LSTM-LIF and Zhang 2025's memory/spiking decomposition. Spectral + projective-reset *instantiation* is novel but should be positioned as such, not as a new paradigm. Missing related work is a citable gap. Score tendency: 4–5/10.

**R4 — Writing / clarity (positive).** Readable, well-figured; abstract overclaims (merged-largest). Minor: "orthogonal to rate decomposition" needs the LSTM-LIF/SSM comparison to be defensible. Score tendency: 6/10.

**R5 — Ethics / reproducibility (positive).** Anonymous, no ethical issues, appendix reproducibility strong. Code on acceptance is acceptable for submission. Score tendency: 7/10.

**AC / meta-review synthesis.** The paper has a genuinely useful, well-analyzed neuron design and unusually careful mechanism evidence (the merged-variant SI-DMS result is compelling). Its main liabilities are *presentation*: (i) an overstated principle-level novelty that collides with existing two-compartment / memory-spiking-decomposition literature, and (ii) an abstract/conclusion statement that the ablation table contradicts on PS-MNIST. Neither is fatal; both are fixable by reframing and one sentence. Likely discussion: **borderline accept**, contingent on (a) citing and positioning against LSTM-LIF + Zhang 2025, and (b) correcting the ablation claim.

---

## 8. Calibration & Score

Using an indicative AAAI-style 1–5 (5 = clear accept) / borderline scale:

- **Overall recommendation: 3 (borderline / weak accept), trending to 2 if novelty/ablation issues unaddressed, to 4 after a strong rebuttal.**
- **Novelty: 3/5** (instantiation novel; principle anticipated).
- **Soundness: 4/5** (math clean, one factual slip).
- **Evidence: 3.5/5** (strong within-family SI-DMS; weak external baselines; 3-seed).
- **Clarity: 4/5** (clear, one overclaim).
- **Reproducibility: 4/5.**

No acceptance-probability claim is made.

---

## 9. Concerns Table

| # | Severity | Criterion | Evidence basis | Fix class | Required action | Owner skill | Score-change condition |
|---|---|---|---|---|---|---|---|
| C1 | Major | Novelty / related work | LSTM-LIF (2307.07231), Zhang 2025 (2504.17751) separate memory from discharge; uncited | literature / positioning | Cite + reposition: "separation *principle* anticipated; SPRiF contributes spectral + projective-reset *instantiation*" | `ccf-literature-search` + `ccf-writing-skills` | +1 novelty if repositioned |
| C2 | Major | Claim–evidence | Abs/L322/concl say merged = largest drop; Tab.2 PS-MNIST: ω=0 drop 3.53 > merged 2.38 | writing / claim | Qualify or correct the statement; state rotation dominates on PS-MNIST | `ccf-writing-skills` | removes reject risk |
| C3 | Moderate | Evidence rigor | SI-DMS external baselines at chance; only 3 seeds; K=8 unseen | experiment | Keep within-family comparison as primary; soften external-baseline framing; note 3-seed limit | `ccf-experiment-designer` | stability if acknowledged |
| C4 | Moderate | Novelty sub-area | Learnable reset (`λ`) crowded (AR-LIF, RPLIF, PSN) | positioning | Tighten "projective reset direction" distinction vs. scalar/adaptive reset | `ccf-writing-skills` | minor |
| C5 | Minor | Limitations | Omits cross-paper caveat, SI-DMS synthetic nature, ablation nuance | writing | Add to Limitations paragraph | `ccf-writing-skills` | — |
| C6 | Minor | Clarity | "orthogonal to rate decomposition" undefended vs. SSM-decomposition line | writing | Add one sentence contrasting with TC-LIF/SpikingSSM | `ccf-writing-skills` | — |

---

## 10. CCFA Handoff Actions

- `ccf-literature-search` — verify and pull LSTM-LIF, Zhang 2025 (Revisiting Reset), PRF/SpikingSSM into related work (C1, C4). *Public-safe queries used; results above.*
- `ccf-writing-skills` — reframe novelty (C1), correct ablation overclaim (C2), tighten limitations (C5), add SSM-decomposition contrast (C6).
- `ccf-experiment-designer` — optional: add a 5-seed SI-DMS or a non-chance external baseline to harden C3 (only if authors want stronger evidence).
- `ccf-paper-compressor` — not needed (within page limit).
- `ccf-conference-writing-reviewer` — already performed the appendix writing pass; continue consistency checks after revisions.

---

## 11. Checklist Status

1. Venue explicit ✓ | 2. Desk checks ✓ | 3. Summary/contrib map ✓ | 4. Related-work search ✓ (public-safe) | 5. Strengths/weaknesses tied to evidence ✓ | 6. Claim–evidence audit ✓ | 7. Exp/abl/robust/limit audit ✓ | 8. Reviewer panel (5 + AC) ✓ | 9. Scores calibrated ✓ | 10. Concerns table ✓ | 11. Fixed-format report written ✓ | 12. Handoff actions ✓.

**Unresolved / requires author input:** whether to pursue stronger SI-DMS evidence (C3) or only the writing/positioning fixes; confirmation that LSTM-LIF / Zhang 2025 are the intended comparators.
