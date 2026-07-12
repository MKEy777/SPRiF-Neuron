# SPRiF AAAI 2027 Conference Review

## 1. Report Metadata

| Field | Value |
|:---|:---|
| Review date | 2026-07-11 |
| Target venue/year/track | AAAI 2027, main track |
| Paper title | SPRiF: Spectral Projective Reset Integrate-and-Fire Neuron |
| Input materials reviewed | `SPRiF_AAAI2027.tex` (334 lines, main paper), appendix referenced but not read in full |
| Search basis | arXiv (4 queries), Semantic Scholar (rate-limited) |
| Report file | `ccfa-review-reports/2026-07-11-sprif-aaai-conference-review.md` |
| Reviewer mode | scientific, standard |

## 2. Desk Rejection Assessment

| Check | Status | Evidence |
|:---|:---|:---|
| Paper length | Pass | `letterpaper` article, two-column format. Text volume appears compatible with AAAI 7+1 page budget; compilation needed to confirm. |
| Topic compatibility | Pass | SNN neuron design for temporal processing — fits AAAI's broad AI scope well. |
| Minimum quality | Pass | All required sections present. Method is mathematically specified. Experiments cover 5 benchmarks. |
| Policy/anonymity/compliance | Pass | Author set to "Anonymous Submission". Template version 2027.1. No identity-revealing links or artifacts visible. |
| Prompt injection | Pass | No hidden instructions detected in text, comments, or captions. |
| Ethics and reviewability | Pass | Limitations section addresses scope boundaries. No human subjects, sensitive data, or misuse risk requiring additional disclosure. |

**Desk rejection risk:** none

## 3. Paper Summary And Contribution Map

### Summary

The paper proposes SPRiF, a spiking neuron that decouples temporal memory storage from spike readout and reset through *functional state decomposition*. Each SPRiF neuron contains: (1) a slow 3D spectral state (real exponential decay + damped rotation) that is never reset by spikes, and (2) a fast 2D discharge state that produces spikes and absorbs a learnable projective reset along $[1,\lambda]^\top$. The core claim is that structurally insulating memory from spike-triggered perturbation improves temporal information retention without abandoning binary communication or increasing network size.

### Claimed Problem

In standard spiking neurons (LIF and variants), the same membrane state accumulates temporal context, triggers spikes, and receives the post-spike reset — causing spike events to perturb the very state that stores input history.

### Claimed Gap

Existing extensions (resonant, adaptive, multi-timescale, reset innovations) enrich dynamics but keep memory storage coupled to the spike-triggered reset target. No prior work structurally separates memory-bearing state from discharge/readout state.

### Method/Contribution Map

| # | Contribution | Section | Evidence |
|:---|:---|:---|:---|
| C1 | Functional state decomposition principle (slow memory + fast discharge) | §3.1 | SI-DMS experiment, state-visualization in Fig.6 |
| C2 | Constrained spectral slow state with real+rotational modes | §3.2 | Impulse response analysis, α/ρ/ω learned distributions |
| C3 | Projective fast-state reset (learnable directional reset on discharge only) | §3.3 | λ=0 ablation, learned λ distributions, correlation analysis |
| C4 | Competitive/SOTA accuracy across 5 temporal benchmarks | §4.1-4.2 | Table 1 (main results) |
| C5 | Ablation: state separation >> rotation > projective reset | §4.3 | Table 2 (ablation) |
| C6 | SI-DMS: separation protects delay-period memory under spike stress | §4.4 | Table 3 + Fig.5 |

### Evidence Package

- 5 benchmarks: S-MNIST, PS-MNIST, QTDB, GSC, SHD
- 12+ baseline methods from published literature
- 3 mechanism ablations (ω=0, merged states, λ=0)
- Controlled causal experiment: SI-DMS with forced-spike intervention
- Impulse response analysis + learned kernel distributions
- Learned reset direction analysis (λ distributions, correlations)

### Stated Limitations

Single-run results, baselines from different protocols, no neuromorphic hardware evaluation, small-medium scale tasks only, no proof that every learned mode is necessary for every input.

## 4. Search And Related-Work Basis

| Query | Source | Results | Relevance |
|:---|:---|:---|:---|
| "functional state decomposition" spiking neuron reset memory | arXiv | 0 results | Confirms novelty of terminology |
| "spectral projective reset" integrate fire neuron | arXiv | 0 results | Confirms novelty of method name |
| "spiking neuron state decomposition" slow fast | arXiv | 0 results | No direct overlap found |
| "spiking neuron projective reset" | arXiv | 1 result (Rodriguez-Garcia et al., 2025, q-bio.NC) | Biological two-compartment model; different domain, no methodological overlap |

**Closest works found:** Not a single paper combining (i) functional decomposition into un-reset memory + discharge, (ii) constrained spectral basis, and (iii) learnable directional reset. The paper properly cites BRF, TC-LIF, DH-SNN, ASRNN as closest competitors. A 2025/2026 gap in the baseline table is notable but may reflect the paper's submission timeline.

**Unverified related-work risks:** Limited search scope due to API rate-limiting. A deeper search on OpenReview (ICLR 2026, NeurIPS 2026) and recent arXiv SNN papers post-2025 Q1 would strengthen confidence.

**Source-quality screening:** ArXiv search only; conference proceedings not exhaustively searched. Confidence: moderate.

## 5. Expected Review Outcome

| | |
|:---|:---|
| Expected outcome | **Weak Accept (borderline 6-7)** |
| Main accept signal | Clean conceptual contribution (functional state decomposition) with compelling causal evidence (SI-DMS) and competitive benchmarks |
| Main reject signal | Single-run results prevent significance claims; S-MNIST margin (0.08) is within noise; no neuromorphic deployment advantage demonstrated |
| Confidence | 3/5 (appendix not fully read, search limited by API restrictions) |

## 6. Strengths And Weaknesses

### Strengths

**S1. Clear conceptual contribution.** The functional state decomposition principle is crisply articulated and distinguished from rate decomposition. The central question ("must the state that triggers a spike also store temporal memory?") frames the contribution effectively.

**S2. Strong causal evidence (SI-DMS).** The spike-intervention experiment directly tests the paper's core mechanism: SPRiF retains 92.8% accuracy under forced-spike stress while the merged variant collapses to 51.7%. This is a well-designed ablation that goes beyond correlation.

**S3. Comprehensive benchmark coverage.** Five datasets spanning four modalities (pixel sequences, ECG, speech, event streams) demonstrate broad applicability. The baseline table includes 12+ published methods.

**S4. Multi-axis ablation design.** Three mechanism ablations (rotation, separation, reset direction) cleanly isolate each component's contribution. The finding that state separation dominates (2.38-5.04 points) supports the central thesis.

**S5. Honest result framing.** The paper explicitly states that the comparison "does not establish universal superiority" and notes the competitive-but-not-dominant position on GSC and SHD. This builds reviewer trust.

**S6. Interpretable design.** Spectral parameters (α, ρ, ω) have direct dynamical meaning. Impulse response analysis and learned kernel distributions provide mechanistic insight beyond black-box benchmarking.

### Weaknesses

**W1. Single-run results undermine SOTA claims.** [Evidence: §4.2, line "The SPRiF entries are single-run values from task-specific implementations"] The paper claims "strongest reported entry" on three benchmarks, but single runs without error bars or repeated-seed statistics make this claim fragile. On S-MNIST, the 0.08-point margin over TC-LIF could easily vanish under 3-5 seed averaging. AAAI reviewers will expect at minimum mean±std or confidence intervals for SOTA claims.

**Reviewer deduction:** The evidence package is strong but the statistical rigor is insufficient for the strength of the claims made. Score impact: -1 on evidence dimension.

**Required fix:** Report mean±std across ≥3 seeds for all SPRiF entries in Table 1, with matching seed statistics for at least the closest baseline (TC-LIF or BRF). At minimum, add a footnote quantifying seed variance on the three lead datasets.

---

**W2. Parameter efficiency comparison is asymmetric.** [Evidence: §4.2, line "approximately 55% fewer parameters..."] SPRiF stores 5 state values per neuron vs. 1 for LIF and ~2-3 for adaptive/resonant variants. The comparison reports end-to-end *network* parameters (dominated by weight matrices) but does not report *activation memory* or *state storage* cost, which differ by a factor of ~3-5×. The parameter-efficiency framing may therefore obscure higher memory pressure at inference.

**Reviewer deduction:** This is a selective reporting concern, not an invalidation of the results. Score impact: -0.5 on soundness if unaddressed.

**Required fix:** Add a column or footnote to Table 1 reporting state-variable count per neuron, or discuss the activation-memory trade-off quantitatively in §4.2. Alternatively, weaken the parameter-efficiency language to "end-to-end weight parameters" and acknowledge state memory cost.

---

**W3. No baseline reimplementation or protocol normalization.** [Evidence: §4.1, line "source studies use task-specific protocols... not intended to support cross-paper significance claims"] The paper correctly acknowledges this limitation but does not mitigate it. Different training protocols, architectures, and hyperparameter tuning budgets across baselines make the comparison table more of a literature survey than a controlled experiment. This is a significant fairness concern.

**Reviewer deduction:** While common in SNN benchmarking, this weakens the "strongest reported entry" framing. Score impact: -0.5 on evidence.

**Required fix:** Either (a) reimplement and retrain the 2-3 strongest baselines under SPRiF's protocol for a fair comparison, or (b) tone down the ranking language (e.g., "achieves competitive accuracy" instead of "strongest reported entry").

---

**W4. Limited scope of significance.** [Evidence: §Limitations] The evaluation is confined to small-to-medium temporal classification. The paper does not demonstrate benefits on large-scale tasks (e.g., longer sequences, larger models, production workloads) or on neuromorphic hardware where SNN efficiency advantages materialize. For an AAAI accept, the practical significance beyond the specific benchmarks needs stronger demonstration.

**Reviewer deduction:** This is primarily a scope limitation, not a technical flaw. It constrains significance but does not independently justify rejection. Score impact: -1 on significance.

---

**W5. Missing ablation on state dimensionality.** [Evidence: §4.3, Table 2] The "merged" ablation combines memory and discharge into a 3D state. But this conflates two changes: removing separation AND reducing total state dimensionality (5D → 3D). A fairer control would be a 5D unified LIF-like state, or a 5D LIF with heterogeneous time constants. This would isolate whether the gain comes from separation per se or simply from more state capacity.

**Reviewer deduction:** Moderate. The SI-DMS experiment partially addresses this concern at the functional level (the merge collapses under spike stress), but a dimensionality-controlled ablation would strengthen the architecture claim.

**Required fix:** Add a "5D LIF" or "5D multi-timescale" baseline to the ablation table, or add a discussion note acknowledging this confound and justifying why functional separation, not dimensionality, is the causal factor (citing SI-DMS results).

---

**W6. Missing wall-clock and energy comparison.** [Evidence: §Limitations] The paper claims competitive parameter count and linear-time recurrence, but provides no measurement of training speed, inference latency, or spike-operation count relative to LIF. For a spiking neuron paper, reviewers will ask whether the added complexity is worth the accuracy gain in practice.

**Reviewer deduction:** Minor-to-moderate. The paper would benefit from at least a per-epoch wall-clock comparison for one dataset (e.g., PS-MNIST).

## 7. Potentially Missing Related Work

| Work | Status | Why relevant | Overlap | Needed comparison |
|:---|:---|:---|:---|:---|
| Rodriguez-Garcia et al. 2025 (q-bio.NC) "Gain neuromodulation in layer-5 pyramidal neurons" | Searched (arXiv) | Biological two-compartment neuron with reset/adaptation split | Low — biological modeling, not SNN architecture; different purpose | Not needed; cite if biological motivation is desired |
| Recent ICLR 2026 / NeurIPS 2026 SNN papers (post-2025 Q1) | Unverified | Could contain unpublished competitors | Unknown | Recommend checking OpenReview for relevant 2025-2026 submissions |

**Status of closest competitors:** The 12+ baselines in Table 1 appear to cover the major published works through 2025. The most recent is DGN (ICLR 2026) and SE-adLIF (Nat. Commun. 2025). No obviously missing major competitor detected.

## 8. Claim-Evidence Audit

| Claim | Where stated | Evidence provided | Strength | Reviewer deduction | Required fix |
|:---|:---|:---|:---|:---|:---|
| Spike events perturb temporal context in standard neurons | Abstract, §1 | Conceptual argument + LIF dynamics | Adequate | Well-established, not contentious | None |
| Functional state decomposition is a new design principle | Abstract, §1, Contributions | Whole-paper evidence package | Strong | Clearly distinguished from rate decomposition | None |
| SPRiF achieves highest accuracy on S-MNIST, PS-MNIST, QTDB | Abstract, §1, §4.2 | Table 1 | Adequate | Single-run; lacks error bars. Margin on S-MNIST (0.08) is marginal | Add seed statistics (W1) |
| State separation is the largest contributing component | §4.3 | Table 2 (ablation) | Strong | Consistent across 3 datasets | Dimensionality confound (W5) |
| Slow-fast separation protects delay-period memory under spike stress | §4.4, Abstract | SI-DMS experiment (Table 3, Fig.5) | Strong | Well-controlled causal experiment | None |
| Parameter efficiency advantage | §4.2, §1 | Table 1 (parameter counts) | Adequate | Ignores state-memory cost | Address activation memory (W2) |
| Learned kernels are heterogeneous + oscillatory | §4.5 | Impulse response figures (appendix) | Adequate | Appendix not fully reviewed; qualitative claims | None |
| SPRiF does not require larger network to attain accuracies | §4.2 | Table 1 parameter counts | Adequate | End-to-end parameters smaller but state variables larger | Address state-memory trade-off (W2) |

## 9. Experiment / Benchmark / Reproducibility Audit

### Baselines
12+ published recurrent SNN methods covering LIF, gated (GLIF), adaptive (ASRNN, SE-adLIF, RadLIF), dendritic (DH-SNN), resonate-and-fire (BRF), compartmental (TC-LIF), and recent architectures (MPS-SNN, DGN, Heterogeneous SNN). Coverage is strong, though some entries are from different protocols. **Verdict: adequate+**

### Ablations
Three mechanism ablations (ω=0, merged, λ=0) on 3 datasets. Clean design, isolates each component. Missing dimensionality-controlled ablation (W5). **Verdict: adequate**

### Datasets/Benchmarks
Five diverse temporal benchmarks. Well-chosen for breadth. All are standard in SNN literature. **Verdict: strong**

### Metrics
Accuracy only. Appropriate for classification tasks. No F1, confusion matrices, or per-class breakdown. **Verdict: adequate**

### Statistical Rigor
Single runs for main results. Seeds specified but no error bars, confidence intervals, or statistical tests. SI-DMS uses 3-seed averaging which is good. **Verdict: weak** (W1)

### Robustness/Failure Cases
SI-DMS provides stress-test. Impulse response analysis examines neuron-level behavior. Noise robustness mentioned as in-appendix. No adversarial or out-of-distribution tests. **Verdict: adequate**

### Implementation Details
Architectures (widths, layer counts), hyperparameters (learning rate, batch size, epochs, optimizer, BPTT chunk length), and seeds provided in §4.1. Full details deferred to appendix. **Verdict: adequate**

### Artifacts and Reproducibility
Code promised upon acceptance. No code link in current submission. Training configs are specific enough to reimplement. **Verdict: adequate (contingent on code release)**

### Limitations
Honest and comprehensive. Covers single-run, protocol differences, scale, hardware, and mode-necessity concerns. **Verdict: strong**

## 10. Multi-Reviewer Panel

### Reviewer 1 — Best-Justified (Sympathetic Expert)

| | |
|:---|:---|
| Expertise | Spiking neural networks, temporal processing |
| Likely score | 7 (weak accept) |
| Confidence | 4/5 |
| Main positive signal | The functional state decomposition concept is clean and the SI-DMS experiment provides direct causal evidence that separation protects memory. This is a genuine neuron-level insight, not incremental tuning. |
| Main negative signal | Single-run results without error bars weaken the quantitative claims. The S-MNIST margin could disappear under multiple seeds. |
| Evidence basis | §3 (method), §4.3-4.4 (ablations + SI-DMS), Table 1 |
| Score-change condition | Would raise to 8 if seed statistics show statistically significant margins on the three lead datasets. |

### Reviewer 2 — Critical (Skeptical Reviewer)

| | |
|:---|:---|
| Expertise | Machine learning, empirical evaluation methodology |
| Likely score | 5 (borderline negative) |
| Confidence | 3/5 |
| Main positive signal | The SI-DMS experiment is clever and the mechanism is well-motivated conceptually. |
| Main negative signal | Five state variables per neuron for modest accuracy gains (0.08 on S-MNIST, 0.50 on PS-MNIST). No wall-clock or spike-count comparison. For an AAAI paper, I expect a stronger case that this complexity is worth it. The single-run protocol is not acceptable for SOTA claims. |
| Evidence basis | §4.1-4.2, Limitations |
| Score-change condition | Would raise to 6 if wall-clock/memory comparison added and multi-seed statistics reported. Would remain skeptical without neuromorphic benefit demonstration. |

### Reviewer 3 — Method / Soundness

| | |
|:---|:---|
| Expertise | Dynamical systems, SNN theory |
| Likely score | 7 (weak accept) |
| Confidence | 4/5 |
| Main positive signal | The spectral parameterization is well-constrained (α, ρ, ω have clear dynamical meaning). The linear-time recurrence with structured state update is correctly argued. The projective reset on fast state only is mathematically clean. |
| Main negative signal | The merged-state ablation reduces total dimensionality (5D → 3D), conflating functional separation with state capacity. A 5D unified baseline would strengthen the claim. |
| Evidence basis | §3 (method), §4.3 (ablations) |
| Score-change condition | Would raise to 8 with dimensionality-controlled ablation + formal stability proof for the coupled slow-fast system. |

### Reviewer 4 — Evidence / Experiment

| | |
|:---|:---|
| Expertise | Empirical ML, benchmarking |
| Likely score | 6 (borderline positive) |
| Confidence | 3/5 |
| Main positive signal | Five benchmarks spanning four modalities is good breadth. Ablations are clean and consistent. SI-DMS is a high-quality causal test. |
| Main negative signal | Single-run results are below AAAI standard for SOTA claims. Cross-paper protocol differences are acknowledged but not mitigated. Parameter efficiency framing is one-sided (ignores state memory). |
| Evidence basis | §4.1-4.5 |
| Score-change condition | Would raise to 7 with 3-seed statistics + per-neuron state-memory reporting. Would raise to 8 with protocol-normalized baseline reimplementation. |

### Reviewer 5 — Novelty / Positioning

| | |
|:---|:---|
| Expertise | SNN architecture, neuromorphic computing |
| Likely score | 8 (accept) |
| Confidence | 4/5 |
| Main positive signal | The functional-vs-rate decomposition distinction is a genuine conceptual advance. It reframes the design space of spiking neurons in a way that existing work (LIF extensions, oscillatory neurons, reset innovations) does not. The paper properly positions against BRF, TC-LIF, DH-SNN, and INFLOR/RPLIF/ARLIF. Search confirms no directly overlapping work. |
| Main negative signal | The paper could cite more work on the theoretical benefits of state augmentation in RNNs/SSMs (e.g., connections to structured state-space models like S4/Mamba) to broaden the argument. |
| Evidence basis | §2 (Related Work), literature search |
| Score-change condition | Minor — none critical. |

### Reviewer 6 — Writing / Clarity

| | |
|:---|:---|
| Expertise | Scientific writing, AAAI reviewing |
| Likely score | 7 (weak accept) |
| Confidence | 4/5 |
| Main positive signal | The paper is well-written: clear research question, clean method exposition, honest result framing. The introduction's flow from problem → gap → question → method is effective. Figures and tables are properly narrated. |
| Main negative signal | The training-details paragraph (§4.1) is dense and the benchmark results paragraph (§4.2) packs many numbers. Some sentences in §4.5 (dynamical analysis) are difficult to parse without the appendix figures. |
| Evidence basis | Full paper |
| Score-change condition | Minor. Would not change score independently. |

### Reviewer 7 — Ethics / Reproducibility

| | |
|:---|:---|
| Expertise | Reproducibility, ML ethics |
| Likely score | 6 (borderline positive) |
| Confidence | 3/5 |
| Main positive signal | Training configs, seeds, and architectures are specified. Limitations are honest. No ethics concerns with datasets or application domain. Code promised. |
| Main negative signal | Single-seed results limit reproducibility assessment. No explicit reproducibility checklist. Appendix not reviewed in full — may contain missing details. |
| Evidence basis | §4.1, Limitations |
| Score-change condition | Would raise to 7 with multi-seed statistics and an explicit reproducibility statement. |

### Panel Synthesis

**Agreement:** All reviewers recognize the conceptual contribution (functional state decomposition) as novel and well-motivated. The SI-DMS experiment is universally praised as strong causal evidence. The benchmark coverage is seen as adequate.

**Disagreement:** Reviewers diverge on whether the empirical evidence package meets AAAI's bar. Reviewer 5 (novelty) sees a clear accept. Reviewer 2 (critical) sees borderline rejection due to single-run results and limited practical significance.

**Decisive positive axis:** Functional state decomposition is a genuine neuron-level insight with compelling causal evidence (SI-DMS).

**Decisive negative axis:** Single-run results prevent confident SOTA claims. Cross-paper protocol differences weaken the quantitative comparison. No practical (wall-clock/energy/memory) advantage demonstrated.

**Unresolved evidence:** Appendix figures for impulse response and ablation grids not reviewed. Dimensionality confound in merged ablation not resolved.

**AC stance:** Weak Accept (6-7). The conceptual contribution is strong enough to justify publication, but the paper should either (a) add multi-seed statistics to substantiate quantitative claims, or (b) reframe as a mechanism paper rather than a SOTA claim paper. The current framing ("strongest reported entry") invites the wrong expectations for a single-run study.

## 11. Concerns Table

| ID | Severity | Concern | Evidence basis | Affected criterion | Fix class | Required action | Owner skill | Score-change condition |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| C1 | Major | Single-run results without error bars or seed statistics | §4.2, Table 1 | Evidence, Reproducibility | experiment | Report mean±std across ≥3 seeds for all SPRiF entries in Table 1 | ccf-experiment-designer | +1 overall if multi-seed margins remain significant |
| C2 | Moderate | Parameter efficiency comparison ignores activation memory cost | §4.2, §3.4 | Soundness, Evidence | experiment | Add state-variable-per-neuron count to Table 1 or discuss activation memory trade-off | ccf-paper-writer | +0.5 evidence if addressed |
| C3 | Moderate | Cross-paper protocol differences weaken "strongest entry" framing | §4.1, §4.2 | Evidence, Significance | experiment | Either reimplement top-3 baselines under same protocol or weaken ranking language | ccf-experiment-designer | +0.5 to +1 significance if baselines reimplemented |
| C4 | Moderate | Dimensionality confound in merged-state ablation | §4.3, Table 2 | Soundness | experiment | Add 5D unified baseline or discuss confound + SI-DMS as causal rebuttal | ccf-experiment-designer | +0.5 soundness if addressed |
| C5 | Minor | No wall-clock or spike-count comparison | §4.2, Limitations | Significance | experiment | Report per-epoch training time and spike count vs. LIF on one dataset | ccf-experiment-designer | +0.5 significance if addressed |
| C6 | Minor | Training-details paragraph is overly dense | §4.1 | Clarity | writing | Split into bullet list or separate per-dataset subsections | ccf-paper-writer | Clarity improved but unlikely to change score |
| C7 | Minor | Recent SNN papers (post-2025 Q1) not exhaustively searched | §2, §4.2 | Novelty, Positioning | related-work | Search OpenReview ICLR/NeurIPS 2026 for new competitors | ccf-literature-searcher | No score change unless direct overlap found |
| C8 | Minor | No reproducibility checklist or explicit code-link policy statement | §4.1 | Reproducibility | reproducibility | Add AAAI reproducibility checklist item or state release plan explicitly | ccf-paper-writer | +0.5 reproducibility if addressed |

## 12. AC / Meta-Review

### Consensus
Reviewers agree the core conceptual contribution (functional state decomposition) is novel, well-motivated, and supported by the SI-DMS causal experiment. The method is technically sound. The benchmark coverage is adequate.

### Disagreement
The panel splits on whether the empirical evidence meets AAAI's bar. The positive reviewers value the conceptual novelty and causal evidence over statistical rigor. The critical reviewer considers single-run SOTA claims insufficient for a CCF-A venue. The AC leans toward accepting with revision, contingent on multi-seed statistics.

### Decisive Acceptance Axis
Functional state decomposition reframes a design question that existing SNN work has not explicitly addressed. The SI-DMS experiment provides rare causal evidence in neuron-design papers.

### Decisive Rejection Axis
If the paper cannot produce multi-seed statistics confirming the SOTA margins, the quantitative claims are not credible enough for a CCF-A accept. The paper could still be accepted if it reframes as a mechanism/insight paper and drops unverified SOTA language.

### Discussion Risks
- Reviewer 2 may resist raising score unless practical significance (wall-clock, memory) is demonstrated
- AAAI's reputation for empirical rigor may lead other reviewers to share the statistical concern
- The 0.08 S-MNIST margin is the most fragile claim and should not headline the evidence package

### AC Stance
**Weak Accept**, conditional on adding multi-seed statistics for the three lead benchmarks and either adding a dimensionality-controlled ablation or discussing the confound explicitly.

## 13. Quantitative Scores

### Scorecard

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
|:---|---:|:---:|:---|:---|
| Novelty | 4 | 4 | §2, §3, literature search | Clean functional decomposition concept. -1 because the individual components (spectral basis, reset) have precedents; novelty is in the structural combination |
| Soundness | 4 | 3 | §3, §4.3 | Method is well-specified. -1 for dimensionality confound in merged ablation (W5) and asymmetric parameter reporting (W2) |
| Evidence | 3 | 3 | §4.1-4.5, Tables 1-3 | Ablations and SI-DMS are strong. -1 for single-run results (W1), -0.5 for protocol differences (W3). Would be 4 with seed statistics |
| Significance | 3 | 3 | §4, §Limitations | Neuron-level design principle is broadly applicable. -1 for limited to small/medium tasks (W4), -0.5 for no practical efficiency demo (W5) |
| Clarity | 4 | 4 | Full paper | Well-organized, clear question framing, honest result presentation. -1 for dense training details paragraph |
| Reproducibility | 3 | 3 | §4.1, appendix (unreviewed) | Configs provided, code promised. -1 for single seeds (W1), -0.5 for no checklist |
| Ethics / Limitations | 4 | 3 | Limitations section | Honest and comprehensive limitations. -1 because limitations identify problems but don't propose mitigation |

**Overall:** 6 (borderline positive, lean toward weak accept)

| Metric | Score |
|:---|:---|
| Quality | 4 |
| Clarity | 4 |
| Significance | 3 |
| Originality | 4 |
| Soundness | 4 |
| Evidence | 3 |
| Reproducibility | 3 |
| Ethics / Limitations | 4 |
| Overall | 6 |
| Confidence | 3 |

**Recommendation:** Weak Accept (conditional on revision)

### Score-Change Conditions

| Change | Condition | Likely affected dimensions | Expected movement |
|:---|:---|:---|:---|
| Raise score | Add ≥3-seed mean±std for all SPRiF entries + at least top-2 baselines; margins remain significant | Evidence, Reproducibility | +1 overall (6 → 7) |
| Raise score | Add wall-clock + spike-count comparison vs LIF on one dataset; show acceptable overhead | Significance | +0.5 overall |
| Raise score | Add dimensionality-controlled ablation (5D unified baseline) | Soundness | +0.5 overall |
| Lower score | Multi-seed statistics reveal S-MNIST margin disappears or PS-MNIST margin shrinks to <0.2 | Evidence | -1 to -2 overall (6 → 4-5) |
| Lower score | Direct competitor paper found with overlapping functional decomposition concept | Novelty | -1 to -2 overall (could become fatal) |
| No quick change | Reimplement baselines under unified protocol | Evidence | Would require significant new experiments; unlikely before submission deadline |

## 14. Questions For Authors

1. **What is the seed-to-seed variance on S-MNIST, PS-MNIST, and QTDB?** Given the 0.08 margin on S-MNIST, this is the single most decision-relevant question.

2. **How does the per-neuron activation memory (5 state values vs 1 for LIF) affect training throughput and inference memory on a realistic batch size?** The parameter-count framing tells only half the efficiency story.

3. **Would a 5-dimensional LIF with heterogeneous time constants perform closer to SPRiF or closer to the merged 3D variant?** This would disentangle the dimensionality confound from the functional-separation claim.

4. **Are there any 2025-2026 SNN papers (e.g., ICLR 2026, NeurIPS 2026 submissions) that propose similar memory-discharge separation?** The baseline table's most recent entries are from 2024-2025.

## 15. Score Revision Criteria

**Raising the score would require:**
- Multi-seed statistics (≥3 seeds) with mean±std for all SPRiF entries in Table 1
- At minimum, a footnote or sentence quantifying seed variance on the three lead datasets

**Lowering the score would be triggered by:**
- Seed variance that erases the S-MNIST margin or substantially shrinks PS-MNIST gains
- Discovery of a directly overlapping prior work (functional decomposition in spiking neurons)

**Concerns unlikely to change before submission:**
- Protocol-normalized baseline reimplementation (time-intensive)
- Neuromorphic hardware deployment (out of scope)
- Long-context benchmark evaluation (out of scope)

## 16. Action Plan And CCFA Handoffs

| Priority | Action | Owner skill | Input needed | Expected output | Handoff required |
|:---|:---|:---|:---|:---|
| P0 | Report multi-seed statistics for SPRiF (≥3 seeds) on S-MNIST, PS-MNIST, QTDB | ccf-experiment-designer | Current training configs | Table with mean±std; update Table 1 | Yes |
| P1 | Add activation-memory or state-variable discussion to address W2 | ccf-paper-writer | State-variable counts per neuron type | Revised §4.2 with memory trade-off note | Yes |
| P2 | Weaken "strongest reported entry" to "competitive" or add seed-statistics caveat | ccf-paper-writer | P0 output | Revised §4.2 and Abstract | Yes |
| P3 | Add dimensionality-controlled ablation discussion (or 5D LIF baseline) | ccf-experiment-designer | Ablation design | Revised §4.3 with confound note or new experiment | Maybe |
| P4 | Search recent SNN literature (OpenReview 2025-2026) for missing competitors | ccf-literature-searcher | None | List of potentially missing papers | Maybe |
| P5 | Report wall-clock time and spike count vs LIF on one dataset | ccf-experiment-designer | Benchmark infrastructure | Efficiency comparison table | Optional |

---

**Checks run:** Desk check (pass), contribution pass, evidence pass, adversarial pass, literature search (4 arXiv queries), claim-evidence audit, experiment/reproducibility audit, 7-reviewer panel, AC synthesis

**Checks skipped:** Full appendix review, complete conference-proceedings search, LaTeX compilation and page-count verification, supplementary material review

**Unresolved risks:**
- Seed variance unknown (P0)
- Dimensionality confound in merged ablation (P3)
- 2025-2026 literature gap (P4)
