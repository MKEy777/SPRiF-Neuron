# Related Work Draft — Checklist & Score-Risk Diagnosis

## Writing Checklist Status

```text
Mode: Standard (full section draft)
Target venue: AAAI (AI/ML family)
Idea scope: PRESERVED — no method, claim, or experiment changes made

1. [✓] Target venue explicit: AAAI, AI/ML family
2. [✓] Available materials: Table 1 baselines (11 papers), Round 1 literature search (12 papers),
         Round 2 literature search (14 new papers), positioning-update.md, experiment results
3. [✓] Global story defined: LIF triple-duty problem → functional decomposition → projective reset →
         empirical validation across 5 benchmarks
4. [✓] Section role: make novelty easy to verify; prepare reader for Method section
5. [✓] Major claims mapped:
         C1 (functional decomposition) → §2.1, §2.2, §2.4
         C2 (spike-never-resets-memory) → §2.3, §2.4
         C3 (spectral constraint) → §2.2, §2.3
         C4 (projective reset) → §2.4 (primary)
6. [✓] Venue-fit risks checked: AAAI expects strong baselines, clear novelty, honest positioning
7. [✓] Idea scope: preserved, no changes
8. [✓] Sibling modules: ccf-literature-search completed (Round 2); writing-only mode active
9. [N/A] Score-lifting loop: section-level only, not whole-paper
10. [✓] Remaining risks labeled below
```

## Score-Risk Diagnosis for Related Work Section

### Likely Reviewer Questions (AAAI AI/ML)

| # | Question | Risk Level | Mitigation in Draft |
|---|----------|------------|---------------------|
| R1 | "Isn't this just multi-timescale LIF? Local Timescale Gates / Bimodal already do slow+fast." | HIGH | §2.2 explicitly distinguishes rate decomposition vs functional decomposition |
| R2 | "D-RF (NeurIPS 2025) also extends RF with dendrites — what's new?" | HIGH | §2.3 explicitly contrasts dendritic frequency decomposition vs spectral functional decomposition |
| R3 | "The reset innovation section cites only arXiv preprints — are these established?" | MEDIUM | §2.4 balances preprints with peer-reviewed work (InfLoR-SNN/ECCV, RPLIF/ACM MM, PSN/NeurIPS) |
| R4 | "Where is the comparison with SiLIF?" | MEDIUM | SiLIF excluded per user request (SSM-related); user may want to add a brief mention |
| R5 | "28 references in Related Work is dense — is all this necessary?" | LOW | Each paragraph is focused; references serve positioning, not padding |
| R6 | "The section doesn't mention any limitations of SPRiF relative to prior work." | LOW | Acknowledged: each subsection fairly describes what prior work achieves before distinguishing |

### Fixability

| Risk | Fixable by Writing? | Fixable by New Experiment? | Accepted Limitation? |
|------|---------------------|---------------------------|---------------------|
| R1 | ✓ Already addressed | — | — |
| R2 | ✓ Already addressed | — | — |
| R3 | ✓ Already balanced | — | — |
| R4 | Partially — needs user decision on whether to mention SiLIF | — | User decision needed |
| R5 | Could compress if page limit is tight | — | — |
| R6 | Could add a limitations sentence per subsection | — | — |

### Remaining Risks

1. **SiLIF mention** (R4): User excluded SSM-related papers, but SiLIF is the closest competitor identified in Round 1. Consider adding 1-2 sentences in §2.3 acknowledging SSM-inspired spiking models as a parallel direction without deep engagement.
2. **Preprint-heavy §2.4**: Three of five reset innovation papers are arXiv preprints. This is unavoidable given the recency of the topic, but reviewers may note it.
3. **DA-LIF venue**: Listed as "IEEE 2025" — the exact venue (conference vs journal) should be verified before final submission.
4. **LT-Gate and Bimodal**: Both are very recent (2025-2026) and may not be familiar to reviewers. Brief descriptions are provided in the text.

## Files Produced

```text
experiment-design-20260606/
  related-work-draft.md        — Full Related Work section (~1500 words, 5 subsections)
  related-work-references.md   — 28 references with keys, venues, and links
  related-work-checklist.md    — This file (checklist + score-risk diagnosis)
```
