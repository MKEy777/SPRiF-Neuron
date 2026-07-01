# Introduction Review — SPRiF Paper (AAAI)

> Reviewer role: skeptical AAAI AI/ML reviewer + internal quality pass
> Date: 2026-07-02
> Reviewed text: introduction-draft.md (7 paragraphs, ~1200 words)

---

## A. Issue Registry

### 🔴 Critical (Must Fix)

| # | Paragraph | Issue | Current Text | Fix |
|---|-----------|-------|-------------|-----|
| A1 | Claim-Evidence Map | Map says "4/5 benchmarks best" but text says 3/5 | `4/5 benchmarks best` | Change to `3/5 benchmarks best; 2/5 competitive (#2)` |
| A2 | P6 | "independently" overclaims ablation evidence | "each design element contributes independently" | Ablation B removes the entire slow/fast separation (not a single element). Change to "each design element contributes materially" |
| A3 | P2→P3 | "AdLIF" naming inconsistency with Related Work | P3 uses "AdLIF \cite{adlif}" but cite key maps to SE-adLIF | Standardize: use "SE-adLIF \cite{adlif}" everywhere, matching Related Work §2.2 |

### 🟡 Important (Should Fix)

| # | Paragraph | Issue | Detail |
|---|-----------|-------|--------|
| A4 | P2 | Paragraph is too long (~280 words, 14 citations) | Dense for Introduction; hard to parse in one pass. Consider splitting or pruning citations. |
| A5 | P3 | "long-range temporal processing" overclaim | Only PS-MNIST is truly long-range (784 timesteps). Other benchmarks are 101–400 timesteps. Should say "temporal processing" without "long-range" qualifier, or specify "including long-range" |
| A6 | P4 | Missing training compatibility mention | Reviewers will ask "how is this trained?" No mention of BPTT + surrogate gradient. One sentence would preempt "is this even trainable?" |
| A7 | P6 | Parameter efficiency story undersold | SPRiF uses **2.2× fewer params** than the top performer on S-MNIST (0.067M vs 0.15M), PS-MNIST (0.067M vs 0.15M), and SHD (0.05M vs 0.11M). Even on GSC where SPRiF is #2, it uses 0.13M vs TC-LIF's 0.20M. "Comparable or fewer" understates this significantly. |
| A8 | P6 | GSC and SHD ranking precision | Text says "94.55% vs. 94.84% best" — correct but SPRiF is actually **2nd** on both (not 3rd or lower). Could strengthen framing slightly. |
| A9 | P5 | Paragraph P5 repeats P4 content | "slow state exists solely to store… fast state exists solely to translate…" echoes P4. The differentiation value is high but needs tighter prose. |

### 🟢 Minor (Nice to Fix)

| # | Paragraph | Issue | Suggestion |
|---|-----------|-------|-----------|
| A10 | P1 | "sequential decision-making" | No benchmarks test this. Remove or replace with "event-based recognition" |
| A11 | P4 | "Crucially" is editorial | Replace with direct statement or "By design" |
| A12 | P7 | No code release mention | Add "Code will be released upon acceptance" to contribution (4) |
| A13 | P6 | Ablation A range "up to 3.5 points on long-range tasks" | PS-MNIST is long-range but QTDB/GSC deltas are only 0.65–0.72. More honest to say "up to 3.5 points (task-dependent)" |
| A14 | — | No computational cost mention | SPRiF has ~3–5× LIF per-neuron cost. Could acknowledge briefly or leave for Method section. |

---

## B. Dimension Scores

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Logical flow** (P1→P7) | 9/10 | Excellent arc: task→landscape→gap→insight→method→differentiation→evidence→contributions. P4→P5 transition is slightly repetitive. |
| **Factual accuracy** | 7/10 | Claim-evidence map error (A1), GSC/SHD ranking undersold, ablation delta ranges need tightening. Core numbers are correct. |
| **Positioning / novelty framing** | 9/10 | "The state that triggers a spike is the state that gets reset" is sticky and defensible. Functional vs rate decomposition is a strong differentiation axis. |
| **Language quality** | 8.5/10 | Strong academic English. A few editorial words ("Crucially"). P2 is dense but well-structured. |
| **Claim-evidence alignment** | 8/10 | Most claims backed by filled tables. "Independently" overclaims. "Long-range" in P3 oversells. |
| **Cross-section consistency** | 8/10 | Naming inconsistency (AdLIF vs SE-adLIF). Otherwise aligned with Related Work terminology. |
| **Reviewer-risk mitigation** | 8.5/10 | P5 proactively addresses R1 (just more params) and R3 (multi-timescale). R2 (oscillation) partially addressed. |
| **AAAI venue fit** | 9/10 | Strong baselines, clear novelty, honest evidence reporting. Fits AAAI AI/ML well. |

**Overall: 8.4/10** — Strong draft with a few fixable factual/precision issues. No structural rewrite needed.

---

## C. Specific Sentence-Level Comments

### P2, line: "Despite this diversity, all existing spiking neurons share a structural assumption"
✅ **Excellent.** This is the rhetorical pivot of the entire Introduction. Memorable and defensible.

### P3, line: "it does not logically require that all accumulated temporal memory be simultaneously weakened or erased"
✅ **Strong argumentation.** Frames the gap as a design choice, not a limitation of prior work.

### P4, line: "This design realizes what we term the *spike-never-resets-memory* principle"
✅ **Good branding.** But "principle" is a strong word — make sure the Method section delivers on this promise clearly. (It does.)

### P5, line: "Prior work assigns different decay rates to parallel state components"
⚠️ **Slight overgeneralization.** HetSyn assigns decay rates to synapses, not neuron states. MTC uses conductance channels. The statement is directionally correct but a careful reviewer of HetSyn might object. Consider "Prior multi-timescale designs assign different decay rates or time constants to parallel processing channels."

### P6, line: "consistently matching or outperforming 11 established baselines"
⚠️ **"Consistently" is generous** when SPRiF is #2 on 2/5 datasets by 0.18–0.29 points. Acceptable but slightly promotional. "Matching or outperforming" alone would be sufficient.

---

## D. Recommended Revision Actions

| Priority | Action | Paragraphs affected | Status |
|----------|--------|-------------------|--------|
| **P0** | Fix Claim-Evidence Map "4/5" → "3/5 best; 2/5 #2" | Map | ✅ Fixed |
| **P0** | Fix "independently" → "materially" | P6 | ✅ Fixed |
| **P0** | Fix "AdLIF" → "SE-adLIF" | P3 | ✅ Fixed |
| **P1** | Strengthen parameter efficiency framing (0.067M vs 0.15M) | P6 | ✅ Fixed |
| **P1** | Add training compatibility sentence (BPTT + surrogate) | P4 | ✅ Fixed |
| **P1** | Remove "long-range" from P3 | P3 | ✅ Fixed |
| **P1** | Tighten P5 prose (cut repetition, fix HetSyn overgeneralization) | P5 | ✅ Fixed |
| **P2** | Remove "sequential decision-making" → "event-based recognition" | P1 | ✅ Fixed |
| **P2** | Replace "Crucially" → "By design" | P4 | ✅ Fixed |
| **P2** | Add code release mention | P7 | ✅ Fixed |
| **P2** | Tighten ablation A range ("up to 3.5 points") | P6 | ✅ Fixed |
