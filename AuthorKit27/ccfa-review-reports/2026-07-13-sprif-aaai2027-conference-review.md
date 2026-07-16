# Conference Review — SPRiF: Spectral Projective Reset Integrate-and-Fire Neuron

## 1. Report Metadata

- **Review date:** 2026-07-13
- **Target venue/year/track:** AAAI 2027, assumed Main Technical Track
- **Paper title:** SPRiF: Spectral Projective Reset Integrate-and-Fire Neuron
- **Input materials reviewed:** `SPRiF_AAAI2027.tex`; current 9-page compiled PDF; `SPRiF_AAAI2027_supp.tex`; current 7-page supplementary PDF; `sprif2027.bib`; `.bbl` and compilation logs; the bundled `ReproducibilityChecklist.tex`; targeted consistency checks of local task model/training scripts for parameter counting, dataset splits, checkpoint selection, and ablation reporting. This was not a full source-code audit.
- **Search basis:** Public-safe queries for AAAI-27 rules and closest public work on explicit slow-memory/fast-spiking pathways, SSM-based spiking neurons, two-compartment neurons, and reset decoupling. No private manuscript wording was placed in queries.
- **Report file:** `ccfa-review-reports/2026-07-13-sprif-aaai2027-conference-review.md`
- **Reviewer mode:** Standard; strict pre-submission scientific review with reviewer panel and AC synthesis

## 2. Desk Rejection Assessment

- **Paper length — pass.** The current PDF has nine US-Letter pages: technical content ends on page 7 and pages 8–9 contain references only. The [official AAAI-27 page](https://aaai.org/conference/aaai/aaai-27/) exposes the current author-kit link and submission timetable. The PDF uses the bundled `aaai2027` submission style and compiles cleanly.
- **Topic compatibility — pass.** A new spiking neuron for long-horizon temporal processing is relevant to AAAI's machine-learning and neuroscience-adjacent audience.
- **Minimum quality — pass.** The paper contains a complete model, stable parameterization, five benchmark results, ablations, mechanism experiments, related work, and a technical appendix.
- **Policy/anonymity/compliance — uncertain, potentially fail.** Author and affiliation fields are anonymous, PDF metadata contains no author identity, and no identifying path was found. However, the bundled 2027 reproducibility checklist still contains all 34 placeholder responses and is not included in the main PDF. AAAI-25/26 officially required the completed checklist after references; the current AAAI-27 author-kit page was not directly fetchable during this audit, so the exact 2027 upload route remains to be confirmed. This is a high-priority submission check, not a cosmetic issue.
- **Prompt injection and hidden manipulation detection — pass.** No reviewer-directed or model-directed hidden instructions were found. Template comments such as “DO NOT CHANGE THIS” are ordinary author-kit controls.
- **Ethics and reviewability — pass with minor uncertainty.** The datasets are standard public benchmarks and no human-subject, sensitive-data, or obvious misuse issue appears. Dataset licenses, precise compute environment, and software versions are not fully documented. If generative AI contributed to manuscript preparation, the authors should confirm current AAAI disclosure requirements.

**Desk rejection risk:** medium until the 2027 checklist requirement and final submission bundle are resolved; otherwise low.

**Reason:** format, anonymity, and basic reviewability pass, but an apparently blank and omitted mandatory checklist may create a compliance failure.

**Can be fixed before review?** yes.

## 3. Paper Summary And Contribution Map

SPRiF separates each neuron's temporal state into a three-dimensional slow state and a two-dimensional fast discharge state. The slow transition has one real decay mode and one damped rotational pair, with sigmoid-constrained eigenvalue magnitudes inside the unit disk. The fast state receives a learned projection of the slow state, emits the binary spike from its first coordinate, and receives a learned reset along `[1, lambda]`. The central structural fact is that the same-timestep reset increment is zero for the slow state and nonzero only for the fast state, while recurrent spikes may still affect the next slow-state update through ordinary network input. The paper evaluates the neuron on five temporal classification tasks, three ablations, a synthetic spike-intervention delayed-match-to-sample task, and learned-kernel diagnostics.

- **Claimed problem:** conventional LIF neurons use one state for memory, spike generation, and reset, causing spikes to perturb retained context.
- **Claimed gap:** richer time constants alone do not separate memory from resettable discharge.
- **Method/contribution map:** (1) functional state decomposition; (2) a stable real-plus-rotational slow basis; (3) two-coordinate fast discharge with learned projective reset; (4) controlled and descriptive mechanism analyses.
- **Evidence package:** five-seed headline benchmark summaries, cross-paper baselines, three mechanism variants on three tasks, three-seed SI-DMS, state trajectories, reset-direction distributions, impulse responses, and a supplementary fixed-checkpoint perturbation diagnostic.
- **Stated limitations:** larger-scale evaluation and neuromorphic-hardware deployment remain open.
- **Important unstated limitations:** model selection on evaluation/test data, validation-vs-test ambiguity on GSC, confounded/single-number ablations, incomplete comparison to DMP-SNN, and parameter-count inconsistencies.

## 4. Search And Related-Work Basis

- **Queries used:** AAAI-27 author kit/page limit/reproducibility checklist; spiking neuron separate memory and discharge; slow memory pathway plus fast spiking; structured state-space spiking neurons; two-compartment spiking memory; reset decoupling.
- **Sources searched:** official AAAI pages; Nature Machine Intelligence; arXiv; OpenReview; NeurIPS proceedings.
- **Closest works found:**
  - [Algorithm–hardware co-design of neuromorphic networks with dual memory pathways](https://www.nature.com/articles/s42256-026-01255-3) (Sun et al., Nature Machine Intelligence 2026; [preprint](https://arxiv.org/abs/2512.07602)). It uses an explicit, stable, low-dimensional slow state that modulates a fast spiking pathway. It reports 99.3% on S-MNIST and 97.3% on PS-MNIST and is the closest missing comparison.
  - [Long Short-term Memory with Two-Compartment Spiking Neuron](https://arxiv.org/abs/2307.07231) (LSTM-LIF). This is now cited and discussed in the current manuscript.
  - [Revisiting Reset Mechanisms in Spiking Neural Networks for Sequential Modeling](https://arxiv.org/abs/2504.17751). It explicitly decomposes a memory module and spiking module and is now cited, but it appears twice under two BibTeX keys.
  - [Structured State Space Model Dynamics and Parametrization for Spiking Neural Networks](https://arxiv.org/abs/2506.06374) (SiLIF/C-SiLIF). It links SSM structure and complex/oscillatory second-order neuron dynamics and is cited.
  - [LMUFormer](https://openreview.net/forum?id=oEF7qExD9F) (ICLR 2024). It is a broader architecture-level use of explicit Legendre memory with spiking modules; less direct than DMP-SNN but relevant to the memory-module lineage.
- **Unverified related-work risks:** exhaustive coverage of all 2026 spiking-neuron preprints was not attempted. The search focused on work capable of changing the novelty or benchmark interpretation.
- **Source-quality screening status:** primary/official paper pages and proceedings were used; no MDPI or low-confidence secondary source supports the novelty judgment.

## 5. Expected Review Outcome

- **Expected outcome:** weak reject / lean reject in the current form
- **Main accept signal:** the neuron is mathematically clean, interpretable, and easy to implement; the direct-reset separation statement is correct and the breadth of analyses is unusually good for a neuron-design paper.
- **Main reject signal:** the empirical protocol and numerical audit do not yet support the headline claims. The local scripts select checkpoints by repeatedly evaluating the test set for S-/PS-MNIST, QTDB, and SHD; GSC reports the best validation accuracy without a held-out test evaluation; the main parameter counts do not match the stated architectures instantiated from the current code; and the central SI-DMS control cannot isolate separation because the merged model is already at chance before intervention.
- **Confidence:** 4/5. The full paper, supplement, compiled PDFs, bibliography, and targeted implementation evidence were available. A complete rerun and full code audit were not performed.

## 6. Strengths And Weaknesses

### Strengths

- The update order and reset boundary are explicit. Equations 4 and 12–13 support the narrow claim that direct same-timestep reset does not edit the slow state.
- The slow transition is stable by construction, and the real/rotational basis gives an interpretable parameterization rather than an unconstrained recurrence.
- The manuscript now acknowledges recurrent feedback, cross-paper comparison limits, the merged variant's capacity confound, and the descriptive nature of the kernel comparison.
- The evaluation spans dense pixel sequences, physiological events, frame-based speech, and event-based speech.
- Figures 2–4 are visually legible, well aligned, and free from clipping. The mechanism and temporal-kernel figures communicate the intended diagnostics clearly.
- The paper avoids unsupported biological interpretations of the reset direction and avoids calling the cross-paper table a matched significance test.

### Weakness 1: Evaluation-set model selection invalidates the main benchmark estimates

- **Evidence basis:** Supplement line 101 states that the highest “evaluation accuracy” observed during training is retained. The current S-/PS-MNIST scripts construct the official MNIST test split, evaluate it every epoch, save the checkpoint with the highest test accuracy, and report that maximum. QTDB and SHD follow the same `best_test_acc` pattern. No independent validation set is described for these tasks.
- **Reviewer deduction:** the test data influence checkpoint selection. Five seeds and standard deviations do not repair this leakage; the reported maxima are optimistically biased. This matters especially for S-MNIST, where the claimed margin over the selected baseline is only 0.08 points.
- **Required fix:** introduce a training/validation/test protocol. Select epochs and hyperparameters on validation data, then evaluate each selected checkpoint once on the untouched test split. Rerun all five seeds and update every claim derived from Table 1.

### Weakness 2: GSC reports validation accuracy as if it were benchmark accuracy

- **Evidence basis:** the current GSC training script builds `train` and `valid` loaders only, selects `best_val_acc`, and never evaluates the `test` mode. The dataset implementation has a distinct `testing_list.txt`, so validation and test sets are not aliases.
- **Reviewer deduction:** the 94.55 entry is not directly comparable with published test-set results. The statement that SPRiF is “competitive” on GSC is unverified under the standard held-out test split.
- **Required fix:** select the checkpoint on validation, evaluate on the test list once per seed, and report the resulting test mean and variation.

### Weakness 3: The closest public method is omitted and changes both novelty and performance context

- **Evidence basis:** DMP-SNN was public as an arXiv preprint in December 2025 and published in Nature Machine Intelligence in 2026. It explicitly couples a stable low-dimensional slow memory pathway with fast spiking activity and reports stronger S-/PS-MNIST results than SPRiF's current table.
- **Reviewer deduction:** “functional state decomposition” is not defensible as a new principle without a direct distinction from DMP-SNN. “Highest among the listed methods” remains literally true only because a close and stronger current method is not listed; reviewers may view the selection as incomplete.
- **Required fix:** cite DMP-SNN, compare the structural unit of memory (layer-shared vs per-neuron), transition basis (LMU-like vs constrained real/rotational), reset treatment, parameter/state cost, recurrence, and matched benchmark protocol. Reframe novelty around SPRiF's specific neuron-level spectral/projective-reset construction and reset-intervention analysis.

### Weakness 4: SI-DMS does not causally isolate the claimed mechanism

- **Evidence basis:** SPRiF full and lambda=0 have clean accuracies above 92%, but the merged control is 51.7% before intervention; LIF and ASRNN are also near chance, and BRF starts at 70.4%. The merged variant also changes total state dimensionality.
- **Reviewer deduction:** low clean accuracy creates a floor effect. The experiment shows that the trained separated models retain accuracy under the chosen intervention, but it does not establish that separation rather than learnability, capacity, optimization, or clean-task competence causes the difference. There is no high-performing non-separated control whose memory is selectively damaged by reset.
- **Required fix:** add a capacity-matched merged/non-separated model that reaches comparable clean accuracy, or use a paired intervention that directly applies/removes reset to the slow path within the same trained architecture. Report condition-level uncertainty and a stress-by-model interaction analysis.

### Weakness 5: The ablation claim contradicts Table 2

- **Evidence basis:** Table 2's caption and the following paragraph say the merged variant has the largest decrease on all three datasets. On PS-MNIST, however, omega=0 drops 3.53 points while merged drops 2.38 points. The abstract and conclusion use a broader “largest observed” formulation that still invites the same incorrect interpretation.
- **Reviewer deduction:** a prominent mechanism conclusion is contradicted by the displayed numbers.
- **Required fix:** state that merged is largest on GSC and QTDB, while rotation removal is largest on PS-MNIST; avoid aggregating unlike, unstandardized drops into a universal ordering.

### Weakness 6: Ablations lack seeds, variation, and a clean capacity control

- **Evidence basis:** Table 2 reports one number per cell with no seed count or uncertainty. The local result file explicitly says standard deviation is not reported, and current ablation scripts default to a single seed. The merged model reduces five state coordinates to three.
- **Reviewer deduction:** 0.33–1.03 point reset-direction differences may be within run-to-run variation, while the large merged drop cannot be attributed to functional separation alone.
- **Required fix:** run matched seeds for all variants, report mean and standard deviation or confidence intervals, use paired comparisons where possible, and add a five-state capacity-matched merged control.

### Weakness 7: Reported parameter counts do not match the current implementation

- **Evidence basis:** instantiating the manuscript's recurrent `[64, 256]` S-/PS-MNIST architecture from the current model code yields 92,810 trainable parameters (0.093M), not 0.067M. The QTDB architecture yields 2,130 parameters (0.00213M), not 0.00177M. GSC (133,512) and SHD (51,268) are consistent only after coarse rounding. The supplement defines 13 trainable dynamics parameters per SPRiF neuron, so silently excluding them is incompatible with the phrase “end-to-end parameter count.”
- **Reviewer deduction:** the parameter-efficiency claims in the abstract/introduction/conclusion are not currently auditable and are numerically false under the obvious total-trainable-parameter convention.
- **Required fix:** define the counting convention, produce a script-generated table for every model, include all trainable weights and neuron parameters, and revise the efficiency claims. If baseline counts use a different convention, do not compare them as matched totals.

### Weakness 8: Reproducibility is incomplete despite a detailed appendix

- **Evidence basis:** compute is described only as “commodity GPU hardware with standard deep learning frameworks”; GPU/CPU, memory, OS, library versions, exact seed list, hyperparameter search ranges/selection, and final checkpoint protocol are absent. The official checklist file is blank and omitted. Code is promised only upon acceptance.
- **Reviewer deduction:** the equations are reproducible, but the reported numbers are not yet independently reproducible or auditable.
- **Required fix:** complete the official checklist, specify hardware/software, exact seeds and permutations, validation/test rules, model-selection rules, all final hyperparameters, and code/data availability.

### Weakness 9: Bibliographic duplication and minor positioning noise

- **Evidence basis:** `zhang2025reset` and `revisiting_reset` are two BibTeX records for the same arXiv paper, producing two consecutive 2025a/2025b references on page 9.
- **Reviewer deduction:** the duplicate is not scientifically fatal but weakens confidence in citation integrity and uses scarce reference space.
- **Required fix:** retain one canonical entry and cite it consistently.

## 7. Potentially Missing Related Work

### Work: Sun et al., “Algorithm–hardware co-design of neuromorphic networks with dual memory pathways,” Nature Machine Intelligence 2026

- **Status:** searched
- **Why relevant:** it is the closest public realization of a slow explicit state plus fast spiking pathway and evaluates overlapping benchmarks.
- **Overlap:** explicit low-dimensional stable memory, fast spiking activity, S-/PS-MNIST and SHD, state/parameter-efficiency motivation.
- **Needed comparison:** per-neuron vs layer-shared memory; real/rotational spectrum vs LMU-like basis; local projective reset vs ordinary fast LIF pathway; recurrent vs feedforward architecture; matched accuracy, trainable parameters, state memory, and operations.

### Work: Liu et al., “LMUFormer,” ICLR 2024

- **Status:** searched
- **Why relevant:** it uses explicit Legendre memory and spiking modules for efficient sequence processing and helps establish the architectural memory-module lineage.
- **Overlap:** structured slow state feeding a spiking computation path.
- **Needed comparison:** architecture-level LMU plus spiking blocks versus SPRiF's per-neuron five-state unit and projective reset.

### Work: Zhang, “Revisiting Reset Mechanisms in Spiking Neural Networks for Sequential Modeling,” 2025

- **Status:** user-provided and searched
- **Why relevant:** it explicitly frames SNNs as separable memory and spiking modules.
- **Overlap:** principle-level decomposition and reset analysis.
- **Needed comparison:** already discussed, but the duplicate bibliography entries must be merged and the novelty language should acknowledge that the decomposition principle predates SPRiF.

## 8. Claim-Evidence Audit

| Claim | Where stated | Evidence provided | Strength | Reviewer deduction | Required fix |
| --- | --- | --- | --- | --- | --- |
| Direct local reset does not update the slow state | Method, Eqs. 4 and 12–13; conclusion | Update equations and unrolled slow recurrence | strong | Correct when holding ordinary input fixed; recurrent spikes can affect the next step and this is disclosed | Keep the narrow “direct local” wording |
| SPRiF has the highest reported mean among listed methods on three tasks | Abstract, introduction, Table 1, conclusion | Cross-paper table | weak | Literally true for the selected list, but evaluation-set selection and omission of DMP-SNN undermine the comparison | Rerun proper test protocol and include/position DMP-SNN |
| End-to-end parameter count is lower than the strongest listed baseline on four tasks | Introduction and conclusion | Table 1 | weak/contradicted | Current code yields different totals for S-/PS-MNIST and QTDB | Script-generate and define all counts |
| Merged variant has the largest decrease on all three datasets | Table 2 caption and mechanism-ablation prose | Table 2 | contradicted | Omega=0 has the largest PS-MNIST drop | Correct caption, prose, abstract, and conclusion |
| Slow–fast separation protects memory under reset stress | SI-DMS section and conclusion | Full/lambda=0 remain high; merged and several baselines near chance | weak | Shows robustness of the separated models but does not isolate separation because the non-separated control fails clean | Add a high-clean-accuracy, capacity-matched causal control |
| Learned projective reset contributes performance | Lambda=0 ablation and lambda diagnostics | 0.33–1.03 point single-number drops; diverse lambda distributions | weak | Direction diversity is descriptive; performance differences lack uncertainty | Multi-seed paired ablation and uncertainty |
| Learned slow kernels are heterogeneous and long-lived | Figure 4 and dynamical analysis | Impulse responses and spectra | adequate | Descriptive claim is supported; two-task ASRNN comparison is appropriately scoped | Keep scope; report checkpoint/selection details |
| Functional state decomposition is a reusable new design principle | Contributions and conclusion | Specific SPRiF construction plus related-work discussion | weak | DMP-SNN and the cited memory/spiking decomposition anticipate the principle | Claim novelty for the specific spectral/projective-reset instantiation |

## 9. Experiment / Benchmark / Reproducibility Audit

- **Baselines:** broad but cross-paper and protocol-mismatched. DMP-SNN is a missing close and stronger comparator on overlapping tasks. Baseline uncertainty is absent.
- **Ablations:** three conceptually relevant variants, but merged changes capacity, no seed/variation is reported, and the interpretation contradicts PS-MNIST.
- **Datasets/benchmarks:** diverse and relevant. PS-MNIST generates a different permutation from each run seed, whereas many published protocols use one fixed permutation; this further complicates cross-paper comparison and should be stated prominently.
- **Metrics:** accuracy is standard. Peak-over-test accuracy is not a valid final-estimate protocol. GSC validation accuracy is not a test result.
- **Statistical rigor:** headline SPRiF results report five-seed standard deviations; no confidence intervals or tests are provided. Ablations have no variation. SI-DMS has three seeds and no formal interaction test.
- **Robustness/failure cases:** SI-DMS and the fixed-checkpoint feature perturbation diagnostic are useful, but the latter is not a multi-seed robustness ranking and SI-DMS controls have severe floor effects.
- **Implementation details:** equations and many task hyperparameters are present. Exact compute/software, seed list, hyperparameter-selection procedure, data-split policy, and checkpoint-selection policy are incomplete.
- **Artifacts and reproducibility:** local code exists, but the manuscript offers code only upon acceptance and the official checklist is blank. No code package was assessed as a submission artifact.
- **Limitations:** the stated limitations are too narrow. They should include cross-paper protocol mismatch, test/validation selection, synthetic SI-DMS, capacity-confounded ablation, lack of runtime/energy evaluation, and incomplete hardware evidence.

## 10. Multi-Reviewer Panel

### Reviewer: Best-Justified Reviewer

- **Expertise:** temporal SNN neuron design
- **Likely score:** 6/10
- **Confidence:** 4/5
- **Main positive signal:** elegant stable dynamics, correct direct-reset separation, broad and visually clear analysis package.
- **Main negative signal:** novelty must be narrowed to the specific construction and the benchmark protocol needs repair.
- **Score-change condition:** a proper validation/test rerun, reconciled parameter counts, and direct DMP-SNN positioning could make the paper borderline positive.

### Reviewer: Critical Reviewer

- **Expertise:** empirical ML methodology
- **Likely score:** 2/10
- **Confidence:** 4/5
- **Main positive signal:** the method itself is understandable and testable.
- **Main negative signal:** test-set checkpoint selection, validation/test mixing, inconsistent parameter counts, and a non-causal central mechanism experiment.
- **Fatal concern if any:** headline empirical evidence is not valid under standard held-out-test practice.
- **Score-change condition:** replace the affected estimates with untouched-test results and provide a causal, capacity-matched SI-DMS control.

### Reviewer: Method / Soundness Reviewer

- **Expertise:** dynamical systems and spiking neurons
- **Likely score:** 6/10
- **Confidence:** 4/5
- **Main positive signal:** spectral radius control, update order, and direct reset identity are sound.
- **Main negative signal:** the paper proves a structural identity but sometimes extrapolates it into a broader functional-memory conclusion not isolated experimentally.
- **Score-change condition:** constrain claims to the proved identity or add evidence that isolates the functional effect.

### Reviewer: Evidence / Experiment Reviewer

- **Expertise:** benchmark design and statistical evaluation
- **Likely score:** 3/10
- **Confidence:** 5/5
- **Main positive signal:** five tasks, mechanism variants, and an intervention task are more extensive than typical neuron papers.
- **Main negative signal:** evaluation leakage, split mismatch, absent ablation uncertainty, floor effects, and unmatched comparisons.
- **Score-change condition:** rerun evaluation and ablations under a preregistered validation/test protocol with matched seeds.

### Reviewer: Novelty / Positioning Reviewer

- **Expertise:** SNNs and state-space sequence models
- **Likely score:** 4/10
- **Confidence:** 4/5
- **Main positive signal:** the exact real-plus-rotational per-neuron slow state combined with learned two-coordinate reset is distinctive.
- **Main negative signal:** the principle-level claim overlaps strongly with DMP-SNN and the already cited memory/spiking decomposition.
- **Score-change condition:** direct comparison and narrower novelty claims.

### Reviewer: Writing / Clarity Reviewer

- **Expertise:** scientific presentation
- **Likely score:** 6/10
- **Confidence:** 4/5
- **Main positive signal:** the architecture, equations, figures, and scope qualifiers are generally clear.
- **Main negative signal:** Table 2's caption/prose contradict the numbers, and the parameter-count convention is unstated.
- **Score-change condition:** correct all claim-table inconsistencies and define evaluation/counting conventions.

### Reviewer: Ethics / Reproducibility Reviewer

- **Expertise:** reproducible ML and research compliance
- **Likely score:** 4/10
- **Confidence:** 4/5
- **Main positive signal:** public datasets, anonymous source, no obvious ethics hazard, detailed mathematical appendix.
- **Main negative signal:** blank/omitted checklist, vague compute environment, no exact seed or software information, and no untouched-test protocol.
- **Score-change condition:** complete the official checklist and reproducibility contract.

### Reviewer: AC / Meta-Reviewer

- **Expertise:** broad AAAI machine learning
- **Likely score:** 4/10
- **Confidence:** 4/5
- **Main positive signal:** potentially useful neuron design with clean structural intuition.
- **Main negative signal:** multiple independent empirical-integrity concerns are aligned rather than isolated.
- **Score-change condition:** evidence repair must precede wording polish; a citation-only or rebuttal-only response would not resolve the current risk.

## 11. Concerns Table

| ID | Severity | Concern | Evidence basis | Affected criterion | Fix class | Required action | Owner skill | Score-change condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | fatal | Test/evaluation data used for checkpoint selection | Supplement reporting rule plus local S-/PS-MNIST, QTDB, SHD scripts | soundness, evidence | experiment | Validation-select, one-shot test evaluation, rerun five seeds | `ccf-experiment-designer` | Required before score can exceed reject |
| C2 | fatal | GSC value is best validation accuracy, not held-out test accuracy | Local GSC train/data split code | soundness, evidence | experiment | Evaluate validation-selected models on `test` split | `ccf-experiment-designer` | Required before GSC claim is reviewable |
| C3 | major | DMP-SNN omitted from novelty and benchmark context | 2025 preprint/2026 NMI paper; overlapping tasks and design | originality, positioning | related-work | Add direct structural and empirical comparison | `ccf-literature-search` | Originality may rise from 2 to 3–4 if delta is clear |
| C4 | major | SI-DMS cannot isolate separation | Merged/LIF/ASRNN at chance before stress; merged changes capacity | evidence, soundness | experiment | High-performing capacity-matched non-separated control or paired reset-path intervention | `ccf-experiment-designer` | Central mechanism claim becomes defensible |
| C5 | major | Table 2 interpretation contradicts PS-MNIST | 3.53-point omega=0 drop vs 2.38-point merged drop | quality, clarity | writing | Correct caption, prose, abstract, conclusion | `ccf-writing-skills` | Removes direct internal contradiction |
| C6 | major | Ablations have no seeds/variation and are capacity-confounded | Single-number table; local result metadata/scripts | evidence | experiment | Matched multi-seed runs and five-state merged control | `ccf-experiment-designer` | Needed to support component claims |
| C7 | major | Parameter counts are inconsistent with code | 0.093M vs 0.067M on S-/PS-MNIST; 0.00213M vs 0.00177M on QTDB | evidence, reproducibility | reproducibility | Script-generate totals and define convention | `ccf-integrity-auditor` | Efficiency claims remain unsupported until reconciled |
| C8 | major | Reproducibility checklist blank and absent | 34 placeholders; no checklist in main PDF | compliance, reproducibility | reproducibility | Confirm AAAI-27 rule, complete and attach/upload correctly | `ccf-submission-checker` | Removes desk/compliance risk |
| C9 | moderate | Compute, software, seeds, and model-selection details incomplete | Supplement compute/reporting sections | reproducibility | reproducibility | Add exact hardware/software and seed protocol | `ccf-writing-skills` | Improves auditability |
| C10 | moderate | PS-MNIST permutation changes with run seed | Local training script and supplement | comparability | experiment | Fix one permutation across seeds or justify and report it | `ccf-experiment-designer` | Makes variance and baseline comparison interpretable |
| C11 | minor | Same reset paper appears twice | Two BibTeX keys and duplicate references on page 9 | clarity, integrity | writing | Merge entries and citations | `ccf-integrity-auditor` | No score change alone |
| C12 | moderate | No runtime/state-memory/energy comparison | Five states and 13 dynamics parameters per neuron; parameter-only claims | significance | experiment | Report state memory, throughput/latency, operations, and spike rate or narrow efficiency claims | `ccf-experiment-designer` | Strengthens practical relevance |

## 12. AC / Meta-Review

- **Reviewer consensus:** the model is intelligible, stable, and potentially useful. The structural “no direct slow-state reset” identity is sound. The current empirical record is not sufficient for acceptance.
- **Reviewer disagreement:** a method-focused reviewer may see an incremental but elegant neuron and score near borderline; an empirical reviewer is likely to reject strongly because the test/validation protocol compromises the headline numbers.
- **Decisive acceptance axis:** can the authors provide untouched-test, multi-seed results under a clearly defined protocol, reconcile parameter counts, and distinguish SPRiF from DMP-SNN?
- **Decisive rejection axis:** if Table 1 remains peak-over-test/validation-selected and SI-DMS remains without a competent non-separated control, the central performance and mechanism claims are not decision-grade.
- **AC stance:** weak reject / lean reject, overall 4/10.
- **Discussion risks:** the paper could be phase-1 rejected before reviewers engage deeply with the mechanism because C1/C2/C7 are easy-to-state empirical-integrity concerns. Novelty discussion will likely focus on DMP-SNN and the already cited memory/spiking decomposition. The Table 2 contradiction may further reduce confidence in the manuscript's quantitative audit.

## 13. Scores

- **Quality:** 2/5
- **Clarity:** 4/5
- **Significance:** 3/5
- **Originality:** 2/5
- **Soundness:** 2/5
- **Evidence:** 2/5
- **Reproducibility:** 2/5
- **Ethics / Limitations:** 3/5
- **Overall:** 4/10 — weak reject / lean reject
- **Confidence:** 4/5

The score is a conditional review-risk diagnosis, not an acceptance probability. It is dominated by the empirical protocol and numerical inconsistencies, not by a claim that the neuron equations are wrong.

## 14. Questions For Authors

1. For S-MNIST, PS-MNIST, QTDB, and SHD, was any split independent of the test set used to select epochs and hyperparameters? If not, can all headline results be regenerated with validation-selected checkpoints and one untouched test evaluation?
2. Is the GSC value 94.55% a validation-set or testing-list result? Where is the one-shot test evaluation for each seed?
3. What exact counting convention produces 0.067M for the recurrent `[64,256]` model and 0.00177M for QTDB? Does it include all 13 trainable neuron-wise dynamics parameters?
4. How many independent seeds produced each Table 2 cell, and what is the run-to-run variance?
5. Can a capacity-matched non-separated model reach clean SI-DMS accuracy comparable to full SPRiF? If not, what evidence isolates reset-memory separation from optimization/capacity?
6. How does SPRiF differ from DMP-SNN under matched concepts and overlapping S-/PS-MNIST benchmarks?
7. Will the completed AAAI-27 reproducibility checklist be included in the required submission artifact, and what are the exact hardware/software/seed details?

## 15. Score Revision Criteria

**Raising the score would require:**

- validation-selected, untouched-test, five-seed results for all headline benchmarks;
- a genuine GSC test-set evaluation;
- corrected and auditable parameter counts;
- direct DMP-SNN positioning and appropriately narrowed novelty claims;
- multi-seed ablations and a capacity-matched/non-floor SI-DMS control;
- completion of the AAAI-27 reproducibility checklist and submission audit;
- correction of the Table 2 contradiction and duplicate citation.

If C1–C8 are resolved with consistent results, a borderline-positive score around 6/10 becomes plausible. This is conditional, not promised.

**Lowering the score would be triggered by:**

- confirmation that the reported benchmark values cannot be reproduced without test-set checkpoint selection;
- inability to reconcile parameter counts or source/result provenance;
- a closer matched analysis showing that SPRiF's specific construction adds no meaningful delta over DMP-SNN/SSM-based spiking neurons;
- evidence that the ablation cells are isolated best runs selected under different tuning budgets.

**Concerns unlikely to change before submission:**

- full neuromorphic hardware validation and broad runtime/energy characterization may be infeasible before the deadline;
- the core principle-level novelty cannot be restored, only accurately repositioned around the specific SPRiF construction;
- SI-DMS remains a synthetic diagnostic even after controls improve.

## 16. Action Plan And CCFA Handoffs

### Action 1

- **Priority:** P0
- **Action:** Freeze a valid data-split and checkpoint-selection protocol, then regenerate all headline results.
- **Owner skill:** `ccf-experiment-designer`
- **Input needed:** task scripts, raw split definitions, existing checkpoints, compute budget
- **Expected output:** validation/test protocol, rerun matrix, seed plan, corrected Table 1
- **Handoff required:** yes

### Action 2

- **Priority:** P0
- **Action:** Run a numeric integrity audit of results, parameter counts, seeds, captions, and bibliography.
- **Owner skill:** `ccf-integrity-auditor`
- **Input needed:** model constructors, checkpoints/logs, result files, BibTeX
- **Expected output:** reconciled count table and corrected claim ledger
- **Handoff required:** yes

### Action 3

- **Priority:** P0
- **Action:** Add and position DMP-SNN; narrow the novelty statement to SPRiF's spectral/projective-reset instantiation.
- **Owner skill:** `ccf-literature-search` followed by `ccf-writing-skills`
- **Input needed:** current related work and contribution text
- **Expected output:** closest-work comparison paragraph/table and revised contribution claims
- **Handoff required:** yes

### Action 4

- **Priority:** P1
- **Action:** Replace the floor-confounded SI-DMS comparison with a capacity-matched, high-clean-accuracy causal control and rerun ablations with matched seeds.
- **Owner skill:** `ccf-experiment-designer`
- **Input needed:** SI-DMS code, model variants, compute budget
- **Expected output:** matched-control results with uncertainty and interaction analysis
- **Handoff required:** yes

### Action 5

- **Priority:** P0
- **Action:** Complete the 2027 checklist and perform final AAAI compliance/build inspection.
- **Owner skill:** `ccf-submission-checker`
- **Input needed:** final PDF, supplement, checklist answers, submission instructions
- **Expected output:** submission-ready compliance report and clean artifacts
- **Handoff required:** yes

### Action 6

- **Priority:** P1
- **Action:** Revise abstract, Table 2 caption, ablation prose, conclusion, limitations, and reproducibility text after numbers are frozen.
- **Owner skill:** `ccf-writing-skills`
- **Input needed:** corrected results and integrity ledger
- **Expected output:** claim-evidence-aligned manuscript text
- **Handoff required:** yes

**Checks run:** venue/track assumption; four-pass manuscript read; current PDF visual inspection; page distribution; anonymity and PDF metadata; compilation warnings; prompt-injection scan; contribution map; public-safe closest-work search; claim-evidence audit; benchmark/ablation/SI-DMS/statistics audit; targeted code consistency checks for splits, checkpointing, seeds, and parameter counts; multi-reviewer panel; AC synthesis; calibrated scoring; concerns table.

**Checks skipped:** full source-code review; independent re-execution of training; statistical recomputation from per-seed raw outputs; exhaustive 2026 literature review; exact AAAI-27 checklist-upload policy verification because the official author-kit endpoint was inaccessible to the browser.

**Unresolved risks:** true untouched-test performance; exact provenance of every reported cell; whether Table 2 contains additional unpublished seeds; the final parameter-count convention; whether the official checklist will be included; whether DMP-SNN can be matched under identical architecture/protocol.
