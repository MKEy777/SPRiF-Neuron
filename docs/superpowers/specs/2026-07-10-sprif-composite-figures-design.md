# SPRiF Composite Figures Design

## Scope

Redraw the existing trajectory, reset, impulse-response, frequency-response, and ASRNN-comparison analyses into two double-column manuscript figures. The figures must be regenerated from the existing NPZ/CSV sources rather than assembled from raster screenshots. The manuscript will replace its two current analysis figures with the new composites and revise the accompanying text without changing benchmark or ablation results.

## Figure 1: Functional Decomposition and Projective Reset

**Core conclusion:** SPRiF preserves a continuous slow memory trajectory across spikes while confining discontinuities to a fast state whose learned reset direction is diverse and largely independent of decay, frequency, and firing rate.

**Archetype:** asymmetric mixed-modality figure, 183 mm wide.

**Panel map:**

- **a — Real pSMNIST trajectory (hero):** slow coordinates and membrane trace aligned to a recorded spike.
- **b — Controlled phase trajectory:** oscillatory slow-state phase plane from the cue-delay probe experiment, with probe times marked.
- **c — Fast-state reset geometry:** representative learned directions `[1,lambda]` drawn from task-wise lambda quantiles. The controlled trajectory record contains no binary spikes or pre/post displacement, so it is not used as event-level reset evidence.
- **d — Learned reset directions:** compact task-wise violin/strip distributions for ECG, GSC, and pSMNIST, including sample sizes and the zero reference.
- **e — Independence summary:** a 3-by-3 heat map of Pearson correlations between lambda and firing rate, alpha, and omega.

**Review risks and controls:** The real trajectory is representative, so the controlled task provides an independent mechanistic view. Distribution and correlation panels use all neurons in the supplied CSV. Correlations are descriptive and will not be presented as significance tests or causal evidence.

## Figure 2: Learned Temporal Kernels

**Core conclusion:** SPRiF learns heterogeneous decay and oscillatory kernels across tasks, including long-memory responses that are not available in the matched trained ASRNN membrane kernels.

**Archetype:** quantitative grid with a compact comparison hero panel, 183 mm wide.

**Panel map:**

- **a — Representative impulse kernels:** four alpha-stratified neurons per task, showing the real coordinate and oscillatory coordinates over 100 steps.
- **b — Frequency-domain diversity:** normalized real-coordinate spectra for the same three tasks, shown as median with a 10–90% envelope across neurons.
- **c — Long-memory baseline contrast:** the slow-alpha GSC and ECG SPRiF kernels compared with the closest available trained ASRNN membrane kernel, with effective timescales stated in-panel.

**Review risks and controls:** Representative neurons are selected deterministically from alpha quantiles. Frequency summaries use all neurons. ASRNN comparison is limited to GSC and ECG because those are the tasks with trained ASRNN checkpoints in the supplied source data; the caption will state this scope.

## Visual and Export Contract

- Python/Matplotlib only.
- Final width 7.2 inches (approximately 183 mm); height no more than 3.8 inches per figure.
- White background, restrained blue/teal/orange palette, color-independent line styles, 6–7 pt text at final size, lowercase bold panel labels.
- Export each figure as editable SVG, PDF with TrueType text, and 600-dpi PNG.
- Preserve direct source traceability to the four existing result directories.
- Do not overwrite existing diagnostic figures.

## Manuscript Integration

Replace the two current standalone analysis figures in `AuthorKit27/SPRiF_AAAI2027.tex` with the composites. Revise the analysis prose and captions to reference every panel, report the lambda distribution and correlation ranges conservatively, and describe the ASRNN comparison only for the tasks represented by source data. Compile in a temporary build tree so the workspace PDF remains untouched.
