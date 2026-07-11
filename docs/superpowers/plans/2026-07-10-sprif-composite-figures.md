# SPRiF Composite Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce two dense, source-traceable manuscript figures and integrate them into the AAAI LaTeX paper without increasing the current eight-page draft.

**Architecture:** A new offline plotting module reads the existing NPZ and CSV artifacts from the four diagnostic result directories, computes deterministic summary views, and exports two composite figures. Existing analysis scripts and diagnostic figures remain untouched. The paper references only the new composites while its prose records the scope and limits of each panel.

**Tech Stack:** Python 3.11, NumPy, pandas, Matplotlib, pytest, LaTeX/latexmk.

## Global Constraints

- Use Python/Matplotlib exclusively for drawing, previewing, exporting, and figure QA.
- Generate figures from NPZ/CSV source data; do not collage raster images.
- Export SVG, PDF, and 600-dpi PNG at 7.2 inches wide and no more than 3.8 inches high.
- Do not overwrite the existing diagnostic figures or the workspace PDF.
- Preserve all benchmark and ablation numbers already present in the manuscript.

---

### Task 1: Define and Test the Offline Plotting Interface

**Files:**
- Create: `代码/experiments/combined_analysis/test_plot_combined_figures.py`
- Create: `代码/experiments/combined_analysis/plot_combined_figures.py`

**Interfaces:**
- Consumes: repository root and the existing trajectory/reset/impulse NPZ and CSV files.
- Produces: `load_source_data(repo_root: Path) -> dict`, `build_mechanism_figure(data: dict) -> Figure`, `build_temporal_figure(data: dict) -> Figure`, and `export_figure(fig, output_base: Path) -> list[Path]`.

- [ ] **Step 1: Write failing tests** that import the four functions, validate required source shapes, require panel labels `a`–`e` and `a`–`c`, and require SVG/PDF/PNG export.
- [ ] **Step 2: Run `pytest 代码/experiments/combined_analysis/test_plot_combined_figures.py -v`** and verify collection fails because the plotting module does not yet exist.
- [ ] **Step 3: Implement source loading and validation** with explicit missing-key errors for trajectory, controlled trajectory, reset statistics, raw impulse responses, and ASRNN comparison data.
- [ ] **Step 4: Implement the two figure builders** according to the approved panel maps, using deterministic alpha/lambda-quantile selection and all-neuron summaries where specified. Render reset geometry from learned `[1,lambda]` directions because the controlled trajectory NPZ contains no spike-triggered pre/post displacement.
- [ ] **Step 5: Implement vector/raster export** with editable SVG text, PDF font type 42, and 600-dpi PNG.
- [ ] **Step 6: Re-run the focused pytest file** and verify all tests pass.

### Task 2: Generate and Visually Audit the Composite Figures

**Files:**
- Create: `experiment-design-20260606/results/figures/combined_analysis/mechanism_composite.svg`
- Create: `experiment-design-20260606/results/figures/combined_analysis/mechanism_composite.pdf`
- Create: `experiment-design-20260606/results/figures/combined_analysis/mechanism_composite.png`
- Create: `experiment-design-20260606/results/figures/combined_analysis/temporal_kernels_composite.svg`
- Create: `experiment-design-20260606/results/figures/combined_analysis/temporal_kernels_composite.pdf`
- Create: `experiment-design-20260606/results/figures/combined_analysis/temporal_kernels_composite.png`

**Interfaces:**
- Consumes: the tested plotting module and existing result artifacts.
- Produces: six publication figure files plus console-reported source coverage and dimensions.

- [ ] **Step 1: Run the plotting module** from the repository root and verify all six files are created.
- [ ] **Step 2: Verify dimensions and format metadata** with Python/Pillow and inspect the SVG for editable text elements.
- [ ] **Step 3: Open both PNGs at original resolution** and check panel labels, axes, legends, clipping, color consistency, and readability at double-column size.
- [ ] **Step 4: Adjust only presentation parameters** when QA reveals crowding; retain data selection and panel semantics.
- [ ] **Step 5: Re-run pytest and figure generation** after any adjustment.

### Task 3: Integrate the Figures into the Manuscript

**Files:**
- Modify: `AuthorKit27/SPRiF_AAAI2027.tex`

**Interfaces:**
- Consumes: the two composite PDF/SVG/PNG outputs.
- Produces: revised analysis prose, two figure environments, captions, and cross-references.

- [ ] **Step 1: Replace the standalone trajectory and impulse figures** with `mechanism_composite.pdf` and `temporal_kernels_composite.pdf`.
- [ ] **Step 2: Rewrite the dynamical-analysis paragraphs** so every panel is narrated exactly once and claims remain bounded by the supplied data.
- [ ] **Step 3: Report reset statistics conservatively:** both signs occur for all tasks; absolute correlations with firing rate, alpha, and omega are at most 0.191.
- [ ] **Step 4: State the ASRNN comparison scope:** trained baseline kernels are available for GSC and ECG only.

### Task 4: Compile and Verify the Final Draft

**Files:**
- Verify: `AuthorKit27/SPRiF_AAAI2027.tex`
- Verify: temporary LaTeX build outputs outside the workspace.

**Interfaces:**
- Consumes: revised TeX, bibliography, and composite figures.
- Produces: a temporary rendered PDF and verification report.

- [ ] **Step 1: Copy manuscript dependencies to a temporary build directory** and run `latexmk -g -pdf -interaction=nonstopmode -halt-on-error SPRiF_AAAI2027.tex`.
- [ ] **Step 2: Require build exit code 0**, no undefined references, no overfull boxes, and no placeholder text.
- [ ] **Step 3: Confirm the draft remains eight pages**; if it grows, compress captions and analysis prose before reducing figure readability.
- [ ] **Step 4: Render all PDF pages to PNG** and inspect the pages containing both composites plus the following page for float collisions or clipped text.
- [ ] **Step 5: Run the focused pytest file and a final source-coverage check** proving all four named result directories contribute data to the two manuscript figures.
