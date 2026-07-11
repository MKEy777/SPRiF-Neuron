# AAAI SPRiF Supplement Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an anonymous, standalone AAAI-27 technical-appendix TeX skeleton and a main-paper/supplement content allocation plan without adding unverified content.

**Architecture:** Mirror the approved AAAI-27 preamble and bibliography setup from the main manuscript. Keep the appendix independently compilable, use `supp:` label prefixes, and place all writing guidance in LaTeX comments so the delivered PDF contains no fabricated results or placeholder claims.

**Tech Stack:** LaTeX, AAAI 2027 author kit, BibTeX, Markdown.

## Global Constraints

- Main paper remains self-contained.
- Submission appendix remains anonymous.
- No external mutable supplementary links.
- No numerical results or claims are inserted.
- Final AAAI-27 supplementary limits remain subject to the official submission portal policy.

### Task 1: Create standalone technical appendix

**Files:**
- Create: `AuthorKit27/SPRiF_AAAI2027_supp.tex`

- [x] Copy only approved packages from the main manuscript.
- [x] Add anonymous title metadata and appendix section skeleton.
- [x] Prefix all labels with `supp:`.
- [x] Leave bibliography and figure/table examples as comments.

### Task 2: Create content allocation plan

**Files:**
- Create: `AuthorKit27/MAIN_SUPP_CONTENT_PLAN.md`

- [x] Map every evidence item to main, supplement, or both.
- [x] Mark SI-DMS core evidence as mandatory in the main paper.
- [x] Record current manuscript sentences that must be updated after results exist.

### Task 3: Verify

- [x] Compile the supplement twice with `pdflatex`.
- [x] Confirm no undefined control sequences or missing files.
- [x] Scan for author identity, machine paths, and unfilled numerical claims.
