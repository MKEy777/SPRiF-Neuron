# Ledger (rendered view -- do not edit; source of truth is the .json)

Manuscript: C:\Users\VECTOR\Desktop\SPRiF-Neuron\AuthorKit27\SPRiF_AAAI2027.tex | venue: ml

Active: 6 (major 6, minor 0; author-required 6). Completion gate (0 gate-blocking active major): PASS (gate-blocking majors: 0).

| id | sig | kind | status | section | summary | close_criterion | by | rounds |
|----|-----|------|--------|---------|---------|-----------------|----|--------|
| I-01 | major | substantive | author-required (needs-human-input) | Neuron Dynamics for Temporal Processing | The paper does not establish the incremental value of SPRiF against the closest memory/discharge or memory/reset decoupling methods in a controlled comparison. |  | R1 | 1 |
| I-04 | major | substantive | author-required (needs-human-input) | Mechanism Ablations | The central merged-state ablation confounds functional separation with a reduction from five to three state dimensions. |  | R1,R2,R3 | 1 |
| I-05 | major | substantive | author-required (needs-human-input) | Mechanism Ablations | The ablation table reports point estimates without seed counts or uncertainty, leaving small reset-direction effects statistically unresolved. |  | R1,R3 | 1 |
| I-06 | major | substantive | author-required (needs-human-input) | Controlled Spike Intervention | SI-DMS overwrites functionally different coordinates across models and therefore does not isolate reset from intervention-induced state overwrite. |  | R2,R3 | 1 |
| I-07 | major | substantive | author-required (needs-human-input) | Controlled Spike Intervention | SI-DMS lacks a high-performing coupled-state control, so reset-stress robustness is confounded with clean-task learnability. |  | R1,R2,R3 | 1 |
| I-08 | major | substantive | author-required (needs-human-input) | Controlled Spike Intervention | The dedicated stress result does not show a benefit from learned projective reset, while benchmark effects are small and lack uncertainty. |  | R1,R3 | 1 |
| I-02 | major | substantive | dropped (invalid-drop) | Introduction | The introductory contrast may overgeneralize multi-timescale models and create a misleading novelty premise. |  | R1 | 1 |
| I-03 | major | substantive | dropped (invalid-drop) | Main Results | The unmatched cross-paper table cannot establish controlled superiority or efficiency. |  | R1,R3 | 1 |
| I-09 | minor | substantive | queued (polish-review) | Introduction | Narrow the introductory contrast to many single-state or parallel-rate models so it matches the exceptions acknowledged in Related Work. |  | R1 | 1 |
| I-10 | minor | substantive | queued (polish-review) | Fast Discharge Dynamics and Projective Reset | The manuscript calls the discharge state fast without stating any learned timescale ordering relative to the slow spectral factors. |  | R2 | 1 |
| I-11 | minor | substantive | queued (polish-review) | Spectral Slow-Memory Dynamics | This sentence is broader than the same-timestep direct-local-reset guarantee because recurrent spikes can affect the next slow update. |  | R1,R2 | 1 |
| I-12 | minor | substantive | queued (polish-review) | Dynamical Analysis | The impulse response characterizes the isolated slow transition rather than the effective input-to-spike kernel of the full neuron. |  | R1,R2 | 1 |
| I-13 | minor | substantive | queued (polish-review) | Dynamical Analysis | The time-constant comparison is selection- and checkpoint-limited and is not a like-for-like population comparison. |  | R2,R3 | 1 |
| I-14 | minor | mechanical | queued (polish-review) | Mechanism Ablations | Explain why PS-MNIST full accuracy is 95.82 in the ablation table but 95.86 plus/minus 0.14 in the main table. |  | R3 | 1 |
| I-15 | minor | substantive | queued (polish-review) | Introduction | Parameter-count statements need an explicit reminder that cross-paper architecture and counting conventions are unmatched. |  | R1,R3 | 1 |
| I-16 | minor | substantive | queued (polish-review) | Controlled Spike Intervention | The SI-DMS table omits dispersion for the reported three-seed averages, although the figure caption states that curves show seed standard deviations. |  | R3 | 1 |
