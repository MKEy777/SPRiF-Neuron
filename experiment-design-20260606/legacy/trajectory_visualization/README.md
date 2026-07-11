# Archived trajectory visualization

This directory preserves the former `trajectory_visualization` artifacts for provenance only.

## Evidence warning

These files must not be used as evidence of an actual pre/post-reset transition unless the recorded trajectory contains a real spike at the claimed reset time and a nonzero state change caused by that reset. Records satisfying `spike.sum() == 0` or `u_pre - u_post == 0` cannot support reset arrows.

The active mechanism experiment replacing this package is SI-DMS, documented in `../../si-dms-experiment-plan.md`.
