# SI-DMS results

This is the active result location replacing the former trajectory-visualization evidence package.

Expected formal-run artifacts:

- `all_metrics.json`: all model, seed, delay, and intervention-count rows.
- `summary.csv`: clean accuracy, maximum-stress accuracy, and stress drop.
- `result_template.json`: paper-facing placeholders until verified results are imported.
- `run_manifest.json`: code revision, configuration, seeds, device, and timestamp.

Do not place smoke-test results here. Before importing formal results, verify every `K>0` row has `forced_hit_rate == 1` and that paired variants used identical task batches and intervention masks.
