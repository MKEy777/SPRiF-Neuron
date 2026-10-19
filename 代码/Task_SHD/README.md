# SHD Benchmark

Spiking Heidelberg Digits: 20-class digit classification from 700-channel event-based cochlea spikes.

Setup: place shd_train.h5 / shd_test.h5 (Zenodo) under Task_SHD/data/SHD/, then `python Task_SHD/generate_data.py`
Run from this directory: `python train.py --train-dir <existing-train-directory> --test-dir <existing-test-directory>`

The TNNLS entry reads the existing NPY files in each directory without creating
a validation split. Both hidden and spiking output layers use the five-state,
eleven-coefficient scalar-reset model. The supplied preprocessing script is
retained; its colon-containing output names require attention when preparing
files on Windows. No dataset has been regenerated here.
