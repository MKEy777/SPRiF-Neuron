"""Compatibility dispatcher. Prefer the three independent figure scripts."""
import runpy
import sys
from pathlib import Path
TARGETS = {'mechanism':'01-机制分析/plot_mechanism.py', 'sidms':'02-SI-DMS/plot_sidms.py', 'kernel':'03-时间核/plot_kernels.py', 'export':'export_parameters.py'}
if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in TARGETS:
        raise SystemExit('Usage: plot_tnnls.py {mechanism,sidms,kernel,export} [options]')
    target = Path(__file__).resolve().parent / TARGETS[sys.argv.pop(1)]
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name='__main__')
