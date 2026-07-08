#!/usr/bin/env python3
"""
SPRiF Health Check — Read-Only Diagnostic Tool

Usage:
    python health_check.py                   # 全量检查（默认）
    python health_check.py --verbose         # 详细信息
    python health_check.py --output json     # JSON 格式输出
    python health_check.py --tasks GSC,ECG   # 只检查指定任务

Exit codes:
    0  ✅ 全部通过
    1  ⚠️  有警告
    2  ❌ 有错误
"""

import argparse
import glob
import hashlib
import os
import re
import sys
import json
from pathlib import Path


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

ALL_TASKS = [
    "Task_GSC",
    "Task_SHD",
    "Task_ECG",
    "Task_S-MNIST",
    "Task_pSMNIST",
]

EXPERIMENTS = [
    "trajectory_analysis",
    "impulse_analysis",
    "reset_analysis",
    "param_visualization",
    "robustness",
]

EXPECTED_TASK_FILES = [
    "train.py",
    "model.py",
    "core_algorithm/__init__.py",
    "core_algorithm/sprif_layer.py",
    "core_algorithm/utils.py",
]

EXPECTED_REQUIREMENTS = [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "librosa",
]

# Checkpoint filename patterns
CKPT_PATTERN = re.compile(
    r"(?P<prefix>\w+?)"
    r"(?:_hs\d[\d_]*(?:_\d+)?)?"
    r"_bs\d+"
    r"_lr[\d.]+"
    r"_seed\d+"
    r"_acc(?P<acc>[\d.]+)"
    r"\.pth$"
)

# ──────────────────────────────────────────────
# Result tracking
# ──────────────────────────────────────────────
class HealthResult:
    def __init__(self):
        self.checks = []

    def ok(self, module: str, message: str):
        self.checks.append(("ok", module, message))

    def warn(self, module: str, message: str):
        self.checks.append(("warn", module, message))

    def err(self, module: str, message: str):
        self.checks.append(("err", module, message))

    @property
    def max_severity(self) -> int:
        """Return 0=ok, 1=warn, 2=err."""
        for s, _, _ in self.checks:
            if s == "err":
                return 2
        for s, _, _ in self.checks:
            if s == "warn":
                return 1
        return 0

    def group_by_module(self):
        groups = {}
        for severity, module, message in self.checks:
            groups.setdefault(module, []).append((severity, message))
        return groups


# ──────────────────────────────────────────────
# Check functions
# ──────────────────────────────────────────────

def check_environment(result: HealthResult, verbose: bool):
    """T2: Check Python / PyTorch / CUDA environment."""
    mod = "Environment"

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    result.ok(mod, f"Python {py_ver}")
    if verbose:
        print(f"  Python executable: {sys.executable}")

    # PyTorch
    try:
        import torch
        torch_ver = torch.__version__
        result.ok(mod, f"PyTorch {torch_ver}")
        if verbose:
            print(f"  PyTorch location: {os.path.dirname(torch.__file__)}")

        # CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            cuda_ver = torch.version.cuda or "unknown"
            result.ok(mod, f"CUDA {cuda_ver} available, {gpu_count} GPU(s)")
            if verbose:
                for i, name in enumerate(gpu_names):
                    print(f"  GPU {i}: {name}")
        else:
            result.warn(mod, "CUDA not available (PyTorch will use CPU)")
    except ImportError:
        result.err(mod, "PyTorch is NOT installed")
    except Exception as e:
        result.warn(mod, f"PyTorch check failed: {e}")


def check_structure(result: HealthResult, verbose: bool, task_filter: set = None):
    """T3: Check project directory structure."""
    mod = "Structure"

    # Check project root exists
    if not PROJECT_ROOT.exists():
        result.err(mod, f"Project root not found: {PROJECT_ROOT}")
        return

    result.ok(mod, f"Project root: {PROJECT_ROOT}")

    # Check requirements.txt
    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.is_file():
        result.ok(mod, "requirements.txt found")
    else:
        result.warn(mod, "requirements.txt missing")

    # Check each task
    tasks_to_check = [t for t in ALL_TASKS if not task_filter or t in task_filter]
    for task_name in tasks_to_check:
        task_dir = PROJECT_ROOT / task_name
        if not task_dir.is_dir():
            result.err(mod, f"Task directory missing: {task_name}")
            continue

        missing_files = []
        for rel_path in EXPECTED_TASK_FILES:
            if not (task_dir / rel_path).is_file():
                missing_files.append(rel_path)

        if missing_files:
            result.warn(mod, f"{task_name}: missing files: {', '.join(missing_files)}")
        else:
            result.ok(mod, f"{task_name}: structure OK")

        # Ablation variants (informational)
        if verbose:
            ablation_files = sorted(task_dir.glob("train_ablation_*.py"))
            if ablation_files:
                names = [f.name for f in ablation_files]
                print(f"  {task_name}: ablation scripts: {', '.join(names)}")

    # Check experiments
    experiments_dir = PROJECT_ROOT / "experiments"
    if experiments_dir.is_dir():
        for exp_name in EXPERIMENTS:
            exp_dir = experiments_dir / exp_name
            py_files = list(exp_dir.glob("*.py")) if exp_dir.is_dir() else []
            if py_files:
                result.ok(mod, f"experiments/{exp_name}: {len(py_files)} script(s)")
            else:
                result.warn(mod, f"experiments/{exp_name}: directory exists but no .py files")
            if verbose and py_files:
                print(f"  experiments/{exp_name}: {', '.join(f.name for f in py_files)}")
    else:
        result.warn(mod, "experiments/ directory missing")

    # Check ASRNN (reference baseline)
    asrnn_dir = PROJECT_ROOT.parent / "ASRNN" if PROJECT_ROOT.name == "代码" else PROJECT_ROOT / "ASRNN"
    if asrnn_dir.is_dir():
        result.ok(mod, "ASRNN reference baseline found")
    else:
        result.warn(mod, "ASRNN reference baseline not found")


def check_datasets(result: HealthResult, verbose: bool, task_filter: set = None):
    """Check dataset availability for each task."""
    mod = "Datasets"

    tasks_to_check = [t for t in ALL_TASKS if not task_filter or t in task_filter]

    # Task_GSC: check data/ directory or cache
    if not task_filter or "Task_GSC" in task_filter:
        gsc_dir = PROJECT_ROOT / "Task_GSC"
        # Check common data paths
        data_paths = [
            gsc_dir / "data" / "SpeechCommands",
            gsc_dir / "SpeechCommands",
            gsc_dir / "dataset",
        ]
        found = False
        for p in data_paths:
            if p.is_dir():
                # Look for testing_list.txt as sanity check
                if any(p.rglob("testing_list.txt")):
                    result.ok(mod, "Task_GSC: dataset found")
                    found = True
                    break
        if not found:
            # Check cache directory from args
            cache_candidates = list(gsc_dir.glob("**/cache*")) + list(gsc_dir.glob("**/cache*_power*"))
            if cache_candidates:
                result.warn(mod, f"Task_GSC: cache directory found (may have partial data)")
            else:
                result.warn(mod, "Task_GSC: dataset not found (download with download_GSC.py)")

    # Task_SHD: check train/test dirs
    if not task_filter or "Task_SHD" in task_filter:
        shd_dir = PROJECT_ROOT / "Task_SHD"
        npy_files = list(shd_dir.rglob("*_1ms/*.npy")) + list(shd_dir.rglob("*_2ms/*.npy"))
        if npy_files:
            result.ok(mod, f"Task_SHD: {len(npy_files)} .npy files found")
        else:
            # Check for generate_data.py output
            gen_output = list(shd_dir.glob("*.npy")) + list(shd_dir.glob("train_*")) + list(shd_dir.glob("test_*"))
            if gen_output:
                result.warn(mod, f"Task_SHD: possible data files found")
            else:
                result.warn(mod, "Task_SHD: dataset not found (run generate_data.py first)")

    # Task_ECG: check .mat files
    if not task_filter or "Task_ECG" in task_filter:
        ecg_dir = PROJECT_ROOT / "Task_ECG"
        mat_files = list(ecg_dir.glob("data/*.mat")) + list(ecg_dir.glob("*.mat"))
        if mat_files:
            result.ok(mod, f"Task_ECG: {len(mat_files)} .mat file(s) found")
        else:
            result.warn(mod, "Task_ECG: QTDB .mat files not found (place in data/)")

    # Task_S-MNIST / Task_pSMNIST: auto-download, check if downloaded
    for task_name in ["Task_S-MNIST", "Task_pSMNIST"]:
        if task_filter and task_name not in task_filter:
            continue
        task_dir = PROJECT_ROOT / task_name
        data_dir = task_dir / "data"
        # MNIST is downloaded by torchvision, check for MNIST folder
        mnist_dir = data_dir / "MNIST"
        if data_dir.is_dir() and (mnist_dir.is_dir() or any(data_dir.iterdir())):
            result.ok(mod, f"{task_name}: data directory found")
        else:
            result.warn(mod, f"{task_name}: data not found (will auto-download on first train)")


def check_checkpoints(result: HealthResult, verbose: bool, task_filter: set = None):
    """T4: Scan for trained .pth checkpoint files."""
    mod = "Checkpoints"

    tasks_to_check = [t for t in ALL_TASKS if not task_filter or t in task_filter]
    total_ckpts = 0
    best_by_task = {}

    for task_name in tasks_to_check:
        task_dir = PROJECT_ROOT / task_name
        if not task_dir.is_dir():
            continue

        pth_files = list(task_dir.glob("*.pth"))
        if not pth_files:
            result.warn(mod, f"{task_name}: no checkpoints found")
            continue

        best_acc = -1.0
        best_name = None
        task_ckpts = 0

        for ckpt in pth_files:
            # Try pattern matching
            m = CKPT_PATTERN.search(ckpt.name)
            if m:
                try:
                    acc = float(m.group("acc"))
                    task_ckpts += 1
                    if acc > best_acc:
                        best_acc = acc
                        best_name = ckpt.name
                except ValueError:
                    pass
            else:
                # Unrecognized .pth file
                if verbose:
                    print(f"  {task_name}: unrecognized checkpoint format: {ckpt.name}")

        total_ckpts += task_ckpts
        if best_name:
            best_by_task[task_name] = (best_name, best_acc)
            result.ok(mod, f"{task_name}: {task_ckpts} checkpoint(s), best={best_acc:.4f}")
            if verbose:
                print(f"  Best: {best_name}")
        else:
            # .pth files exist but none match the pattern
            result.warn(mod, f"{task_name}: {len(pth_files)} .pth file(s) but none match known pattern")

    if total_ckpts == 0:
        result.warn(mod, "No checkpoints found across any task")

    if verbose and best_by_task:
        print(f"  Total checkpoints: {total_ckpts}")
        for task, (name, acc) in sorted(best_by_task.items()):
            print(f"  {task}: {acc:.4f} ({name})")


def check_core_algorithm(result: HealthResult, verbose: bool, task_filter: set = None):
    """T5: Check core_algorithm consistency across tasks."""
    mod = "core_algorithm"

    # Find which tasks have core_algorithm
    tasks_with_ca = []
    for task_name in ALL_TASKS:
        if task_filter and task_name not in task_filter:
            continue
        ca_dir = PROJECT_ROOT / task_name / "core_algorithm"
        if ca_dir.is_dir():
            tasks_with_ca.append(task_name)

    if not tasks_with_ca:
        result.err(mod, "No task has core_algorithm/ directory")
        return

    if len(tasks_with_ca) < 2:
        result.warn(mod, f"Only 1 task ({tasks_with_ca[0]}) has core_algorithm, cannot compare")
        return

    # Use first task as reference
    ref_task = tasks_with_ca[0]
    ref_dir = PROJECT_ROOT / ref_task / "core_algorithm"

    # Files to compare
    ca_files = ["__init__.py", "sprif_layer.py", "utils.py"]

    inconsistent = []
    not_found_in_ref = []

    for ca_file in ca_files:
        ref_file = ref_dir / ca_file
        if not ref_file.is_file():
            not_found_in_ref.append(ca_file)
            continue

        ref_hash = hashlib.md5(ref_file.read_bytes()).hexdigest()

        for other_task in tasks_with_ca[1:]:
            other_dir = PROJECT_ROOT / other_task / "core_algorithm"
            other_file = other_dir / ca_file

            if not other_file.is_file():
                inconsistent.append((other_task, ca_file, "missing"))
                continue

            other_hash = hashlib.md5(other_file.read_bytes()).hexdigest()
            if ref_hash != other_hash:
                inconsistent.append((other_task, ca_file, "DIFFERS"))

    # Report
    if not_found_in_ref:
        result.err(mod, f"Reference ({ref_task}): missing files: {', '.join(not_found_in_ref)}")
    elif not inconsistent:
        result.ok(mod, f"All {len(tasks_with_ca)} tasks have identical core_algorithm (reference: {ref_task})")
    else:
        for task, fname, status in inconsistent:
            if status == "missing":
                result.err(mod, f"{task}/{fname}: MISSING")
            else:
                result.warn(mod, f"{task}/{fname}: differs from {ref_task}")

    if verbose and inconsistent:
        print(f"  Inconsistent files:")
        for task, fname, status in inconsistent:
            print(f"    {task}/core_algorithm/{fname}: {status}")


def check_dependencies(result: HealthResult, verbose: bool):
    """T6: Check installed packages against requirements.txt."""
    mod = "Dependencies"

    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.is_file():
        result.warn(mod, "requirements.txt not found, skipping dependency check")
        return

    # Parse requirements
    required_pkgs = {}
    with open(req_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Handle ==, >=, <=, ~=, !=
            match = re.match(r"([a-zA-Z0-9_.-]+)\s*([><=!~]+)\s*([a-zA-Z0-9_.*]+)?", line)
            if match:
                pkg_name = match.group(1).lower()
                required_pkgs[pkg_name] = line
            elif re.match(r"^[a-zA-Z0-9_.-]+$", line):
                required_pkgs[line.lower()] = line

    if not required_pkgs:
        result.warn(mod, "No packages parsed from requirements.txt")
        return

    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:
        # Python 3.7 fallback
        try:
            from importlib_metadata import version, PackageNotFoundError
        except ImportError:
            result.warn(mod, "Cannot check dependencies (importlib.metadata not available)")
            return

    installed_count = 0
    missing_count = 0
    version_mismatch = []

    for pkg_name_lower, req_line in required_pkgs.items():
        try:
            installed_ver = version(pkg_name_lower)
            installed_count += 1
            if verbose:
                print(f"  {req_line} → installed ({installed_ver})")
        except PackageNotFoundError:
            missing_count += 1
            result.err(mod, f"{req_line} → NOT INSTALLED")

    if missing_count == 0:
        result.ok(mod, f"All {installed_count} required packages installed")
    else:
        result.warn(mod, f"{missing_count} package(s) missing, {installed_count} installed")

    # Also check common ML packages not in requirements
    extra_pkgs = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
    ]
    for pkg, display_name in extra_pkgs:
        if pkg not in required_pkgs:
            try:
                ver = version(pkg)
                if verbose:
                    print(f"  {display_name} (extra): {ver}")
            except PackageNotFoundError:
                pass


# ──────────────────────────────────────────────
# Report formatting
# ──────────────────────────────────────────────

def format_report_text(result: HealthResult, verbose: bool) -> str:
    """Format as human-readable text."""
    lines = []
    sep = "─" * 60

    lines.append("")
    lines.append("  SPRiF 健康检查报告")
    lines.append(sep)

    groups = result.group_by_module()
    module_order = ["Environment", "Structure", "Datasets", "Checkpoints", "core_algorithm", "Dependencies"]

    for mod in module_order:
        if mod not in groups:
            continue
        checks = groups[mod]
        lines.append(f"\n  [{mod}]")

        for severity, message in checks:
            if severity == "ok":
                prefix = "  [OK]"
            elif severity == "warn":
                prefix = "  [WARN]"
            else:
                prefix = "  [ERR]"
            lines.append(f"{prefix}  {message}")

    lines.append("")
    lines.append(sep)

    # Summary
    total = len(result.checks)
    ok_count = sum(1 for s, _, _ in result.checks if s == "ok")
    warn_count = sum(1 for s, _, _ in result.checks if s == "warn")
    err_count = sum(1 for s, _, _ in result.checks if s == "err")

    if err_count > 0:
        lines.append(f"  Total: {total} | OK {ok_count} | WARN {warn_count} | ERR {err_count}")
        lines.append("  Status: ERR - Errors found")
    elif warn_count > 0:
        lines.append(f"  Total: {total} | OK {ok_count} | WARN {warn_count}")
        lines.append("  Status: WARN - Warnings found")
    else:
        lines.append(f"  Total: {total} | OK {ok_count}")
        lines.append("  Status: OK - All good")

    lines.append(sep)
    lines.append("")

    return "\n".join(lines)


def format_report_json(result: HealthResult) -> str:
    """Format as JSON."""
    checks_list = []
    for severity, module, message in result.checks:
        checks_list.append({
            "severity": severity,
            "module": module,
            "message": message,
        })

    total = len(result.checks)
    summary = {
        "total": total,
        "ok": sum(1 for s, _, _ in result.checks if s == "ok"),
        "warn": sum(1 for s, _, _ in result.checks if s == "warn"),
        "err": sum(1 for s, _, _ in result.checks if s == "err"),
        "status": "error" if result.max_severity == 2 else ("warning" if result.max_severity == 1 else "ok"),
    }

    report = {
        "tool": "SPRiF Health Check",
        "project_root": str(PROJECT_ROOT),
        "summary": summary,
        "checks": checks_list,
    }

    return json.dumps(report, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SPRiF Health Check — Read-only diagnostic tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0  OK    All checks passed\n"
            "  1  WARN  Warnings present\n"
            "  2  ERR   Errors present\n"
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information for each check",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names to check (e.g., GSC,ECG,SHD)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    return parser.parse_args(argv)


def resolve_task_filter(tasks_arg: str = None) -> set:
    """Convert --tasks argument to set of full task directory names."""
    if not tasks_arg:
        return set()

    short_names = {
        "gsc": "Task_GSC",
        "shd": "Task_SHD",
        "ecg": "Task_ECG",
        "smnist": "Task_S-MNIST",
        "s-mnist": "Task_S-MNIST",
        "psmnist": "Task_pSMNIST",
        "p-smnist": "Task_pSMNIST",
    }

    user_tasks = set()
    for part in tasks_arg.split(","):
        part = part.strip()
        if not part:
            continue
        full_name = short_names.get(part.lower())
        if full_name:
            user_tasks.add(full_name)
        elif part in ALL_TASKS:
            user_tasks.add(part)
        else:
            print(f"  ⚠️  Unknown task '{part}', ignoring (valid: {', '.join(ALL_TASKS)})")

    return user_tasks


def main():
    args = parse_args()
    task_filter = resolve_task_filter(args.tasks) if args.tasks else set()

    result = HealthResult()

    # Run all checks (each handles its own task filtering)
    check_environment(result, args.verbose)
    check_structure(result, args.verbose, task_filter)
    check_datasets(result, args.verbose, task_filter)
    check_checkpoints(result, args.verbose, task_filter)
    check_core_algorithm(result, args.verbose, task_filter)
    check_dependencies(result, args.verbose)

    # Output
    if args.output == "json":
        print(format_report_json(result))
    else:
        print(format_report_text(result, args.verbose))

    # Exit code
    sys.exit(result.max_severity)


if __name__ == "__main__":
    main()
