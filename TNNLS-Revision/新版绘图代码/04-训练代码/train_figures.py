"""Train only the benchmark models used by the two composite figures."""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "benchmarks"
TASK_DIRS = {"pSMNIST": "Task_pSMNIST", "GSC": "Task_GSC", "QTDB": "Task_ECG"}
JOBS = {
    "mechanism": [("pSMNIST", "sprif_scalar"), ("pSMNIST", "memory_reset5")],
    "kernel": [("GSC", "sprif_scalar"), ("QTDB", "sprif_scalar")],
    "asrnn_gsc": [("GSC", "asrnn")],
    "asrnn_qtdb": [("QTDB", "asrnn")],
}
JOBS["all"] = JOBS["mechanism"] + JOBS["kernel"]


def expected_checkpoint(root, task, model, seed):
    return root / task / model / f"seed_{seed}" / "checkpoint.pt"


def verify_checkpoint(path, task, model, seed, args=None):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=True)
    expected_protocol = "benchmark-asrnn-original-v1" if model == "asrnn" else "benchmark-spectral-reset-v1"
    if not isinstance(obj, dict) or not isinstance(obj.get("model"), dict):
        raise ValueError(f"Checkpoint lacks expected metadata: {path}")
    if model == "asrnn":
        required = {"dense_1.tau_m"} if task == "GSC" else {"tau_m_h"}
    else:
        required = {f"layers.0.{key}" for key in
                    ("alpha_raw", "rho_raw", "omega_raw", "eta_raw", "bar_G", "input_linear.weight")}
    if not required.issubset(obj["model"]):
        raise ValueError(f"Checkpoint is missing required model parameters: {path}")
    expected = {"task": task, "name": model, "seed": seed, "protocol_version": expected_protocol}
    for key, value in expected.items():
        if obj.get(key) != value:
            raise ValueError(f"{path}: {key}={obj.get(key)!r}, expected {value!r}")
    recorded = obj.get("training_args")
    if not isinstance(recorded, dict) or not isinstance(obj.get("selected_epoch"), int):
        raise ValueError(f"Checkpoint lacks training arguments or selected epoch: {path}")
    if args is not None:
        mappings = {"pSMNIST": [("psmnist_data_root", "data_root")],
                    "GSC": [("gsc_data_root", "data_root")],
                    "QTDB": [("qtdb_train_mat", "train_mat"), ("qtdb_test_mat", "test_mat")]}
        workdir = BENCH / TASK_DIRS[task]
        for requested_key, recorded_key in mappings[task]:
            requested = getattr(args, requested_key)
            if requested is None:
                continue
            if not recorded.get(recorded_key):
                raise ValueError(f"Cannot verify {recorded_key} in {path}")
            original = Path(recorded[recorded_key])
            if not original.is_absolute():
                original = workdir / original
            if original.resolve() != requested.resolve():
                raise ValueError(f"Existing checkpoint was trained with a different {recorded_key}: {path}")


def command_for(args, task, model, staging):
    workdir = BENCH / TASK_DIRS[task]
    program = "train_asrnn.py" if model == "asrnn" else "train.py"
    command = [sys.executable, str(workdir / program), "--seed", str(args.seed),
               "--checkpoint-path", str(staging)]
    if task == "pSMNIST":
        command += ["--model", model]
        if args.psmnist_data_root:
            command += ["--data-root", str(args.psmnist_data_root.resolve())]
    elif task == "GSC":
        if args.gsc_data_root:
            command += ["--data-root", str(args.gsc_data_root.resolve())]
        if args.gsc_cache_root:
            command += ["--cache-root", str(args.gsc_cache_root.resolve())]
    elif task == "QTDB":
        if args.qtdb_train_mat:
            command += ["--train-mat", str(args.qtdb_train_mat.resolve())]
        if args.qtdb_test_mat:
            command += ["--test-mat", str(args.qtdb_test_mat.resolve())]
    return workdir, command


def validate_data_paths(args, pending):
    tasks = {task for task, _ in pending}
    if "GSC" in tasks:
        gsc = (args.gsc_data_root.resolve() if args.gsc_data_root else
               BENCH / "Task_GSC" / "dataset" / "SpeechCommands" / "speech_commands_v0.02")
        required = [gsc / name for name in ("testing_list.txt", "validation_list.txt", "silence_validation_list.txt")]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("GSC data directory is incomplete: " + ", ".join(missing))
    if "QTDB" in tasks:
        task = BENCH / "Task_ECG"
        train = args.qtdb_train_mat.resolve() if args.qtdb_train_mat else task / "data" / "QTDB_train.mat"
        test = args.qtdb_test_mat.resolve() if args.qtdb_test_mat else task / "data" / "QTDB_test.mat"
        missing = [str(path) for path in (train, test) if not path.is_file()]
        if missing:
            raise FileNotFoundError("QTDB data files are missing: " + ", ".join(missing))


def validate_runtime(pending):
    modules = {"torch", "numpy"}
    if any(task in {"pSMNIST", "GSC"} for task, _ in pending):
        modules.add("torchvision")
    if any(task in {"GSC", "QTDB"} for task, _ in pending):
        modules.add("scipy")
    if any(task == "GSC" for task, _ in pending):
        modules.add("librosa")
    for name in sorted(modules) if pending else ():
        try:
            importlib.import_module(name)
        except Exception as exc:
            raise RuntimeError(f"Training environment cannot import {name}: {exc}") from exc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", choices=list(JOBS), required=True,
                   help="mechanism: 2 pSMNIST models; kernel: GSC/QTDB SPRiF; all: these four; ASRNN jobs: one baseline each if needed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--psmnist-data-root", type=Path)
    p.add_argument("--gsc-data-root", type=Path)
    p.add_argument("--gsc-cache-root", type=Path)
    p.add_argument("--qtdb-train-mat", type=Path)
    p.add_argument("--qtdb-test-mat", type=Path)
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = p.parse_args()
    if args.seed < 0:
        p.error("--seed must be nonnegative")
    root = args.checkpoint_root.resolve()
    print(f"Job {args.job}: {len(JOBS[args.job])} model(s), seed={args.seed}", flush=True)
    if not args.dry_run:
        pending = []
        for task, model in JOBS[args.job]:
            final = expected_checkpoint(root, task, model, args.seed)
            staging = final.with_name("checkpoint.training.pt")
            if final.exists():
                verify_checkpoint(final, task, model, args.seed, args)
            elif staging.exists():
                raise FileExistsError(f"Incomplete prior run preserved at {staging}; inspect it before starting a new run")
            else:
                pending.append((task, model))
        validate_data_paths(args, pending)
        validate_runtime(pending)
    for task, model in JOBS[args.job]:
        final = expected_checkpoint(root, task, model, args.seed)
        staging = final.with_name("checkpoint.training.pt")
        workdir, command = command_for(args, task, model, staging)
        print(f"{task} {model}: {final}", flush=True)
        if args.dry_run:
            print("  " + subprocess.list2cmdline(command), flush=True)
            continue
        if final.exists():
            verify_checkpoint(final, task, model, args.seed, args)
            print("  Reusing completed checkpoint; no retraining.", flush=True)
            continue
        if staging.exists():
            raise FileExistsError(f"Incomplete prior run preserved at {staging}; inspect it before starting a new run")
        final.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=workdir, check=True)
        if not staging.exists():
            raise RuntimeError(f"Training ended without a selected checkpoint: {task} {model}")
        verify_checkpoint(staging, task, model, args.seed, args)
        if final.exists():
            raise FileExistsError(f"A final checkpoint appeared while training: {final}")
        os.replace(staging, final)
        print(f"  Completed: {final}", flush=True)
    print("Requested job complete.", flush=True)


if __name__ == "__main__":
    main()
