"""
Parallel launcher for the TSB-AD-M anomaly detection benchmark.

Dispatches per-file work to multiple worker subprocesses. For each
(model, file) pair, it invokes run_one_file.py, which appends one CSV
row to results/<model>_per_file.csv with cross-process file locking.

Two pools run SIMULTANEOUSLY:
  - CPU pool: unsupervised models (PCA, IForest, CBLOF, RobustPCA, KMeansAD)
  - GPU pool: OmniAnomaly (or any model in --gpu-models)

This is the key difference from a sequential launcher: while the CPU pool
chews through 900 unsupervised tasks, the GPU pool simultaneously chews
through 180 OmniAnomaly tasks. End-to-end wall time = max(CPU group, GPU
group), not sum.

Why subprocesses (not threads or in-process multiprocessing)?
  - Many TSB-AD models hold large NumPy/PyTorch state. Forking can mutate
    shared state in surprising ways. A clean subprocess per file is
    safer and matches the single-CPU runner exactly.
  - Each subprocess gets fresh imports, so a crash in one file can't
    poison the rest of the run.
  - Works identically on Linux and Windows (no fork-vs-spawn drama).

Usage:
    # All 6 models, CPU and GPU pools running in parallel
    python launcher.py --models PCA IForest CBLOF RobustPCA KMeansAD OmniAnomaly \
        --data-dir ./TSB-AD-M --workers 8 --gpu-workers 1

    # Smoke test: 1 file per dataset
    python launcher.py --models PCA --data-dir ./TSB-AD-M --limit-per-dataset 1
"""

import argparse
import multiprocessing
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


# Same dataset list and matching logic as the single-CPU runner -------------

F2A_DATASETS = [
    "GECCO", "PSM", "Daphnet", "Genesis", "SWaT", "CreditCard",
    "GHL", "OPP", "SMAP", "MSL", "MITDB", "SVDB", "Exathlon",
    "SMD", "LTDB", "TAO",
]

DATASET_ALIASES = {
    "OPP": ["OPP", "OPPORTUNITY", "Opportunity"],
}

FNAME_RE = re.compile(r"^\d+_([A-Za-z0-9]+)_id_")

GPU_MODELS_DEFAULT = {"OmniAnomaly"}


def _canonical_dataset(token: str):
    if token in F2A_DATASETS:
        return token
    for canon, aliases in DATASET_ALIASES.items():
        if token in aliases:
            return canon
    return None


def discover_files(data_dir: Path, eval_only: bool, eval_csv_path: Path):
    """Return list of (canonical_dataset, filepath) tuples."""
    eval_files = None
    if eval_only:
        if not eval_csv_path.exists():
            raise FileNotFoundError(
                f"Eval list not found at {eval_csv_path}. "
                f"Pass --eval-csv or use --no-eval-filter."
            )
        eval_files = set(pd.read_csv(eval_csv_path)["file_name"])

    out = []
    for fp in sorted(data_dir.glob("*.csv")):
        if eval_only and fp.name not in eval_files:
            continue
        m = FNAME_RE.match(fp.name)
        if not m:
            continue
        canon = _canonical_dataset(m.group(1))
        if canon:
            out.append((canon, fp))
    return out


def _run_worker(model: str, dataset: str, filepath: Path, out_csv: Path,
                python_exe: str, worker_script: Path):
    """Invoke run_one_file.py as a subprocess. Runs in a pool worker thread."""
    cmd = [
        python_exe, str(worker_script),
        "--model", model,
        "--file", str(filepath),
        "--dataset", dataset,
        "--out-csv", str(out_csv),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=3600,    # 1 hour per file is generous
        )
        return (model, dataset, filepath.name, result.returncode, result.stdout.strip())
    except subprocess.TimeoutExpired:
        return (model, dataset, filepath.name, -1, "TIMEOUT after 3600s")
    except Exception as e:
        return (model, dataset, filepath.name, -2, f"DISPATCH ERROR: {e}")


def _aggregate(per_file_csv: Path, out_dataset_csv: Path, model: str):
    """Compute per-dataset means from the per-file CSV."""
    if not per_file_csv.exists():
        print(f"  [WARN] No per-file results found for {model}")
        return

    df = pd.read_csv(per_file_csv)
    if df.empty:
        print(f"  [WARN] {per_file_csv} is empty")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = (
        df.groupby("dataset")[numeric_cols]
        .mean()
        .reindex(F2A_DATASETS)
        .round(4)
    )
    summary.insert(
        0, "n_files",
        df.groupby("dataset").size().reindex(F2A_DATASETS).fillna(0).astype(int),
    )
    summary.to_csv(out_dataset_csv)

    print(f"\n=== {model} per-dataset means ===")
    priority = ["VUS-PR", "VUS_PR", "AUC-PR", "AUC-ROC", "Standard-F1"]
    cols = ["n_files"] + [c for c in priority if c in summary.columns]
    print(summary[cols].to_string())


def _run_pool(label: str, tasks: list, n_workers: int, python_exe: str,
              worker_script: Path, progress_state: dict, lock: threading.Lock):
    """Run a list of tasks in a ProcessPoolExecutor.

    `tasks` is a list of (model, dataset, filepath, out_csv) tuples.
    `progress_state` and `lock` are shared with the other pool so we can
    print a unified progress line.
    """
    if not tasks:
        print(f"--- {label} pool: no tasks, skipping ---")
        return

    print(f"--- {label} pool: {len(tasks)} tasks across {n_workers} workers ---")
    pool_t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_run_worker, m, d, fp, oc, python_exe, worker_script): (m, d, fp.name)
            for (m, d, fp, oc) in tasks
        }
        for fut in as_completed(futures):
            m, d, fname = futures[fut]
            try:
                model, dataset, filename, rc, msg = fut.result()
                with lock:
                    progress_state["completed"] += 1
                    if rc != 0 or msg.startswith("[FAIL]") or msg.startswith("TIMEOUT"):
                        progress_state["failed"] += 1
                    completed = progress_state["completed"]
                    failed = progress_state["failed"]
                    total = progress_state["total"]
                    if completed % 10 == 0 or completed == total:
                        elapsed = time.time() - progress_state["start_time"]
                        rate = completed / max(elapsed, 1)
                        eta = (total - completed) / max(rate, 0.001)
                        print(f"  [{completed}/{total}] elapsed={elapsed:.0f}s "
                              f"rate={rate:.2f}/s ETA={eta:.0f}s failures={failed}")
            except Exception as e:
                with lock:
                    progress_state["failed"] += 1
                print(f"  [DISPATCH ERROR] {m} {fname}: {e}")

    print(f"--- {label} pool finished in {time.time() - pool_t0:.0f}s ---\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, nargs="+",
                        help="One or more model names to run.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("./results"), type=Path)
    parser.add_argument("--eval-csv", default=Path("./TSB-AD-M-Eva.csv"), type=Path)
    parser.add_argument("--no-eval-filter", action="store_true")
    parser.add_argument("--limit-per-dataset", type=int, default=None)

    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for CPU models. Default: min(#cores-1, 8).")
    parser.add_argument("--gpu-workers", type=int, default=1,
                        help="Parallel workers for GPU models (default 1 to avoid OOM).")
    parser.add_argument("--gpu-models", nargs="+", default=list(GPU_MODELS_DEFAULT),
                        help="Models that should use the GPU pool.")

    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing per-file CSVs before starting.")
    args = parser.parse_args()

    if args.workers is None:
        args.workers = min(max(1, multiprocessing.cpu_count() - 1), 8)
    args.workers = max(1, args.workers)
    args.gpu_workers = max(1, args.gpu_workers)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    worker_script = Path(__file__).parent / "run_one_file.py"
    if not worker_script.exists():
        print(f"ERROR: run_one_file.py not found at {worker_script}")
        sys.exit(1)

    # Discover files once; reuse across all models
    file_list = discover_files(
        args.data_dir,
        eval_only=not args.no_eval_filter,
        eval_csv_path=args.eval_csv,
    )

    if args.limit_per_dataset:
        seen = {}
        capped = []
        for dataset, fp in file_list:
            seen[dataset] = seen.get(dataset, 0) + 1
            if seen[dataset] <= args.limit_per_dataset:
                capped.append((dataset, fp))
        file_list = capped

    eval_label = "Eval-only" if not args.no_eval_filter else "ALL files"

    # Split tasks into CPU and GPU groups
    gpu_set = set(args.gpu_models)
    cpu_tasks = []
    gpu_tasks = []
    for model in args.models:
        tag = model.lower()
        out_csv = args.out_dir / f"{tag}_per_file.csv"
        target = gpu_tasks if model in gpu_set else cpu_tasks
        for dataset, fp in file_list:
            target.append((model, dataset, fp, out_csv))

    total_tasks = len(cpu_tasks) + len(gpu_tasks)

    print(f"\nLauncher configured:")
    print(f"  Models:        {args.models}")
    print(f"  Mode:          {eval_label}")
    print(f"  Files:         {len(file_list)} per model")
    print(f"  CPU pool:      {args.workers} workers, {len(cpu_tasks)} tasks")
    print(f"  GPU pool:      {args.gpu_workers} workers, {len(gpu_tasks)} tasks (models: {args.gpu_models})")
    print(f"  Total tasks:   {total_tasks}")
    print()

    # Clean stale per-file CSVs if --fresh
    if args.fresh:
        for model in args.models:
            tag = model.lower()
            for suffix in ["_per_file.csv", "_per_file.failures.csv"]:
                p = args.out_dir / f"{tag}{suffix}"
                if p.exists():
                    p.unlink()
                    print(f"  Cleaned: {p}")
        print()

    # Shared progress state for both pools
    progress_state = {
        "completed": 0,
        "failed": 0,
        "total": total_tasks,
        "start_time": time.time(),
    }
    lock = threading.Lock()

    # Launch both pools in parallel via threads. Each thread runs its own
    # ProcessPoolExecutor. They share progress_state but have independent
    # process pools, so OmniAnomaly tasks don't block PCA tasks and vice versa.
    threads = []

    if cpu_tasks:
        t_cpu = threading.Thread(
            target=_run_pool,
            args=("CPU", cpu_tasks, args.workers, sys.executable,
                  worker_script, progress_state, lock),
            daemon=False,
        )
        threads.append(t_cpu)

    if gpu_tasks:
        t_gpu = threading.Thread(
            target=_run_pool,
            args=("GPU", gpu_tasks, args.gpu_workers, sys.executable,
                  worker_script, progress_state, lock),
            daemon=False,
        )
        threads.append(t_gpu)

    overall_t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"\nAll pools finished in {time.time() - overall_t0:.0f}s\n")

    # Aggregate per-model summaries
    for model in args.models:
        tag = model.lower()
        per_file_csv = args.out_dir / f"{tag}_per_file.csv"
        per_dataset_csv = args.out_dir / f"{tag}_per_dataset.csv"
        _aggregate(per_file_csv, per_dataset_csv, model)
        print()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # required on Windows
    main()
