"""
Worker: run ONE model on ONE file, write ONE CSV row.

Invoked by launcher.py via subprocess for file-level parallelism.

CRITICAL: We set CUDA_VISIBLE_DEVICES based on the model BEFORE any imports.
This prevents CPU-bound models (PCA, IForest, CBLOF, RobustPCA, KMeansAD)
from reserving GPU memory through PyTorch's import-time CUDA initialization.
Without this, every CPU worker also tries to load cufft/cublas DLLs,
multiplying memory pressure by the number of workers.

Usage:
    python run_one_file.py --model PCA --file ./TSB-AD-M/174_..._.csv \
        --out-csv ./results/pca_per_file.csv --dataset Exathlon
"""

import os
import sys

# ---- GPU isolation: BEFORE any other imports -------------------------------
# Parse just the --model arg ourselves so we can set CUDA_VISIBLE_DEVICES
# correctly. Using argparse here would import too much before we control
# the environment.
_GPU_MODELS = {"OmniAnomaly"}
_model_arg = None
for i, arg in enumerate(sys.argv):
    if arg == "--model" and i + 1 < len(sys.argv):
        _model_arg = sys.argv[i + 1]
        break

if _model_arg in _GPU_MODELS:
    # Allow GPU access for OmniAnomaly. Don't override if user already set it.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
else:
    # Hide GPU from CPU-only workers. PyTorch will still import (because it's
    # in the TSB-AD import chain), but it won't reserve GPU memory or load
    # CUDA DLLs. This dramatically cuts per-process memory.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Reduce thread pool sizes per worker. With multiple workers, having each
# spawn 8+ BLAS threads causes contention and inflates memory.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Now safe to import the rest -------------------------------------------------
import argparse
import time
import traceback
from pathlib import Path

import pandas as pd


# ---- Cross-platform file locking -------------------------------------------
if os.name == "nt":
    import msvcrt

    def _lock(fh):
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        except OSError:
            pass

    def _unlock(fh):
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _lock(fh):
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

    def _unlock(fh):
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _append_row_safe(csv_path: Path, row: dict):
    """Append one row to csv_path with cross-process locking."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        _lock(fh)
        try:
            df = pd.DataFrame([row])
            df.to_csv(fh, mode="a", header=new_file, index=False)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            _unlock(fh)


def _load_model(name: str):
    """Lazy-import the helper. Mirrors run_model_all_datasets.py."""
    if name == "PCA":
        from pca_helper import anomaly_PCA
        return anomaly_PCA
    if name == "IForest":
        from iforest_helper import anomaly_IForest
        return anomaly_IForest
    if name == "CBLOF":
        from cblof_helper import anomaly_CBLOF
        return anomaly_CBLOF
    if name == "RobustPCA":
        from robustpca_helper import anomaly_RobustPCA
        return anomaly_RobustPCA
    if name == "KMeansAD":
        from kmeansad_helper import anomaly_KMeansAD
        return anomaly_KMeansAD
    if name == "OmniAnomaly":
        from omnianomaly_helper import anomaly_OmniAnomaly
        return anomaly_OmniAnomaly
    raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()

    fail_csv = args.out_csv.with_suffix(".failures.csv")

    try:
        detector_fn = _load_model(args.model)
    except Exception as e:
        _append_row_safe(fail_csv, {
            "model": args.model,
            "dataset": args.dataset,
            "filename": args.file.name,
            "error": f"ImportError: {e}",
            "trace_tail": traceback.format_exc().splitlines()[-1],
        })
        sys.exit(0)

    try:
        t0 = time.time()
        _, metrics = detector_fn(str(args.file))
        dt = time.time() - t0
        metrics = {k: float(v) for k, v in dict(metrics).items()}

        row = {
            "model": args.model,
            "dataset": args.dataset,
            "filename": args.file.name,
            "runtime_sec": round(dt, 2),
            **metrics,
        }
        _append_row_safe(args.out_csv, row)
        print(f"[OK] {args.model} | {args.file.name} | {dt:.1f}s")
    except Exception as e:
        _append_row_safe(fail_csv, {
            "model": args.model,
            "dataset": args.dataset,
            "filename": args.file.name,
            "error": f"{type(e).__name__}: {e}",
            "trace_tail": traceback.format_exc().splitlines()[-1],
        })
        print(f"[FAIL] {args.model} | {args.file.name} | {type(e).__name__}: {e}")

    sys.exit(0)


if __name__ == "__main__":
    main()