"""Run a full ablation sweep over the named heuristics.

For each heuristic registered in `src.zero_shot.heuristics.HEURISTICS`:
  1. Apply it to the cached features → eval/val_zs_<heuristic>.csv
  2. Run scripts/eval_harness.py on the result → reports/zs_<heuristic>/
  3. (optional) Same with TTA if a flipped feature cache is provided

After running everything, prints a summary table comparing all variants.

Usage:
    # Standard ablation on the cached features
    python scripts/zero_shot/run_ablation.py \\
        --features eval/cache/val_segformer_features.csv

    # With TTA (also needs the flipped cache)
    python scripts/zero_shot/run_ablation.py \\
        --features eval/cache/val_segformer_features.csv \\
        --features-flipped eval/cache/val_segformer_features_flip.csv

    # Filter to a subset of heuristics
    python scripts/zero_shot/run_ablation.py \\
        --features eval/cache/val_segformer_features.csv \\
        --only simple_hull simple_hull_scaled multi_feature
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.zero_shot.heuristics import HEURISTICS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True,
                   help="Path to val features cache CSV")
    p.add_argument("--features-flipped", default=None,
                   help="Optional: path to flipped features cache CSV (enables TTA variants)")
    p.add_argument("--only", nargs="*", default=None,
                   help="Subset of heuristic names to run (default: all)")
    p.add_argument("--prefix", default="zs",
                   help="Variant name prefix (default: 'zs')")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "eval"),
                   help="Directory for the heuristic prediction CSVs")
    p.add_argument("--design-samples", default="",
                   help="Comma-separated filenames used to design the heuristics (TF disclosure)")
    p.add_argument("--force", action="store_true",
                   help="Pass --force to eval_harness.py (overwrites existing reports)")
    return p.parse_args()


def run(cmd: list[str]) -> int:
    """Run a subprocess, stream output, return exit code."""
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        # Print last few useful lines
        lines = [l for l in proc.stdout.split("\n") if l.strip()]
        for line in lines[-8:]:
            print(f"  {line}")
    if proc.returncode != 0:
        print(f"  (exit code {proc.returncode})")
        if proc.stderr:
            print(f"  stderr: {proc.stderr[:500]}")
    return proc.returncode


def main() -> None:
    args = parse_args()
    heuristics = list(args.only) if args.only else list(HEURISTICS)
    for h in heuristics:
        if h not in HEURISTICS:
            print(f"ERROR: unknown heuristic {h!r}. Available: {list(HEURISTICS)}")
            sys.exit(1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    repo = str(REPO_ROOT)

    plan = []
    for h in heuristics:
        plan.append((h, False, f"{args.prefix}_{h}"))
        if args.features_flipped:
            plan.append((h, True, f"{args.prefix}_{h}_tta"))

    print(f"running {len(plan)} variants: {[p[2] for p in plan]}")

    failures = []
    for h, use_tta, variant_name in plan:
        pred_path = out_dir / f"val_{variant_name}.csv"
        cmd = [python, "scripts/zero_shot/heuristic_predict.py",
               "--features", args.features,
               "--heuristic", h,
               "--out", str(pred_path)]
        if use_tta:
            cmd += ["--features-flipped", args.features_flipped]
        rc = run(cmd)
        if rc != 0:
            failures.append(variant_name)
            continue

        notes = (f"TF-compliant SegFormer heuristic '{h}'"
                 + (" + TTA flip averaging" if use_tta else ""))
        harness_cmd = [python, "scripts/eval_harness.py",
                       "--predictions", str(pred_path),
                       "--variant", variant_name,
                       "--notes", notes]
        if args.design_samples:
            harness_cmd += ["--design-samples", args.design_samples]
        if args.force:
            harness_cmd += ["--force"]
        rc = run(harness_cmd)
        if rc != 0:
            failures.append(variant_name)

    if failures:
        print(f"\n{len(failures)} variant(s) failed: {failures}")

    print("\n" + "=" * 70 + "\nSummary (sorted by estimated test-like score):")
    run([python, "scripts/compare_reports.py",
         "--filter", args.prefix + "_",
         "--markdown"])


if __name__ == "__main__":
    main()
