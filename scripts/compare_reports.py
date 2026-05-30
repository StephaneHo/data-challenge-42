"""Side-by-side comparison of all eval_harness reports.

Reads every reports/<variant>/score.json (and meta.json), prints a single
comparison table. Useful for the Moodle report and for quick decision-making
across variants.

Usage:
    python scripts/compare_reports.py                  # all reports
    python scripts/compare_reports.py --filter zero_shot  # only matching variants
    python scripts/compare_reports.py --out reports/comparison.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", default=str(REPO_ROOT / "reports"))
    p.add_argument("--filter", default="", help="substring match on variant name")
    p.add_argument("--sort-by", default="test-like (from brief)",
                   help="column to sort by (default: estimated final score)")
    p.add_argument("--out", default=None, help="optional path to save the CSV")
    return p.parse_args()


def load_reports(reports_dir: Path, filter_str: str) -> pd.DataFrame:
    rows = []
    for sub in sorted(reports_dir.iterdir()):
        if not sub.is_dir():
            continue
        if filter_str and filter_str not in sub.name:
            continue
        score_path = sub / "score.json"
        meta_path = sub / "meta.json"
        if not score_path.exists() or not meta_path.exists():
            print(f"  skipping {sub.name}: missing score.json or meta.json")
            continue
        with open(score_path) as f:
            score_data = json.load(f)
        with open(meta_path) as f:
            meta = json.load(f)
        row = {
            "variant": sub.name,
            "score": score_data["official_score"]["score"],
            "err_F": score_data["official_score"]["err_female"],
            "err_M": score_data["official_score"]["err_male"],
            "gap": score_data["official_score"]["gap"],
            "n_F": score_data["official_score"]["n_female"],
            "n_M": score_data["official_score"]["n_male"],
        }
        for k, v in score_data.get("reweighted_scores", {}).items():
            row[k] = v
        row["subset_hash"] = meta.get("eval_subset_hash", "")
        row["notes"] = meta.get("notes", "")[:60]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        print(f"no reports directory at {reports_dir}")
        sys.exit(1)

    df = load_reports(reports_dir, args.filter)
    if df.empty:
        print("no reports found")
        sys.exit(0)

    # Ensure all reports used the same eval subset
    if df["subset_hash"].nunique() > 1:
        print("WARNING: reports were generated against DIFFERENT eval subsets — "
              "their absolute scores are NOT directly comparable:")
        print(df.groupby("subset_hash")["variant"].apply(list).to_string())
        print()

    if args.sort_by in df.columns:
        df = df.sort_values(args.sort_by).reset_index(drop=True)

    # Pretty-print
    display_cols = ["variant", "score", "err_F", "err_M", "gap"]
    reweight_cols = [c for c in df.columns
                     if c in ("val (subset native)", "test-like (from brief)",
                              "test-like (more spread)", "uniform [0, 0.50)")]
    display_cols += reweight_cols
    print(df[display_cols].round(5).to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
