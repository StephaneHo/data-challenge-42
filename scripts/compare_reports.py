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
    p.add_argument("--markdown", action="store_true",
                   help="emit a paste-ready markdown table (for EXPERIMENTS.md)")
    p.add_argument("--reference", default=None,
                   help="Variant name to use as the reference for delta columns. "
                        "Defaults to the first row after sorting (i.e. the best variant).")
    p.add_argument("--per-bin", action="store_true",
                   help="Also load reports/<variant>/per_bin.csv and print bin breakdown "
                        "per gender for each variant.")
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

    # Determine reference row for delta computation
    if args.reference and args.reference in df["variant"].values:
        ref_row = df[df["variant"] == args.reference].iloc[0]
        ref_name = args.reference
    else:
        ref_row = df.iloc[0]
        ref_name = ref_row["variant"]

    # Compute deltas vs reference
    df["delta_score"] = df["score"] - ref_row["score"]
    df["delta_gap"] = df["gap"] - ref_row["gap"]
    if "test-like (from brief)" in df.columns:
        df["delta_test_like"] = df["test-like (from brief)"] - ref_row["test-like (from brief)"]

    # Pretty-print
    display_cols = ["variant", "score", "err_F", "err_M", "gap",
                    "delta_score", "delta_gap"]
    reweight_cols = [c for c in df.columns
                     if c in ("val (subset native)", "test-like (from brief)",
                              "test-like (more spread)", "uniform [0, 0.50)")]
    display_cols += reweight_cols
    if "delta_test_like" in df.columns:
        display_cols.append("delta_test_like")

    print(f"\nreference variant: {ref_name}")
    print("(delta_* columns: negative = improvement vs reference)\n")

    if args.markdown:
        # Compact markdown row format intended for EXPERIMENTS.md
        print("| Variant | Subset score | delta vs ref | Test-like est. | delta vs ref | Gap F/M | delta gap |")
        print("|---|---|---|---|---|---|---|")
        for _, row in df.iterrows():
            test_like = row.get("test-like (from brief)", float("nan"))
            delta_tl = row.get("delta_test_like", float("nan"))
            arrow_score = "(ref)" if row["variant"] == ref_name else (
                f"{row['delta_score']:+.5f}"
            )
            arrow_tl = "(ref)" if row["variant"] == ref_name else f"{delta_tl:+.5f}"
            arrow_gap = "(ref)" if row["variant"] == ref_name else f"{row['delta_gap']:+.5f}"
            print(f"| `{row['variant']}` | {row['score']:.5f} | {arrow_score} | "
                  f"{test_like:.5f} | {arrow_tl} | {row['gap']:.5f} | {arrow_gap} |")
    else:
        print(df[display_cols].round(5).to_string(index=False))

    # Optional per-bin breakdown
    if args.per_bin:
        print("\n" + "=" * 70 + "\nPer-bin breakdown (weighted error per gender × bin)")
        for _, row in df.iterrows():
            variant = row["variant"]
            bin_path = Path(args.reports_dir) / variant / "per_bin.csv"
            if not bin_path.exists():
                continue
            bb = pd.read_csv(bin_path)
            pivot = bb.pivot(index="bin", columns="gender", values="weighted_err")
            print(f"\n--- {variant} ---")
            print(pivot.round(5).to_string())

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
