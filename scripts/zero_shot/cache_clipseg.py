"""Cache CLIPSeg prompt-based segmentation on val or test images.

CLIPSeg (CIDAS/clipseg-rd64-refined) is a lightweight text-prompted
segmentation model (~50M params). Faster than Florence-2 and compatible with
our transformers version. Output is a soft segmentation mask per prompt.

For each image and prompt, we save:
  - <prompt>_max_logit : max activation on the soft mask (proxy for detection confidence)
  - <prompt>_pixel_frac : fraction of pixels with sigmoid(logit) > threshold
  - <prompt>_bbox_* : bounding box of activated region (normalized [0, 1])

Usage:
    # Sanity test on 5 images
    python scripts/zero_shot/cache_clipseg.py --prompts hat "face mask" hand sunglasses \\
        --source val --limit 5 --out eval/cache/clipseg_test5.csv

    # Full val with one or more prompts
    python scripts/zero_shot/cache_clipseg.py --prompts hat "face mask" hand \\
        --source val --out eval/cache/val_clipseg.csv
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import stratified_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--source", choices=["val", "test"], default="val")
    p.add_argument("--prompts", nargs="+", required=True,
                   help="One or more text prompts (e.g. hat 'face mask' hand)")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="CIDAS/clipseg-rd64-refined")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="Threshold on sigmoid(logit) to count a pixel as positive")
    p.add_argument("--filter-extreme", action="store_true",
                   help="Only process Julien's extreme cases (pred > 0.7)")
    return p.parse_args()


def setup_clipseg(model_name: str):
    import torch
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    print(f"loading {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    processor = CLIPSegProcessor.from_pretrained(model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(device).eval()
    return model, processor, device


def slug(prompt: str) -> str:
    return prompt.lower().replace(" ", "_").replace("-", "_")


def main() -> None:
    args = parse_args()

    if args.source == "val":
        train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
        _, df = stratified_split(train_csv, val_frac=args.val_frac, seed=args.seed)
    else:
        df = pd.read_csv(Path(args.data_dir) / "test_students.csv")

    if args.filter_extreme:
        if args.source != "val":
            raise SystemExit("--filter-extreme only valid with val")
        jul = pd.read_csv(REPO_ROOT / "eval" / "val_julien_baseline.csv")
        extreme = set(jul[jul.pred > 0.7].filename.tolist())
        df = df[df["filename"].isin(extreme)]
        print(f"filtering to {len(df)} extreme cases (Julien pred > 0.7)")

    if args.limit > 0:
        df = df.head(args.limit)
    print(f"processing {len(df)} images for prompts: {args.prompts}")

    model, processor, device = setup_clipseg(args.model)
    import torch

    rows = []
    image_dir = Path(args.image_dir)
    t0 = time.time()
    for i, fn in enumerate(tqdm(df["filename"].tolist(), desc="clipseg")):
        img = Image.open(image_dir / fn).convert("RGB")
        w, h = img.size

        # Process all prompts in one batch (shared image)
        inputs = processor(
            text=args.prompts,
            images=[img] * len(args.prompts),
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        # logits: [num_prompts, H_low, W_low]
        logits = outputs.logits  # (P, H_low, W_low)
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits).cpu().numpy()

        row = {"filename": fn}
        for j, prompt in enumerate(args.prompts):
            # Upsample to image size
            prob = cv2.resize(probs[j].astype(np.float32), (w, h),
                              interpolation=cv2.INTER_LINEAR)
            mask = (prob > args.threshold).astype(np.uint8)
            s = slug(prompt)
            row[f"{s}_max_prob"] = float(prob.max())
            row[f"{s}_mean_prob"] = float(prob.mean())
            row[f"{s}_pixel_frac"] = float(mask.sum()) / (h * w)
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                row[f"{s}_detected"] = 1
                row[f"{s}_bbox_y0"] = float(ys.min()) / h
                row[f"{s}_bbox_y1"] = float(ys.max()) / h
                row[f"{s}_bbox_x0"] = float(xs.min()) / w
                row[f"{s}_bbox_x1"] = float(xs.max()) / w
            else:
                row[f"{s}_detected"] = 0
                row[f"{s}_bbox_y0"] = 0.0
                row[f"{s}_bbox_y1"] = 0.0
                row[f"{s}_bbox_x0"] = 0.0
                row[f"{s}_bbox_x1"] = 0.0
        rows.append(row)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df) - i - 1) / rate
            print(f"  [{i+1}/{len(df)}] {rate:.2f} img/s, ETA {eta/60:.1f} min")

    out_df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    elapsed = time.time() - t0
    print(f"\nwrote {out} ({len(out_df)} rows in {elapsed:.0f}s = {elapsed/len(out_df):.2f}s/img)")
    for prompt in args.prompts:
        s = slug(prompt)
        n_det = int(out_df[f"{s}_detected"].sum())
        print(f"  {prompt!r}: detected (>{args.threshold}) in {n_det}/{len(out_df)} "
              f"({100*n_det/len(out_df):.1f}%), mean_prob={out_df[f'{s}_mean_prob'].mean():.3f}")


if __name__ == "__main__":
    main()
