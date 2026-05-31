"""Cache CLIP zero-shot occlusion predictions on val.

For each image, we run CLIP and compute the similarity to each of:
    "a clear face with no occlusion"
    "a face with about 10% occluded"
    "a face with about 20% occluded"
    ...
    "a face that is mostly occluded"

The softmax over similarities gives a probability distribution over occlusion
levels, and we take the expected value (sum of level × prob) as the prediction.

Also computes mask-specific signals:
    P("a face wearing a surgical mask")
    P("a face wearing sunglasses")
    P("a face partially covered by a hand")

Compliance: CLIP is a pre-trained foundation model with public weights.
No fit on the IDEMIA dataset.

Output columns: filename, clip_occlusion_pred, clip_p_mask, clip_p_glasses, clip_p_hand

Usage:
    python scripts/zero_shot/cache_val_clip.py --source val
    python scripts/zero_shot/cache_val_clip.py --source test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import stratified_split  # noqa: E402


OCCLUSION_PROMPTS = [
    ("a clear unobstructed face fully visible",           0.00),
    ("a face that is very lightly occluded",              0.05),
    ("a face that is slightly occluded",                  0.10),
    ("a face with mild occlusion",                        0.15),
    ("a face that is partially covered",                  0.20),
    ("a face moderately occluded",                        0.25),
    ("a face that is significantly occluded",             0.35),
    ("a face that is largely covered",                    0.50),
    ("a face that is mostly hidden",                      0.70),
]

OCCLUDER_PROMPTS = {
    "mask": "a person wearing a surgical face mask",
    "glasses": "a face wearing dark sunglasses covering the eyes",
    "hand": "a face partially covered by a hand",
}

NONOCCLUDER_PROMPT = "a clear face with no covering"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--source", choices=["val", "test"], default="val")
    p.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.source == "val":
        train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
        _, df = stratified_split(train_csv, val_frac=args.val_frac, seed=args.seed)
    else:
        df = pd.read_csv(Path(args.data_dir) / "test_students.csv")
    if args.limit > 0:
        df = df.head(args.limit)
    print(f"running CLIP on {len(df)} {args.source} rows")

    out = Path(args.out) if args.out else (
        REPO_ROOT / "eval" / "cache" / f"{args.source}_clip.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading CLIP model ({args.model_name})...")
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(device).eval()

    # Pre-tokenize prompts (text inputs are cached and passed each batch)
    all_prompts = [p for p, _ in OCCLUSION_PROMPTS] + list(OCCLUDER_PROMPTS.values()) + [NONOCCLUDER_PROMPT]
    text_inputs = processor(text=all_prompts, return_tensors="pt", padding=True)
    text_input_ids = text_inputs["input_ids"].to(device)
    text_attention_mask = text_inputs["attention_mask"].to(device)

    n_occ = len(OCCLUSION_PROMPTS)
    occ_levels = np.array([lvl for _, lvl in OCCLUSION_PROMPTS], dtype=np.float64)

    rows = []
    image_dir = Path(args.image_dir)
    filenames = df["filename"].tolist()
    with torch.inference_mode():
        for i in tqdm(range(0, len(filenames), args.batch_size), desc="clip"):
            batch_fns = filenames[i:i + args.batch_size]
            imgs = [Image.open(image_dir / fn).convert("RGB") for fn in batch_fns]
            img_inputs = processor(images=imgs, return_tensors="pt").to(device)

            outputs = model(
                pixel_values=img_inputs["pixel_values"],
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            )
            # logits_per_image: (batch_size, num_prompts), already scaled by logit_scale
            sims = outputs.logits_per_image.cpu().numpy()

            for j, fn in enumerate(batch_fns):
                occ_sims = sims[j, :n_occ]
                occ_probs = np.exp(occ_sims - occ_sims.max())
                occ_probs = occ_probs / occ_probs.sum()
                clip_pred = float(np.sum(occ_probs * occ_levels))

                # Binary occluder probabilities via softmax(occluder vs nonoccluder)
                nonocc_sim = float(sims[j, -1])
                row = {"filename": fn, "clip_occlusion_pred": clip_pred}
                for idx, key in enumerate(OCCLUDER_PROMPTS):
                    occ_sim = float(sims[j, n_occ + idx])
                    # Softmax between this occluder and the non-occluder prompt
                    e_occ = np.exp(occ_sim - max(occ_sim, nonocc_sim))
                    e_non = np.exp(nonocc_sim - max(occ_sim, nonocc_sim))
                    row[f"clip_p_{key}"] = float(e_occ / (e_occ + e_non))
                rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(out_df)} rows)")
    print(f"clip_occlusion_pred: mean={out_df['clip_occlusion_pred'].mean():.4f}, "
          f"std={out_df['clip_occlusion_pred'].std():.4f}")
    for key in OCCLUDER_PROMPTS:
        col = f"clip_p_{key}"
        print(f"{col}: mean={out_df[col].mean():.4f}, "
              f"frac > 0.5: {(out_df[col] > 0.5).mean():.3f}")


if __name__ == "__main__":
    main()
