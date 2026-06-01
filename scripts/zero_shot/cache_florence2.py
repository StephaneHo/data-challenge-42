"""Cache Florence-2 prompt-based segmentation on val or test images.

Florence-2 is Microsoft's multimodal model. We use it for REFERRING_EXPRESSION_SEGMENTATION:
given an image and a text prompt (e.g., "hat"), it returns polygon(s) of the
matching region.

For each image, we save:
  - <prompt>_detected : 1 if Florence2 found any matching region
  - <prompt>_pixel_frac : proportion of image pixels covered by the polygon
  - <prompt>_bbox_y0, <prompt>_bbox_y1, <prompt>_bbox_x0, <prompt>_bbox_x1 :
    bounding box of the union of all polygons (normalized [0, 1])

This output is then combined with Julien's 3DDFA mesh (downstream) to compute
the fraction of the *theoretical face area* covered by the detected object.

Usage:
    # On test/dev: 5 images only with --limit 5, validates the pipeline + timing
    python scripts/zero_shot/cache_florence2.py --prompt hat --limit 5 --out eval/cache/val_florence2_hat_sample.csv

    # On full val 15k (slow on CPU)
    python scripts/zero_shot/cache_florence2.py --prompt hat --source val --out eval/cache/val_florence2_hat.csv

    # Multiple prompts in one run is faster (model loaded once)
    python scripts/zero_shot/cache_florence2.py --prompts hat "surgical mask" hand --source val --out eval/cache/val_florence2_multi.csv
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
    p.add_argument("--prompt", default=None, help="Single prompt (legacy form)")
    p.add_argument("--prompts", nargs="*", default=None, help="Multiple prompts in one run")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="microsoft/Florence-2-base",
                   help="Florence-2 variant (base ~230M, large ~770M)")
    p.add_argument("--filter-extreme", action="store_true",
                   help="Only process images where Julien predicts > 0.7 "
                        "(targets the 823 problematic cases)")
    return p.parse_args()


def setup_florence2(model_name: str):
    """Load Florence-2 from HuggingFace."""
    print(f"loading {model_name}...")
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor, device


def run_florence2_segmentation(model, processor, device, image: Image.Image, prompt: str):
    """Run Florence-2 referring expression segmentation.

    Returns a list of polygons. Each polygon is a list of (x, y) tuples in image coords.
    """
    import torch
    task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
    full_prompt = task_prompt + prompt
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=image.size
    )
    seg = parsed.get(task_prompt, {})
    polygons = seg.get("polygons", [])
    # polygons is a list-of-list: outer = objects, inner = polygons of that object
    flat = []
    for obj_polys in polygons:
        for poly in obj_polys:
            # poly is a flat list [x0, y0, x1, y1, ...]
            pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
            flat.append(pts)
    return flat


def stats_from_polygons(polygons, image_size):
    """Compute pixel_frac and bbox from a list of polygons."""
    w, h = image_size
    if not polygons:
        return {"detected": 0, "pixel_frac": 0.0,
                "bbox_y0": 0.0, "bbox_y1": 0.0, "bbox_x0": 0.0, "bbox_x1": 0.0}
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in polygons:
        if len(pts) < 3:
            continue
        arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [arr], 1)
    pixel_frac = float(mask.sum()) / (h * w)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return {"detected": 0, "pixel_frac": 0.0,
                "bbox_y0": 0.0, "bbox_y1": 0.0, "bbox_x0": 0.0, "bbox_x1": 0.0}
    return {
        "detected": 1,
        "pixel_frac": pixel_frac,
        "bbox_y0": float(ys.min()) / h,
        "bbox_y1": float(ys.max()) / h,
        "bbox_x0": float(xs.min()) / w,
        "bbox_x1": float(xs.max()) / w,
    }


def slug(prompt: str) -> str:
    return prompt.lower().replace(" ", "_").replace("-", "_")


def main() -> None:
    args = parse_args()
    if args.prompts is None and args.prompt is None:
        raise SystemExit("Provide --prompt or --prompts")
    prompts = args.prompts if args.prompts else [args.prompt]
    print(f"prompts: {prompts}")

    # Build the image list
    if args.source == "val":
        train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
        _, df = stratified_split(train_csv, val_frac=args.val_frac, seed=args.seed)
    else:
        df = pd.read_csv(Path(args.data_dir) / "test_students.csv")

    # Optional: only the 823 "extreme" Julien cases
    if args.filter_extreme:
        if args.source != "val":
            raise SystemExit("--filter-extreme only makes sense with val (we have val_julien_baseline.csv)")
        jul = pd.read_csv(REPO_ROOT / "eval" / "val_julien_baseline.csv")
        extreme_filenames = set(jul[jul.pred > 0.7].filename.tolist())
        df = df[df["filename"].isin(extreme_filenames)]
        print(f"filtering to {len(df)} extreme cases (Julien pred > 0.7)")

    if args.limit > 0:
        df = df.head(args.limit)
    print(f"will process {len(df)} images for {len(prompts)} prompts each")

    model, processor, device = setup_florence2(args.model)

    rows = []
    image_dir = Path(args.image_dir)
    t0 = time.time()
    for i, fn in enumerate(tqdm(df["filename"].tolist(), desc="florence2")):
        img = Image.open(image_dir / fn).convert("RGB")
        row = {"filename": fn}
        for prompt in prompts:
            try:
                polys = run_florence2_segmentation(model, processor, device, img, prompt)
                stats = stats_from_polygons(polys, img.size)
            except Exception as e:
                print(f"\nfailed {fn} prompt={prompt!r}: {e}")
                stats = {"detected": 0, "pixel_frac": 0.0,
                         "bbox_y0": 0.0, "bbox_y1": 0.0, "bbox_x0": 0.0, "bbox_x1": 0.0}
            s = slug(prompt)
            row[f"{s}_detected"] = stats["detected"]
            row[f"{s}_pixel_frac"] = stats["pixel_frac"]
            row[f"{s}_bbox_y0"] = stats["bbox_y0"]
            row[f"{s}_bbox_y1"] = stats["bbox_y1"]
            row[f"{s}_bbox_x0"] = stats["bbox_x0"]
            row[f"{s}_bbox_x1"] = stats["bbox_x1"]
        rows.append(row)

        # Periodic timing print
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(df) - i - 1) / rate
            print(f"  [{i+1}/{len(df)}] {rate:.2f} img/s, ETA {remaining/60:.1f} min")

    out_df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    elapsed = time.time() - t0
    print(f"\nwrote {out} ({len(out_df)} rows in {elapsed:.0f}s = {elapsed/len(out_df):.2f}s/img)")
    for prompt in prompts:
        s = slug(prompt)
        n_detected = int(out_df[f"{s}_detected"].sum())
        mean_frac = float(out_df[f"{s}_pixel_frac"].mean())
        print(f"  {prompt!r}: detected in {n_detected}/{len(out_df)} ({100*n_detected/len(out_df):.1f}%), "
              f"mean pixel_frac = {mean_frac:.4f}")


if __name__ == "__main__":
    main()
