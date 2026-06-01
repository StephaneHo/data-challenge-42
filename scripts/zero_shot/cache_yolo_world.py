"""Cache YOLO-World open-vocabulary detection on val or test images.

YOLO-World (Ultralytics yolov8m-world or yolov8l-world) is a fast
open-vocabulary detector — you provide a list of class names as text and it
finds bounding boxes for each. Much faster than CLIPSeg/Florence-2 on CPU
(~100-300 ms/image with the small/medium variants).

For each image and each prompt, we save:
  - <prompt>_detected     : 1 if any box found
  - <prompt>_max_conf     : highest confidence among the boxes
  - <prompt>_n_boxes      : number of boxes found
  - <prompt>_total_area_frac : sum of all box areas / image area (proxy for object size)
  - <prompt>_bbox_*       : tightest enclosing box of all detections (normalized [0, 1])

Usage:
    # Sanity test
    python scripts/zero_shot/cache_yolo_world.py --prompts hat "face mask" hand sunglasses \\
        --source val --limit 20 --out eval/cache/yoloworld_test20.csv

    # On the 823 extreme cases first (smart targeting)
    python scripts/zero_shot/cache_yolo_world.py --prompts hat "face mask" hand sunglasses \\
        --source val --filter-extreme --out eval/cache/val_yoloworld_extreme.csv

    # Full val (recommended after positive signal on extreme)
    python scripts/zero_shot/cache_yolo_world.py --prompts hat "face mask" hand sunglasses \\
        --source val --out eval/cache/val_yoloworld.csv
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="yolov8m-world.pt",
                   choices=["yolov8s-world.pt", "yolov8m-world.pt",
                            "yolov8l-world.pt", "yolov8x-worldv2.pt"],
                   help="YOLO-World variant (s ~22M, m ~52M, l ~92M, x ~145M params)")
    p.add_argument("--conf", type=float, default=0.05,
                   help="Confidence threshold for detection (default 0.05 = permissive)")
    p.add_argument("--filter-extreme", action="store_true",
                   help="Only process Julien's extreme cases (pred > 0.7)")
    return p.parse_args()


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

    print(f"loading {args.model}...")
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.set_classes(args.prompts)
    print(f"classes set to: {args.prompts}")

    rows = []
    image_dir = Path(args.image_dir)
    t0 = time.time()
    for i, fn in enumerate(tqdm(df["filename"].tolist(), desc="yolo-world")):
        img_path = str(image_dir / fn)
        result = model.predict(img_path, conf=args.conf, verbose=False)[0]
        # result.boxes has xyxy (in pixels), conf, cls
        h, w = result.orig_shape
        row = {"filename": fn}

        boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else np.zeros((0, 4))
        confs = result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else np.zeros((0,))
        classes = result.boxes.cls.cpu().numpy().astype(int) if len(result.boxes) > 0 else np.zeros((0,), dtype=int)

        for cls_idx, prompt in enumerate(args.prompts):
            s = slug(prompt)
            mask = classes == cls_idx
            cls_boxes = boxes[mask]
            cls_confs = confs[mask]

            row[f"{s}_n_boxes"] = int(mask.sum())
            row[f"{s}_detected"] = int(mask.sum() > 0)
            row[f"{s}_max_conf"] = float(cls_confs.max()) if mask.sum() > 0 else 0.0

            if mask.sum() > 0:
                # Total area of all boxes (with overlap counted multiple times, but ok for proxy)
                areas = (cls_boxes[:, 2] - cls_boxes[:, 0]) * (cls_boxes[:, 3] - cls_boxes[:, 1])
                row[f"{s}_total_area_frac"] = float(areas.sum()) / (h * w)
                # Enclosing bbox of all class detections
                row[f"{s}_bbox_x0"] = float(cls_boxes[:, 0].min()) / w
                row[f"{s}_bbox_y0"] = float(cls_boxes[:, 1].min()) / h
                row[f"{s}_bbox_x1"] = float(cls_boxes[:, 2].max()) / w
                row[f"{s}_bbox_y1"] = float(cls_boxes[:, 3].max()) / h
            else:
                row[f"{s}_total_area_frac"] = 0.0
                row[f"{s}_bbox_x0"] = 0.0
                row[f"{s}_bbox_y0"] = 0.0
                row[f"{s}_bbox_x1"] = 0.0
                row[f"{s}_bbox_y1"] = 0.0

        rows.append(row)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df) - i - 1) / rate
            print(f"  [{i+1}/{len(df)}] {rate:.2f} img/s, ETA {eta/60:.1f} min")

    out_df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    elapsed = time.time() - t0
    print(f"\nwrote {out} ({len(out_df)} rows in {elapsed:.0f}s = {elapsed/len(out_df):.3f}s/img)")
    for prompt in args.prompts:
        s = slug(prompt)
        n_det = int(out_df[f"{s}_detected"].sum())
        mean_conf = out_df.loc[out_df[f"{s}_detected"] == 1, f"{s}_max_conf"].mean() if n_det > 0 else 0
        print(f"  {prompt!r}: detected in {n_det}/{len(out_df)} ({100*n_det/len(out_df):.1f}%), "
              f"mean_conf (when detected) = {mean_conf:.3f}")


if __name__ == "__main__":
    main()
