"""Test alternative occlusion definitions that handle the hair-vs-shadow ambiguity.

On a stratified sample of 500 val images, computes:
  r_current = 1 - skin_inside_mask / mask         (= v9)
  r_no_hair  = 1 - (skin + hair) inside mask / mask  (= treat hair as visible)
  r_half_hair = 1 - (skin + 0.5 * hair) inside mask / mask  (= ambiguous hair)
  r_only_explicit = (eye_g + hat + cloth) inside mask / mask  (= only declared occluders)

Compares scores under brief / val natif distributions, with per-gender best cal.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "3DDFA_V2"))

warnings.filterwarnings("ignore")

# SegFormer (jonathandinu/face-parsing) class indices
SF_SKIN_VISIBLE = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # skin+nose+eyes+brows+ears+mouth+lips
SF_HAIR = [13]
SF_EXPLICIT_OCC = [3, 14, 18]  # eye_g, hat, cloth


def setup():
    from insightface.app import FaceAnalysis
    from TDDFA import TDDFA
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                       allowed_modules=["detection"])
    app.prepare(ctx_id=0, det_size=(224, 224))

    cfg = yaml.load(open(REPO_ROOT / "3DDFA_V2/configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
    cfg["checkpoint_fp"] = str(REPO_ROOT / "3DDFA_V2/weights/mb1_120x120.pth")
    cfg["bfm_fp"] = str(REPO_ROOT / "3DDFA_V2/configs/bfm_noneck_v3.pkl")
    cfg["param_mean_std_fp"] = str(REPO_ROOT / "3DDFA_V2/configs/param_mean_std_62d_120x120.pkl")
    tddfa = TDDFA(**cfg)

    sf = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").eval()
    sf_proc = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")

    return app, tddfa, sf, sf_proc


def compute(img_bgr, app, tddfa, sf, sf_proc):
    h, w = img_bgr.shape[:2]
    out = {}
    faces = app.get(img_bgr)
    if not faces:
        return {"face_detected": 0}
    bbox = faces[0].bbox
    param_lst, roi_box_lst = tddfa(img_bgr, [bbox])
    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
    pts = ver[:2, :].T.astype(np.int32)
    hull = cv2.convexHull(pts)
    mask_3d = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask_3d, hull, 1)
    a3d = max(int(mask_3d.sum()), 1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    inputs = sf_proc(images=pil, return_tensors="pt")
    with torch.inference_mode():
        sf_out = sf(**inputs)
    seg_sf = torch.nn.functional.interpolate(
        sf_out.logits, size=(h, w), mode="bilinear", align_corners=False
    ).argmax(1).squeeze(0).numpy().astype(np.uint8)

    skin_in_mask = float(((np.isin(seg_sf, SF_SKIN_VISIBLE).astype(np.uint8)) & mask_3d).sum())
    hair_in_mask = float(((np.isin(seg_sf, SF_HAIR).astype(np.uint8)) & mask_3d).sum())
    explicit_occ_in_mask = float(((np.isin(seg_sf, SF_EXPLICIT_OCC).astype(np.uint8)) & mask_3d).sum())

    out["mask_3d_area"] = a3d
    out["skin_in_mask_frac"] = skin_in_mask / a3d
    out["hair_in_mask_frac"] = hair_in_mask / a3d
    out["explicit_occ_in_mask_frac"] = explicit_occ_in_mask / a3d

    # 4 alternative ratios
    out["r_current"] = 1.0 - skin_in_mask / a3d
    out["r_no_hair"] = 1.0 - (skin_in_mask + hair_in_mask) / a3d
    out["r_half_hair"] = 1.0 - (skin_in_mask + 0.5 * hair_in_mask) / a3d
    out["r_only_explicit"] = explicit_occ_in_mask / a3d

    return out


def main():
    print("Loading models...")
    app, tddfa, sf, sf_proc = setup()

    val = pd.read_csv(REPO_ROOT / "eval" / "val_julien_baseline.csv").rename(columns={"pred": "pj"})
    # Stratified 500 images
    np.random.seed(42)
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
    val["bin"] = pd.cut(val.target, bins=bins, right=False, labels=False).fillna(0).astype(int)
    rng = np.random.default_rng(42)
    samples = []
    for (b, g), grp in val.groupby(["bin", "gender"]):
        n = min(50, len(grp))  # 50 per bin × gender = up to 700 total
        idx = rng.choice(len(grp), n, replace=False)
        samples.append(grp.iloc[idx])
    samples = pd.concat(samples).reset_index(drop=True)
    print(f"Sample size: {len(samples)} images")

    rows = []
    t0 = time.time()
    image_dir = REPO_ROOT / "crops"
    for i, row in enumerate(tqdm(samples.itertuples(), total=len(samples), desc="alt-defs")):
        img = cv2.imread(str(image_dir / row.filename))
        if img is None:
            continue
        try:
            d = compute(img, app, tddfa, sf, sf_proc)
        except Exception as e:
            print(f"\nfailed {row.filename}: {e}")
            continue
        d["filename"] = row.filename
        d["target"] = float(row.target)
        d["gender"] = float(row.gender)
        rows.append(d)

    df = pd.DataFrame(rows)
    out_path = REPO_ROOT / "eval" / "cache" / "alt_hair_definitions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows in {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
