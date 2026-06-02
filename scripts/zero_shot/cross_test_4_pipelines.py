"""Julien's 2x2 factorial cross-test: 4 pipelines comparing mask source x skin source.

For each image, we compute the 4 occlusion ratios:
  1. r_3D_Bi:    3DDFA mask  -  BiSeNet skin            (= Julien's original)
  2. r_3D_Sf:    3DDFA mask  -  SegFormer skin          (NEW)
  3. r_Bi_Cv:    BiSeNet-hull -  BiSeNet skin           (NEW)
  4. r_Sf_Cv:    SegFormer-hull - SegFormer skin        (= our heuristic core)

The formula is:    ratio = 1 - (skin_pixels INSIDE mask) / mask_area
   where mask = the theoretical face region.

For ratios 2 and 4 we also save the raw "before power 0.7 + TTA" form.

Stratified sample of N images by target bin x gender so we get good coverage.
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
import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "3DDFA_V2"))
sys.path.insert(0, str(REPO_ROOT / "face-parsing.PyTorch"))

from src.data import stratified_split  # noqa: E402

# BiSeNet (face-parsing.PyTorch / CelebAMask-HQ) — visible face classes.
# 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye, 6=eye_g, 7=l_ear, 8=r_ear,
# 9=ear_r, 10=nose, 11=mouth, 12=u_lip, 13=l_lip, 14=neck, 15=neck_l,
# 16=cloth, 17=hair, 18=hat
BISENET_VISIBLE = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]

# SegFormer (jonathandinu/face-parsing) — DIFFERENT class indices!
# 1=skin, 2=nose, 3=eye_g(GLASSES), 4=l_eye, 5=r_eye, 6=l_brow, 7=r_brow,
# 8=l_ear, 9=r_ear, 10=mouth, 11=u_lip, 12=l_lip, 13=hair, 14=hat,
# 15=ear_r, 16=neck_l, 17=neck, 18=cloth
SEGFORMER_VISIBLE = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# skin + nose + l_eye + r_eye + l_brow + r_brow + l_ear + r_ear + mouth + u_lip + l_lip
# Excludes: 0(bg), 3(eye_g=glasses!), 13(hair), 14(hat), 15(ear_r), 16(neck_l), 17(neck), 18(cloth)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--out", required=True)
    p.add_argument("--n-per-bin", type=int, default=50,
                   help="Samples per target bin (7 bins x 2 genders = 14 groups)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_models():
    print("loading InsightFace...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                       allowed_modules=["detection"])
    app.prepare(ctx_id=0, det_size=(224, 224))

    print("loading 3DDFA-V2...")
    from TDDFA import TDDFA
    cfg = yaml.load(open(REPO_ROOT / "3DDFA_V2/configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
    cfg["checkpoint_fp"] = str(REPO_ROOT / "3DDFA_V2/weights/mb1_120x120.pth")
    cfg["bfm_fp"] = str(REPO_ROOT / "3DDFA_V2/configs/bfm_noneck_v3.pkl")
    cfg["param_mean_std_fp"] = str(REPO_ROOT / "3DDFA_V2/configs/param_mean_std_62d_120x120.pkl")
    tddfa = TDDFA(**cfg)

    print("loading BiSeNet face-parsing...")
    from model import BiSeNet
    net_bi = BiSeNet(n_classes=19)
    net_bi.load_state_dict(
        torch.load(REPO_ROOT / "weights/79999_iter.pth", map_location="cpu", weights_only=True)
    )
    net_bi.eval()

    print("loading SegFormer face-parsing...")
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    sf_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").eval()
    sf_proc = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return app, tddfa, net_bi, sf_model, sf_proc, to_tensor


def convex_hull_mask(face_mask, image_shape):
    """Return a filled binary mask = convex hull of `face_mask` (binary HxW)."""
    h, w = image_shape
    ys, xs = np.where(face_mask > 0)
    if len(xs) < 4:
        return np.zeros((h, w), dtype=np.uint8)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


def compute_4_pipelines(img_bgr, app, tddfa, net_bi, sf_model, sf_proc, to_tensor):
    """Run all 3 models, compute 4 ratios.

    Returns dict with:
       r_3D_Bi, r_3D_Sf, r_Bi_Cv, r_Sf_Cv
       plus diagnostics: mask_3d_frac, mask_bi_hull_frac, mask_sf_hull_frac,
                          skin_bi_frac, skin_sf_frac
    """
    h, w = img_bgr.shape[:2]
    out = {k: float("nan") for k in [
        "r_3D_Bi", "r_3D_Sf", "r_Bi_Cv", "r_Sf_Cv",
        "mask_3d_frac", "mask_bi_hull_frac", "mask_sf_hull_frac",
        "skin_bi_frac", "skin_sf_frac",
        "iou_mask_3d_bi", "iou_mask_3d_sf", "iou_skin_bi_sf",
    ]}

    # === Face detection (shared) ===
    faces = app.get(img_bgr)
    if not faces:
        return out
    bbox = faces[0].bbox

    # === 3DDFA mask ===
    boxes = [bbox]
    try:
        param_lst, roi_box_lst = tddfa(img_bgr, boxes)
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
        pts = ver_lst[0][:2, :].T.astype(np.int32)
        hull = cv2.convexHull(pts)
        mask_3d = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask_3d, hull, 1)
    except Exception:
        mask_3d = np.zeros((h, w), dtype=np.uint8)

    out["mask_3d_frac"] = float(mask_3d.sum()) / (h * w)

    # === BiSeNet parsing ===
    # IMPORTANT: BiSeNet (face-parsing.PyTorch) needs 512x512 input, per the
    # reference test.py. Without resize on 224x224 crops, the model misses many
    # face parts (no eyes/brows/mouth detected). Resize → infer → resize back.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_512 = Image.fromarray(img_rgb).resize((512, 512), Image.BILINEAR)
    input_tensor = to_tensor(pil_512).unsqueeze(0)
    with torch.inference_mode():
        bi_out = net_bi(input_tensor)[0]
    parsing_bi_512 = bi_out.squeeze(0).cpu().numpy().argmax(0).astype(np.uint8)
    parsing_bi = cv2.resize(parsing_bi_512, (w, h), interpolation=cv2.INTER_NEAREST)
    skin_bi = np.isin(parsing_bi, BISENET_VISIBLE).astype(np.uint8)
    out["skin_bi_frac"] = float(skin_bi.sum()) / (h * w)

    # === SegFormer parsing ===
    pil_img = Image.fromarray(img_rgb)
    inputs = sf_proc(images=pil_img, return_tensors="pt")
    with torch.inference_mode():
        sf_out = sf_model(**inputs)
    seg_sf = torch.nn.functional.interpolate(
        sf_out.logits, size=(h, w), mode="bilinear", align_corners=False
    ).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    skin_sf = np.isin(seg_sf, SEGFORMER_VISIBLE).astype(np.uint8)
    out["skin_sf_frac"] = float(skin_sf.sum()) / (h * w)

    # === Hulls from parsings ===
    mask_bi_hull = convex_hull_mask(skin_bi, (h, w))
    mask_sf_hull = convex_hull_mask(skin_sf, (h, w))
    out["mask_bi_hull_frac"] = float(mask_bi_hull.sum()) / (h * w)
    out["mask_sf_hull_frac"] = float(mask_sf_hull.sum()) / (h * w)

    # === The 4 ratios ===
    # 1. r_3D_Bi : 3DDFA mask, BiSeNet skin
    a = float(mask_3d.sum())
    if a > 0:
        out["r_3D_Bi"] = 1.0 - float((skin_bi & mask_3d).sum()) / a
    # 2. r_3D_Sf : 3DDFA mask, SegFormer skin
    if a > 0:
        out["r_3D_Sf"] = 1.0 - float((skin_sf & mask_3d).sum()) / a
    # 3. r_Bi_Cv : BiSeNet hull, BiSeNet skin
    a = float(mask_bi_hull.sum())
    if a > 0:
        out["r_Bi_Cv"] = 1.0 - float((skin_bi & mask_bi_hull).sum()) / a
    # 4. r_Sf_Cv : SegFormer hull, SegFormer skin
    a = float(mask_sf_hull.sum())
    if a > 0:
        out["r_Sf_Cv"] = 1.0 - float((skin_sf & mask_sf_hull).sum()) / a

    # IoU diagnostics
    def iou(a, b):
        inter = float((a & b).sum())
        union = float((a | b).sum())
        return inter / union if union > 0 else 0.0

    out["iou_mask_3d_bi"] = iou(mask_3d.astype(bool), mask_bi_hull.astype(bool))
    out["iou_mask_3d_sf"] = iou(mask_3d.astype(bool), mask_sf_hull.astype(bool))
    out["iou_skin_bi_sf"] = iou(skin_bi.astype(bool), skin_sf.astype(bool))

    return out


def stratified_sample(df, n_per_bin, seed=42):
    """Pick ~n_per_bin samples per (target_bin × gender)."""
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
    df = df.copy()
    df["bin"] = pd.cut(df["FaceOcclusion"], bins=bins, labels=False, right=False)
    df["bin"] = df["bin"].fillna(0).astype(int)
    rng = np.random.default_rng(seed)
    out = []
    for (b, g), grp in df.groupby(["bin", "gender"]):
        n = min(n_per_bin, len(grp))
        idx = rng.choice(len(grp), n, replace=False)
        out.append(grp.iloc[idx])
    return pd.concat(out).reset_index(drop=True)


def main():
    args = parse_args()

    train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
    _, val_df = stratified_split(train_csv, val_frac=0.15, seed=42)

    sample = stratified_sample(val_df, n_per_bin=args.n_per_bin, seed=args.seed)
    print(f"sampled {len(sample)} val images")
    print("Distribution:")
    print(sample.groupby(["bin", "gender"]).size().unstack())

    app, tddfa, net_bi, sf_model, sf_proc, to_tensor = setup_models()

    rows = []
    image_dir = Path(args.image_dir)
    t0 = time.time()
    n_err = 0
    for i, (fn, target, gender) in enumerate(
        tqdm(
            list(zip(sample["filename"], sample["FaceOcclusion"], sample["gender"])),
            desc="4-pipelines",
        )
    ):
        img = cv2.imread(str(image_dir / fn))
        if img is None:
            n_err += 1
            continue
        try:
            d = compute_4_pipelines(img, app, tddfa, net_bi, sf_model, sf_proc, to_tensor)
        except Exception as e:
            n_err += 1
            print(f"\nfailed {fn}: {e}")
            continue
        d["filename"] = fn
        d["target"] = float(target)
        d["gender"] = float(gender)
        rows.append(d)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(sample) - i - 1) / rate
            print(f"  [{i+1}/{len(sample)}] {rate:.2f} img/s ETA {eta/60:.1f}min errs={n_err}")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    elapsed = time.time() - t0
    print(f"\nwrote {args.out} ({len(df)} rows in {elapsed/60:.1f} min, errs={n_err})")


if __name__ == "__main__":
    main()
