"""Cache pipeline v10 ETENDU avec features par classe (hat, hair, other).

Output CSV columns (PAR IMAGE) :
  filename, target, gender, face_detected, mask_3d_area,
  # Existant (deja dans val_v10_bg.csv) :
  skin_bi_in_mask, bg_bi_in_mask,
  skin_sf_in_mask, bg_sf_in_mask,
  r_3D_Bi, r_3D_Bi_bg, r_3D_Sf, r_3D_Sf_bg,
  # NOUVEAU (pour calibration par feature) :
  hat_bi_in_mask, hair_bi_in_mask, other_bi_in_mask,
  hat_sf_in_mask, hair_sf_in_mask, other_sf_in_mask,

Conventions :
  - "in_mask" signifie "intersection avec le masque 3DDFA, divise par l'aire du masque"
  - Pour CHAQUE modele : skin + bg + hair + hat + other = 1 (par construction)
  - "other" inclut : eye_g (glasses), ear_r (earrings), neck, neck_l, cloth

Usage:
    python scripts/zero_shot/cache_v10_features.py --source val --out eval/cache/val_v10_features.csv
    python scripts/zero_shot/cache_v10_features.py --source test --out eval/cache/test_v10_features.csv
    python scripts/zero_shot/cache_v10_features.py --source val --limit 28 --out eval/cache/smoke.csv
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "3DDFA_V2"))
sys.path.insert(0, str(REPO_ROOT / "face-parsing.PyTorch"))

warnings.filterwarnings("ignore")

# =====================================================
# CLASSE INDICES - VERIFIES AVEC LES MODELES (cf. audit)
# =====================================================
# BiSeNet (CelebAMask-HQ standard)
#   0=bg, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye, 6=eye_g (glasses),
#   7=l_ear, 8=r_ear, 9=ear_r, 10=nose, 11=mouth, 12=u_lip, 13=l_lip,
#   14=neck, 15=neck_l, 16=cloth, 17=hair, 18=hat
BI_SKIN = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]  # visible face parts
BI_BG = [0]
BI_HAIR = [17]
BI_HAT = [18]
# BI_OTHER = tout le reste = [6, 9, 14, 15, 16] (eye_g, ear_r, neck, neck_l, cloth)

# SegFormer (jonathandinu/face-parsing - DIFFERENT INDICES!)
#   0=bg, 1=skin, 2=nose, 3=eye_g, 4=l_eye, 5=r_eye, 6=l_brow, 7=r_brow,
#   8=l_ear, 9=r_ear, 10=mouth, 11=u_lip, 12=l_lip, 13=hair, 14=hat,
#   15=ear_r, 16=neck_l, 17=neck, 18=cloth
SF_SKIN = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # visible face parts
SF_BG = [0]
SF_HAIR = [13]
SF_HAT = [14]
# SF_OTHER = tout le reste = [3, 15, 16, 17, 18] (eye_g, ear_r, neck_l, neck, cloth)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--source", choices=["val", "test"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=500)
    return p.parse_args()


def setup_models():
    print("loading InsightFace (detection only)...")
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

    print("loading BiSeNet (face-parsing.PyTorch)...")
    from model import BiSeNet
    net_bi = BiSeNet(n_classes=19)
    net_bi.load_state_dict(
        torch.load(REPO_ROOT / "weights/79999_iter.pth", map_location="cpu", weights_only=True)
    )
    net_bi.eval()

    print("loading SegFormer (jonathandinu/face-parsing)...")
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    sf = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").eval()
    sf_proc = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return app, tddfa, net_bi, sf, sf_proc, to_tensor


def NAN_RESULT():
    return {
        "face_detected": 0, "mask_3d_area": 0,
        "skin_bi_in_mask": float("nan"), "bg_bi_in_mask": float("nan"),
        "hat_bi_in_mask": float("nan"), "hair_bi_in_mask": float("nan"),
        "other_bi_in_mask": float("nan"),
        "skin_sf_in_mask": float("nan"), "bg_sf_in_mask": float("nan"),
        "hat_sf_in_mask": float("nan"), "hair_sf_in_mask": float("nan"),
        "other_sf_in_mask": float("nan"),
        "r_3D_Bi": float("nan"), "r_3D_Bi_bg": float("nan"),
        "r_3D_Sf": float("nan"), "r_3D_Sf_bg": float("nan"),
    }


def compute(img_bgr, app, tddfa, net_bi, sf, sf_proc, to_tensor):
    h, w = img_bgr.shape[:2]

    faces = app.get(img_bgr)
    if not faces:
        return NAN_RESULT()
    bbox = faces[0].bbox

    # 3DDFA mask
    try:
        param_lst, roi_box_lst = tddfa(img_bgr, [bbox])
        ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
        pts = ver[:2, :].T.astype(np.int32)
        hull = cv2.convexHull(pts)
        mask_3d = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask_3d, hull, 1)
    except Exception:
        return NAN_RESULT()
    mask_area = int(mask_3d.sum())
    if mask_area == 0:
        return NAN_RESULT()

    # BiSeNet @ 512x512 (fix applique)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_512 = Image.fromarray(img_rgb).resize((512, 512), Image.BILINEAR)
    with torch.inference_mode():
        out_bi = net_bi(to_tensor(pil_512).unsqueeze(0))[0]
    parsing_bi_512 = out_bi.squeeze(0).numpy().argmax(0).astype(np.uint8)
    parsing_bi = cv2.resize(parsing_bi_512, (w, h), interpolation=cv2.INTER_NEAREST)

    # Fractions BiSeNet
    skin_bi = np.isin(parsing_bi, BI_SKIN).astype(np.uint8)
    bg_bi = np.isin(parsing_bi, BI_BG).astype(np.uint8)
    hat_bi = np.isin(parsing_bi, BI_HAT).astype(np.uint8)
    hair_bi = np.isin(parsing_bi, BI_HAIR).astype(np.uint8)

    skin_bi_in = float(int((skin_bi & mask_3d).sum()) / mask_area)
    bg_bi_in = float(int((bg_bi & mask_3d).sum()) / mask_area)
    hat_bi_in = float(int((hat_bi & mask_3d).sum()) / mask_area)
    hair_bi_in = float(int((hair_bi & mask_3d).sum()) / mask_area)
    other_bi_in = max(0.0, 1.0 - skin_bi_in - bg_bi_in - hat_bi_in - hair_bi_in)

    # SegFormer
    inputs = sf_proc(images=Image.fromarray(img_rgb), return_tensors="pt")
    with torch.inference_mode():
        sf_out = sf(**inputs)
    seg_sf = torch.nn.functional.interpolate(
        sf_out.logits, size=(h, w), mode="bilinear", align_corners=False
    ).argmax(1).squeeze(0).numpy().astype(np.uint8)

    # Fractions SegFormer
    skin_sf = np.isin(seg_sf, SF_SKIN).astype(np.uint8)
    bg_sf = np.isin(seg_sf, SF_BG).astype(np.uint8)
    hat_sf = np.isin(seg_sf, SF_HAT).astype(np.uint8)
    hair_sf = np.isin(seg_sf, SF_HAIR).astype(np.uint8)

    skin_sf_in = float(int((skin_sf & mask_3d).sum()) / mask_area)
    bg_sf_in = float(int((bg_sf & mask_3d).sum()) / mask_area)
    hat_sf_in = float(int((hat_sf & mask_3d).sum()) / mask_area)
    hair_sf_in = float(int((hair_sf & mask_3d).sum()) / mask_area)
    other_sf_in = max(0.0, 1.0 - skin_sf_in - bg_sf_in - hat_sf_in - hair_sf_in)

    # Ratios (compatibilite avec cache existant)
    r_3D_Bi = 1.0 - skin_bi_in
    r_3D_Bi_bg = 1.0 - skin_bi_in - bg_bi_in
    r_3D_Sf = 1.0 - skin_sf_in
    r_3D_Sf_bg = 1.0 - skin_sf_in - bg_sf_in

    return {
        "face_detected": 1,
        "mask_3d_area": mask_area,
        "skin_bi_in_mask": skin_bi_in,
        "bg_bi_in_mask": bg_bi_in,
        "hat_bi_in_mask": hat_bi_in,
        "hair_bi_in_mask": hair_bi_in,
        "other_bi_in_mask": other_bi_in,
        "skin_sf_in_mask": skin_sf_in,
        "bg_sf_in_mask": bg_sf_in,
        "hat_sf_in_mask": hat_sf_in,
        "hair_sf_in_mask": hair_sf_in,
        "other_sf_in_mask": other_sf_in,
        "r_3D_Bi": r_3D_Bi,
        "r_3D_Bi_bg": r_3D_Bi_bg,
        "r_3D_Sf": r_3D_Sf,
        "r_3D_Sf_bg": r_3D_Sf_bg,
    }


def main():
    args = parse_args()

    if args.source == "val":
        from src.data import stratified_split
        train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
        _, df = stratified_split(train_csv, val_frac=0.15, seed=42)
        df = df.rename(columns={"FaceOcclusion": "target"})
    else:
        df = pd.read_csv(Path(args.data_dir) / "test_students.csv")
        df["target"] = float("nan")
        df["gender"] = float("nan")

    if args.limit > 0:
        df = df.head(args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()
    if args.resume and out_path.exists():
        existing = pd.read_csv(out_path)
        rows = existing.to_dict("records")
        seen = set(existing.filename)
        print(f"RESUME: skipping {len(seen)} already-processed")
        df = df[~df.filename.isin(seen)].reset_index(drop=True)

    print(f"will process {len(df)} {args.source} images")
    if len(df) == 0:
        print("nothing to do.")
        return

    app, tddfa, net_bi, sf, sf_proc, to_tensor = setup_models()

    t0 = time.time()
    image_dir = Path(args.image_dir)
    n_err = 0
    for i, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="v10-features")):
        img_path = image_dir / row.filename
        img = cv2.imread(str(img_path))
        if img is None:
            n_err += 1
            d = NAN_RESULT()
        else:
            try:
                d = compute(img, app, tddfa, net_bi, sf, sf_proc, to_tensor)
            except Exception as e:
                n_err += 1
                print(f"\nERROR on {row.filename}: {e}")
                continue
        d["filename"] = row.filename
        d["target"] = float(row.target) if not pd.isna(row.target) else float("nan")
        d["gender"] = float(row.gender) if not pd.isna(row.gender) else float("nan")
        rows.append(d)

        if (i + 1) % args.save_every == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df) - i - 1) / rate
            print(f"\n  [{i+1}/{len(df)}] checkpoint, {rate:.2f} img/s, "
                  f"ETA {eta/3600:.1f}h, errs={n_err}")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    elapsed = time.time() - t0
    print(f"\nDONE -- {out_path} ({len(rows)} rows in {elapsed/60:.1f} min, errs={n_err})")


if __name__ == "__main__":
    main()
