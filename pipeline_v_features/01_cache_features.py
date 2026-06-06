"""Etape 1 : Cache des features par image (val ou test).

Pour chaque image, calcule 6 fractions PAR MODELE (BiSeNet + SegFormer) :
  skin, bg, hat, hair, other  (la somme des 5 = 1.0 par construction)

Plus les 4 ratios derives :
  r_3D_Bi    = 1 - skin_bi_in_mask                  (notre def, bg = occlusion)
  r_3D_Bi_bg = 1 - skin_bi_in_mask - bg_bi_in_mask  (def Julien, bg = visible)
  r_3D_Sf    = 1 - skin_sf_in_mask                  (notre def)
  r_3D_Sf_bg = 1 - skin_sf_in_mask - bg_sf_in_mask  (def Julien)

POUR INTEGRATION DANS LE NOTEBOOK JULIEN :
  La fonction `compute_features_for_image()` est la fonction cle.
  Elle prend une image BGR et les modeles deja charges, retourne un dict des
  features. Julien peut la copier-coller telle quelle.

Usage CLI :
    python 01_cache_features.py --source val --resume --out eval/cache/val_features.csv
    python 01_cache_features.py --source test --resume --out eval/cache/test_features.csv
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "3DDFA_V2"))
sys.path.insert(0, str(REPO_ROOT / "face-parsing.PyTorch"))

warnings.filterwarnings("ignore")


# =====================================================================
# INDICES DE CLASSES (verifies via model.config)
# =====================================================================
# ATTENTION : BiSeNet et SegFormer utilisent des indices DIFFERENTS !

# BiSeNet (CelebAMask-HQ standard) :
#   0=bg, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye, 6=eye_g (glasses),
#   7=l_ear, 8=r_ear, 9=ear_r (earring), 10=nose, 11=mouth, 12=u_lip, 13=l_lip,
#   14=neck, 15=neck_l, 16=cloth, 17=hair, 18=hat
BI_SKIN = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]
BI_BG = [0]
BI_HAIR = [17]
BI_HAT = [18]

# SegFormer (jonathandinu/face-parsing) :
#   0=bg, 1=skin, 2=nose, 3=eye_g (glasses), 4=l_eye, 5=r_eye, 6=l_brow, 7=r_brow,
#   8=l_ear, 9=r_ear, 10=mouth, 11=u_lip, 12=l_lip, 13=hair, 14=hat,
#   15=ear_r, 16=neck_l, 17=neck, 18=cloth (clothes / masks)
SF_SKIN = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
SF_BG = [0]
SF_HAIR = [13]
SF_HAT = [14]


# =====================================================================
# CORRECTION DE MASQUE (JULIEN)
# =====================================================================
# Le masque 3DDFA brut a un decalage systematique vs le visage reel.
# Julien a empiriquement determine cette transformation affine qui ameliore
# l'alignement :
#   - Retrecit horizontalement (0.9x)
#   - Etire legerement verticalement (1.05x)
#   - Translate de (15, -10) pixels
MASK_SCALE_X = 0.9
MASK_SCALE_Y = 1.05
MASK_TX = 15
MASK_TY = -10


def apply_julien_mask_correction(mask_bin: np.ndarray, img_shape) -> np.ndarray:
    """Applique la correction affine de Julien au mask binaire 3DDFA."""
    M = np.array([
        [MASK_SCALE_X, 0, MASK_TX],
        [0, MASK_SCALE_Y, MASK_TY],
    ], dtype=np.float32)
    return cv2.warpAffine(mask_bin, M, dsize=img_shape[:2])


# =====================================================================
# CHARGEMENT DES MODELES (a faire une fois au debut)
# =====================================================================
def load_all_models():
    """Charge les 4 modeles et retourne un tuple (app, tddfa, net_bi, sf, sf_proc, to_tensor)."""
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


# =====================================================================
# *** FONCTION CLE : extraction des features pour UNE image ***
# Cette fonction peut etre copiee-collee dans le notebook Julien.
# =====================================================================
def compute_features_for_image(img_bgr, app, tddfa, net_bi, sf, sf_proc, to_tensor):
    """Extrait toutes les features v_features pour UNE image BGR.

    Etapes :
      1. Detection visage via InsightFace
      2. Mask theorique : 3DDFA -> convex hull -> correction Julien (warpAffine)
      3. BiSeNet @ 512x512 : segmentation 19 classes
      4. SegFormer @ 512x512 : segmentation 19 classes
      5. Calcul des fractions skin/bg/hair/hat/other pour chaque modele
      6. Calcul des 4 ratios r_3D_*

    Retourne un dict avec 16 cles, ou NaN_RESULT si la detection echoue.
    """
    h, w = img_bgr.shape[:2]

    # === 1. Detection visage ===
    faces = app.get(img_bgr)
    if not faces:
        return _nan_result()
    bbox = faces[0].bbox

    # === 2. Mask theorique : 3DDFA + correction Julien ===
    try:
        param_lst, roi_box_lst = tddfa(img_bgr, [bbox])
        vertices = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
        pts_2d = vertices[:2, :].T.astype(np.int32)
        hull = cv2.convexHull(pts_2d)
        mask_3d_raw = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask_3d_raw, hull, 1)
        # *** correction Julien ***
        mask_3d = apply_julien_mask_correction(mask_3d_raw, img_bgr.shape)
    except Exception:
        return _nan_result()
    mask_area = int(mask_3d.sum())
    if mask_area == 0:
        return _nan_result()

    # === 3. BiSeNet @ 512x512 ===
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_512 = Image.fromarray(img_rgb).resize((512, 512), Image.BILINEAR)
    with torch.inference_mode():
        out_bi = net_bi(to_tensor(pil_512).unsqueeze(0))[0]
    parsing_bi_512 = out_bi.squeeze(0).numpy().argmax(0).astype(np.uint8)
    parsing_bi = cv2.resize(parsing_bi_512, (w, h), interpolation=cv2.INTER_NEAREST)

    # === 4. SegFormer (processor gere le resize 512 en interne) ===
    inputs = sf_proc(images=Image.fromarray(img_rgb), return_tensors="pt")
    with torch.inference_mode():
        sf_out = sf(**inputs)
    seg_sf = torch.nn.functional.interpolate(
        sf_out.logits, size=(h, w), mode="bilinear", align_corners=False
    ).argmax(1).squeeze(0).numpy().astype(np.uint8)

    # === 5. Fractions par classe inside mask ===
    # On agrege les pixels par "classe d'interet" (skin, bg, hair, hat, other)
    # puis on divise par l'aire du mask.
    bi_fractions = _compute_class_fractions(parsing_bi, mask_3d, mask_area,
                                             BI_SKIN, BI_BG, BI_HAIR, BI_HAT)
    sf_fractions = _compute_class_fractions(seg_sf, mask_3d, mask_area,
                                             SF_SKIN, SF_BG, SF_HAIR, SF_HAT)

    # === 6. Ratios ===
    r_3D_Bi = 1.0 - bi_fractions["skin"]
    r_3D_Bi_bg = 1.0 - bi_fractions["skin"] - bi_fractions["bg"]
    r_3D_Sf = 1.0 - sf_fractions["skin"]
    r_3D_Sf_bg = 1.0 - sf_fractions["skin"] - sf_fractions["bg"]

    return {
        "face_detected": 1,
        "mask_3d_area": mask_area,
        # BiSeNet fractions
        "skin_bi_in_mask": bi_fractions["skin"],
        "bg_bi_in_mask": bi_fractions["bg"],
        "hat_bi_in_mask": bi_fractions["hat"],
        "hair_bi_in_mask": bi_fractions["hair"],
        "other_bi_in_mask": bi_fractions["other"],
        # SegFormer fractions
        "skin_sf_in_mask": sf_fractions["skin"],
        "bg_sf_in_mask": sf_fractions["bg"],
        "hat_sf_in_mask": sf_fractions["hat"],
        "hair_sf_in_mask": sf_fractions["hair"],
        "other_sf_in_mask": sf_fractions["other"],
        # Ratios (notation compatible avec v9, v10_hyb, v_features)
        "r_3D_Bi": r_3D_Bi,
        "r_3D_Bi_bg": r_3D_Bi_bg,
        "r_3D_Sf": r_3D_Sf,
        "r_3D_Sf_bg": r_3D_Sf_bg,
    }


def _compute_class_fractions(parsing, mask, mask_area, skin_idx, bg_idx, hair_idx, hat_idx):
    """Calcule les 5 fractions (skin/bg/hair/hat/other) inside mask.

    'other' = 1 - skin - bg - hair - hat (par construction la somme = 1).
    """
    skin_pix = np.isin(parsing, skin_idx) & mask.astype(bool)
    bg_pix = np.isin(parsing, bg_idx) & mask.astype(bool)
    hair_pix = np.isin(parsing, hair_idx) & mask.astype(bool)
    hat_pix = np.isin(parsing, hat_idx) & mask.astype(bool)

    skin_frac = float(skin_pix.sum()) / mask_area
    bg_frac = float(bg_pix.sum()) / mask_area
    hair_frac = float(hair_pix.sum()) / mask_area
    hat_frac = float(hat_pix.sum()) / mask_area
    other_frac = max(0.0, 1.0 - skin_frac - bg_frac - hair_frac - hat_frac)

    return {"skin": skin_frac, "bg": bg_frac, "hair": hair_frac,
            "hat": hat_frac, "other": other_frac}


def _nan_result():
    """Resultat NaN quand la detection echoue."""
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


# =====================================================================
# MAIN : iterer sur val ou test, sauver le cache
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--source", choices=["val", "test"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=500)
    args = p.parse_args()

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

    app, tddfa, net_bi, sf, sf_proc, to_tensor = load_all_models()

    t0 = time.time()
    image_dir = Path(args.image_dir)
    n_err = 0
    for i, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="v_features cache")):
        img_path = image_dir / row.filename
        img = cv2.imread(str(img_path))
        if img is None:
            n_err += 1
            features = _nan_result()
        else:
            try:
                features = compute_features_for_image(
                    img, app, tddfa, net_bi, sf, sf_proc, to_tensor
                )
            except Exception as e:
                n_err += 1
                print(f"\nERROR on {row.filename}: {e}")
                continue
        features["filename"] = row.filename
        features["target"] = float(row.target) if not pd.isna(row.target) else float("nan")
        features["gender"] = float(row.gender) if not pd.isna(row.gender) else float("nan")
        rows.append(features)

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
