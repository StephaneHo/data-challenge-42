"""Reproduit la pipeline du notebook DataChallengeJulien_TF_with_v2_fallback.ipynb
sur 10 images de validation pour verifier l'integration.

Ce script :
  1. Charge les memes modeles que le notebook
  2. Definit occlusion_computation() avec les 4 modifs du README (Etapes 1-4)
  3. Definit aussi occlusion_computation_julien_original() pour comparaison
  4. Selectionne 10 images : 5 "normales" + 5 "plante" (a partir de val_features.csv)
  5. Compare side-by-side les 2 versions
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "3DDFA_V2"))
sys.path.insert(0, str(REPO / "face-parsing.PyTorch"))
warnings.filterwarnings("ignore")

# Import du module v2 fallback (Etape 2)
from pipeline_julien_integration import apply_v2_fallback, detect_plante_regime

# ============================================================================
# Constantes (Cell 35 du notebook, avec ajouts Etape 1)
# ============================================================================
VISIBLE_FACE_CLASSES_B = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
VISIBLE_FACE_CLASSES_S = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
HAIR_B = [17]
HAT_B = [18]
HAIR_S = [13]
HAT_S = [14]

# Etape 1 - SKIN pur pour detection plante
SKIN_ONLY_B = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]
SKIN_ONLY_S = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
BG_B        = [0]

F_WEIGHTS = {"hair_bi": 0.376, "hat_bi": 0.425, "other_bi": 0.478,
             "hair_sf": 0.619, "hat_sf": 0.902, "other_bg_sf": 1.087}
M_WEIGHTS = {"hair_bi": 0.489, "hat_bi": 0.294, "other_bi": 0.210,
             "hair_sf": 0.484, "hat_sf": 0.609, "other_bg_sf": 0.382}

from torchvision import transforms
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

device = "cpu"


# ============================================================================
# Chargement des modeles (cells 30-33 du notebook)
# ============================================================================
def setup_models():
    print("Loading models...", flush=True)
    # InsightFace
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(224, 224))
    print("  InsightFace OK", flush=True)

    # 3DDFA-V2
    from TDDFA import TDDFA
    cfg = yaml.load(open(REPO / "3DDFA_V2/configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
    cfg["checkpoint_fp"] = str(REPO / "3DDFA_V2/weights/mb1_120x120.pth")
    cfg["bfm_fp"] = str(REPO / "3DDFA_V2/configs/bfm_noneck_v3.pkl")
    cfg["param_mean_std_fp"] = str(REPO / "3DDFA_V2/configs/param_mean_std_62d_120x120.pkl")
    tddfa = TDDFA(**cfg)
    print("  3DDFA-V2 OK", flush=True)

    # BiSeNet
    from model import BiSeNet
    net = BiSeNet(n_classes=19)
    net.load_state_dict(torch.load(REPO / "weights/79999_iter.pth", map_location="cpu", weights_only=True))
    net.eval()
    print("  BiSeNet OK", flush=True)

    # SegFormer
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    seg_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").eval()
    seg_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    print("  SegFormer OK", flush=True)

    return app, tddfa, net, seg_model, seg_processor


# ============================================================================
# Pipeline complete (reproduction du Cell 37 du notebook AVEC modifs Etapes 3 & 4)
# ============================================================================
def occlusion_computation(app, tddfa, net, seg_model, seg_processor, img):
    """Reproduit occlusion_computation() du notebook AVEC le v2 fallback."""
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    faces = app.get(img)
    if not faces:
        return None
    face = faces[0]
    bbox = face.bbox
    pred_gender = face.sex

    # 3DDFA
    param_lst, roi_box_lst = tddfa(img, [bbox])
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

    mask_theoretical = np.zeros(img.shape[:2], dtype=np.uint8)
    pts = ver_lst[0][:2, :].T.astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask_theoretical, hull, 1)

    # Correction mask
    M = np.array([[0.9, 0, 15], [0, 1.05, -10]], dtype=np.float32)
    mask_theoretical = cv2.warpAffine(mask_theoretical, M, dsize=img.shape[:2])
    total_pixels = float(np.sum(mask_theoretical))
    if total_pixels == 0:
        return None

    # BiSeNet
    input_tensor = to_tensor(img).unsqueeze(0).to(device)
    out = net(input_tensor)[0]
    parsing_b = out.squeeze(0).cpu().numpy().argmax(0)

    visible_skin_mask_b = np.isin(parsing_b, VISIBLE_FACE_CLASSES_B).astype(np.uint8)
    hat_b = np.isin(parsing_b, HAT_B).astype(np.uint8)
    hair_b = np.isin(parsing_b, HAIR_B).astype(np.uint8)

    visible_pixels_b = np.sum(visible_skin_mask_b & mask_theoretical)
    hair_b_in_mask = np.sum(hair_b & mask_theoretical)
    hat_b_in_mask = np.sum(hat_b & mask_theoretical)
    other_b_in_mask = max(0.0, total_pixels - (float(visible_pixels_b) + float(hair_b_in_mask) + float(hat_b_in_mask)))
    hair_ratio_b = hair_b_in_mask / total_pixels
    hat_ratio_b = hat_b_in_mask / total_pixels
    other_ratio_b = other_b_in_mask / total_pixels

    # SegFormer
    inputs = seg_processor(images=img, return_tensors="pt").to(device)
    outputs = seg_model(**inputs)
    logits = outputs.logits
    h, w = img.shape[:2]
    upsampled_logits = nn.functional.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
    labels = upsampled_logits.argmax(dim=1)[0]
    parsing_s = labels.cpu().numpy()

    visible_skin_mask_s = np.isin(parsing_s, VISIBLE_FACE_CLASSES_S).astype(np.uint8)
    hat_s = np.isin(parsing_s, HAT_S).astype(np.uint8)
    hair_s = np.isin(parsing_s, HAIR_S).astype(np.uint8)

    visible_pixels_s = np.sum(visible_skin_mask_s & mask_theoretical)
    hair_s_in_mask = np.sum(hair_s & mask_theoretical)
    hat_s_in_mask = np.sum(hat_s & mask_theoretical)
    other_s_in_mask = max(0.0, total_pixels - (float(visible_pixels_s) + float(hair_s_in_mask) + float(hat_s_in_mask)))
    hair_ratio_s = hair_s_in_mask / total_pixels
    hat_ratio_s = hat_s_in_mask / total_pixels
    other_ratio_s = other_s_in_mask / total_pixels

    # === [Etape 3 du README] Detection plante v2 ===
    skin_only_b = np.isin(parsing_b, SKIN_ONLY_B).astype(np.uint8)
    bg_b        = np.isin(parsing_b, BG_B).astype(np.uint8)
    skin_only_s = np.isin(parsing_s, SKIN_ONLY_S).astype(np.uint8)
    skin_only_b_ratio = float(np.sum(skin_only_b & mask_theoretical)) / float(total_pixels)
    bg_b_ratio        = float(np.sum(bg_b        & mask_theoretical)) / float(total_pixels)
    skin_only_s_ratio = float(np.sum(skin_only_s & mask_theoretical)) / float(total_pixels)

    # === Calcul Julien ORIGINAL (pour comparaison) ===
    if pred_gender == 'M':
        W = M_WEIGHTS
    else:
        W = F_WEIGHTS
    score_julien_original = float(np.clip(
        W["hair_bi"] * hair_ratio_b + W["hat_bi"] * hat_ratio_b + W["other_bi"] * other_ratio_b +
        W["hair_sf"] * hair_ratio_s + W["hat_sf"] * hat_ratio_s + W["other_bg_sf"] * other_ratio_s,
        0, 1
    ))

    # === [Etape 4 du README] Avec v2 fallback ===
    score_v2 = apply_v2_fallback(
        pred_gender=pred_gender,
        hair_ratio_b=hair_ratio_b, hat_ratio_b=hat_ratio_b, other_ratio_b=other_ratio_b,
        hair_ratio_s=hair_ratio_s, hat_ratio_s=hat_ratio_s, other_ratio_s=other_ratio_s,
        skin_only_b_ratio=skin_only_b_ratio,
        bg_b_ratio=bg_b_ratio,
        skin_only_s_ratio=skin_only_s_ratio,
        M_WEIGHTS=M_WEIGHTS, F_WEIGHTS=F_WEIGHTS,
    )

    regime = detect_plante_regime(skin_only_b_ratio, bg_b_ratio, skin_only_s_ratio)

    return {
        "pred_gender": pred_gender,
        "score_julien_original": score_julien_original,
        "score_v2_fallback": score_v2,
        "regime": regime,
        "skin_only_b_ratio": skin_only_b_ratio,
        "bg_b_ratio": bg_b_ratio,
        "skin_only_s_ratio": skin_only_s_ratio,
        "hair_ratio_b": hair_ratio_b, "hat_ratio_b": hat_ratio_b, "other_ratio_b": other_ratio_b,
        "hair_ratio_s": hair_ratio_s, "hat_ratio_s": hat_ratio_s, "other_ratio_s": other_ratio_s,
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    # 1. Selectionne 10 images : 5 normales + 5 plante (a partir de val_features.csv)
    print("\n=== Selection des 10 images ===", flush=True)
    df = pd.read_csv(REPO / "eval/cache/val_features.csv")
    df = df[df.face_detected == 1].copy()

    # 5 images normales (bg_bi_in_mask bas, skin_sf_in_mask haut)
    normal = df[(df.bg_bi_in_mask < 0.10) & (df.skin_sf_in_mask > 0.60)].sample(5, random_state=42)
    # 5 images plante (extreme)
    plante = df[(df.bg_bi_in_mask > 0.80) | (df.skin_sf_in_mask < 0.20)].sample(5, random_state=42)
    sample = pd.concat([normal, plante], ignore_index=True)
    print(f"  {len(normal)} normales + {len(plante)} plante = {len(sample)} images", flush=True)

    # 2. Charge les modeles
    app, tddfa, net, seg_model, seg_processor = setup_models()

    # 3. Run sur chaque image
    print("\n=== Resultats par image ===", flush=True)
    print(f"{'#':>2} {'filename':<55} {'target':>7} {'gender':>6} {'pred_g':>6} {'Julien':>8} {'v2':>8} {'delta':>8} {'regime':>10}")
    print("-" * 130, flush=True)

    results = []
    for i, row in enumerate(sample.itertuples()):
        img = cv2.imread(str(REPO / "crops" / row.filename))
        if img is None:
            print(f"{i:>2} FAILED to load {row.filename}", flush=True)
            continue
        out = occlusion_computation(app, tddfa, net, seg_model, seg_processor, img)
        if out is None:
            print(f"{i:>2} {row.filename[-55:]:<55} (no face detected)", flush=True)
            continue
        delta = out["score_v2_fallback"] - out["score_julien_original"]
        gender_str = "F" if row.gender == 0 else "M"
        print(f"{i:>2} {row.filename[-55:]:<55} {row.target:>7.3f} {gender_str:>6} {out['pred_gender']:>6} "
              f"{out['score_julien_original']:>8.4f} {out['score_v2_fallback']:>8.4f} {delta:>+8.4f} {out['regime']:>10}", flush=True)
        results.append({**out, "filename": row.filename, "target": row.target, "gender": row.gender})

    # 4. Resume
    print("\n=== Resume ===", flush=True)
    rdf = pd.DataFrame(results)
    n_normal_regime = (rdf.regime == "normal").sum()
    n_plante_regime = (rdf.regime != "normal").sum()
    print(f"  Regime 'normal' : {n_normal_regime}/{len(rdf)} -> score_v2 == score_julien_original")
    print(f"  Regime plante   : {n_plante_regime}/{len(rdf)} -> score_v2 different (fallback active)")
    print()
    print("Verifications :")
    for _, r in rdf.iterrows():
        if r.regime == "normal":
            ok = abs(r.score_v2_fallback - r.score_julien_original) < 1e-6
            sym = "OK " if ok else "FAIL"
            print(f"  [{sym}] regime=normal : v2 doit == julien_original   delta={r.score_v2_fallback-r.score_julien_original:+.6f}")
        else:
            print(f"  [INFO] regime={r.regime:<8s} : v2 active fallback   target={r.target:.3f}  julien={r.score_julien_original:.3f}  v2={r.score_v2_fallback:.3f}")


if __name__ == "__main__":
    main()
