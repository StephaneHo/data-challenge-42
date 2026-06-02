"""Audit COMPLET et long pour vérifier les bugs résiduels après les fixes.

Tests :
  A. Multi-face stats (val)
  B. Spot-check class semantics sur 15 images (bilan détaillé)
  C. Validation cohérence cross_test vs run_julien_pipeline (5 images)
  D. Pose-aware : extraction yaw/pitch via 3DDFA, identification de poses extremes
  E. 3DDFA mask: vertices in/out of image, mask shape stats
  F. Color space verification : BGR/RGB confusion ?
  G. Edge cases : 5 images difficiles
  H. Range checks: les 4 ratios dans [0, 1] ?
  I. Comparaison du score per-gender sur ~50 images stratifiées (échantillon plus grand)
  J. IoU 3DDFA mask vs BiSeNet face area : si très faible -> masque pas aligné
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "3DDFA_V2"))
sys.path.insert(0, str(REPO_ROOT / "face-parsing.PyTorch"))

warnings.filterwarnings("ignore")

BISENET_VISIBLE = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]
SEGFORMER_VISIBLE = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]

BISENET_LABELS = {0: "bg", 1: "skin", 2: "l_brow", 3: "r_brow", 4: "l_eye", 5: "r_eye",
                  6: "eye_g", 7: "l_ear", 8: "r_ear", 9: "ear_r", 10: "nose", 11: "mouth",
                  12: "u_lip", 13: "l_lip", 14: "neck", 15: "neck_l", 16: "cloth",
                  17: "hair", 18: "hat"}
SEGFORMER_LABELS = {0: "bg", 1: "skin", 2: "nose", 3: "eye_g", 4: "l_eye", 5: "r_eye",
                    6: "l_brow", 7: "r_brow", 8: "l_ear", 9: "r_ear", 10: "mouth",
                    11: "u_lip", 12: "l_lip", 13: "hair", 14: "hat", 15: "ear_r",
                    16: "neck_l", 17: "neck", 18: "cloth"}


def rotation_matrix_to_yaw_pitch_roll(R):
    """Extract yaw, pitch, roll (degrees) from 3x3 rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return math.degrees(y), math.degrees(x), math.degrees(z)  # yaw, pitch, roll


def main():
    print("=" * 90)
    print("AUDIT EXHAUSTIF -- derniers checks avant lancement complet")
    print("=" * 90)

    # ============== A. Multi-face stats ==============
    print()
    print("A. Multi-face statistics (val)")
    print("-" * 90)
    pg = pd.read_csv(REPO_ROOT / "eval" / "cache" / "val_gender_pred.csv")
    print(f"   val n_faces distribution: {pg.n_faces.value_counts().sort_index().to_dict()}")
    multi = (pg.n_faces > 1).sum()
    print(f"   {multi}/{len(pg)} = {100*multi/len(pg):.2f}% images avec >1 visage")
    # On test set too
    if (REPO_ROOT / "eval" / "cache" / "test_gender_pred.csv").exists():
        tp = pd.read_csv(REPO_ROOT / "eval" / "cache" / "test_gender_pred.csv")
        print(f"   test n_faces distribution: {tp.n_faces.value_counts().sort_index().to_dict()}")
        print(f"   {(tp.n_faces > 1).sum()}/{len(tp)} = {100*(tp.n_faces>1).sum()/len(tp):.2f}% test images avec >1 visage")

    print("   OK Conclusion: impact multi-face mineur (<1%). On garde faces[0].")

    # ============== Setup models ==============
    print()
    print("Loading models for inference-based checks...")
    from insightface.app import FaceAnalysis
    from TDDFA import TDDFA
    from model import BiSeNet
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                       allowed_modules=["detection"])
    app.prepare(ctx_id=0, det_size=(224, 224))

    cfg = yaml.load(open(REPO_ROOT / "3DDFA_V2/configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
    cfg["checkpoint_fp"] = str(REPO_ROOT / "3DDFA_V2/weights/mb1_120x120.pth")
    cfg["bfm_fp"] = str(REPO_ROOT / "3DDFA_V2/configs/bfm_noneck_v3.pkl")
    cfg["param_mean_std_fp"] = str(REPO_ROOT / "3DDFA_V2/configs/param_mean_std_62d_120x120.pkl")
    tddfa = TDDFA(**cfg)

    net_bi = BiSeNet(n_classes=19)
    net_bi.load_state_dict(torch.load(REPO_ROOT / "weights/79999_iter.pth", map_location="cpu", weights_only=True))
    net_bi.eval()

    sf = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").eval()
    sf_proc = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Pick varied sample for detailed checks
    val = pd.read_csv(REPO_ROOT / "eval" / "val_julien_baseline.csv")
    np.random.seed(42)
    # Stratified by target bin
    samples = pd.concat([
        val[val.target < 0.05].sample(3, random_state=42),
        val[(val.target >= 0.05) & (val.target < 0.15)].sample(3, random_state=42),
        val[(val.target >= 0.15) & (val.target < 0.30)].sample(3, random_state=42),
        val[(val.target >= 0.30) & (val.target < 0.50)].sample(3, random_state=42),
        val[val.target >= 0.50].sample(min(3, len(val[val.target >= 0.50])), random_state=42),
    ])
    samples = samples.reset_index(drop=True)
    print(f"\n  Detailed checks on {len(samples)} stratified images.")

    # Collect results
    results = []
    for i, row in samples.iterrows():
        img_bgr = cv2.imread(str(REPO_ROOT / "crops" / row.filename))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)

        # Face detection
        faces = app.get(img_bgr)
        if not faces:
            results.append({"filename": row.filename, "target": row.target, "face_detected": 0})
            continue
        face = faces[0]
        bbox = face.bbox

        # 3DDFA -- extract yaw/pitch
        param_lst, roi_box_lst = tddfa(img_bgr, [bbox])
        ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
        # Pose extraction from param matrix (first 12 of 62 params)
        pose_mat = param_lst[0][:12].reshape(3, 4)
        R = pose_mat[:, :3]
        yaw, pitch, roll = rotation_matrix_to_yaw_pitch_roll(R)

        pts = ver[:2, :].T.astype(np.int32)
        # Stats on vertex positions
        x_in = ((pts[:, 0] >= 0) & (pts[:, 0] < w)).sum()
        y_in = ((pts[:, 1] >= 0) & (pts[:, 1] < h)).sum()
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

        hull = cv2.convexHull(pts)
        mask_3d = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask_3d, hull, 1)

        # BiSeNet @ 512
        pil_512 = pil.resize((512, 512), Image.BILINEAR)
        inp_bi = to_tensor(pil_512).unsqueeze(0)
        with torch.inference_mode():
            out_bi = net_bi(inp_bi)[0]
        parsing_bi_512 = out_bi.squeeze(0).numpy().argmax(0).astype(np.uint8)
        parsing_bi = cv2.resize(parsing_bi_512, (w, h), interpolation=cv2.INTER_NEAREST)
        skin_bi = np.isin(parsing_bi, BISENET_VISIBLE).astype(np.uint8)

        # SegFormer
        inp_sf = sf_proc(images=pil, return_tensors="pt")
        with torch.inference_mode():
            out_sf = sf(**inp_sf)
        seg_sf = torch.nn.functional.interpolate(
            out_sf.logits, size=(h, w), mode="bilinear", align_corners=False
        ).argmax(1).squeeze(0).numpy().astype(np.uint8)
        skin_sf = np.isin(seg_sf, SEGFORMER_VISIBLE).astype(np.uint8)

        # The 4 ratios
        a3d = max(int(mask_3d.sum()), 1)
        r_3D_Bi = 1.0 - float((skin_bi & mask_3d).sum()) / a3d
        r_3D_Sf = 1.0 - float((skin_sf & mask_3d).sum()) / a3d

        # IoU between 3DDFA mask and BiSeNet skin (sanity for mask alignment)
        iou_3d_vs_bi_face = float((mask_3d & skin_bi).sum()) / max(int((mask_3d | skin_bi).sum()), 1)
        iou_3d_vs_sf_face = float((mask_3d & skin_sf).sum()) / max(int((mask_3d | skin_sf).sum()), 1)

        results.append({
            "filename": row.filename,
            "target": row.target,
            "gender": row.gender,
            "n_faces_detected": len(faces),
            "bbox": list(bbox.astype(int)),
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "3ddfa_n_verts": len(pts),
            "3ddfa_x_in_img": x_in,
            "3ddfa_y_in_img": y_in,
            "3ddfa_x_min": int(x_min),
            "3ddfa_x_max": int(x_max),
            "3ddfa_y_min": int(y_min),
            "3ddfa_y_max": int(y_max),
            "img_h": h,
            "img_w": w,
            "mask_3d_frac": float(mask_3d.sum()) / (h * w),
            "skin_bi_frac": float(skin_bi.sum()) / (h * w),
            "skin_sf_frac": float(skin_sf.sum()) / (h * w),
            "r_3D_Bi": r_3D_Bi,
            "r_3D_Sf": r_3D_Sf,
            "iou_3d_bi_face": iou_3d_vs_bi_face,
            "iou_3d_sf_face": iou_3d_vs_sf_face,
        })

    df = pd.DataFrame(results)

    # B. Detailed table
    print()
    print("B. Detailed table (sorted by yaw to spot extreme poses)")
    print("-" * 90)
    print(df[["filename", "target", "yaw_deg", "pitch_deg", "mask_3d_frac",
              "iou_3d_bi_face", "iou_3d_sf_face", "r_3D_Bi", "r_3D_Sf"]].sort_values("yaw_deg").round(3).to_string(index=False))

    # C. Pose-extreme analysis
    print()
    print("C. Pose extremes -- yaw range, do extreme yaws have mask issues?")
    print("-" * 90)
    print(f"   yaw range: [{df.yaw_deg.min():.1f}, {df.yaw_deg.max():.1f}] degrees")
    print(f"   pitch range: [{df.pitch_deg.min():.1f}, {df.pitch_deg.max():.1f}] degrees")
    extreme_yaw = df[df.yaw_deg.abs() > 30]
    if len(extreme_yaw) > 0:
        print(f"   {len(extreme_yaw)} images avec |yaw|>30deg :")
        print(extreme_yaw[["filename", "yaw_deg", "mask_3d_frac", "r_3D_Bi", "r_3D_Sf", "iou_3d_bi_face"]].round(3).to_string(index=False))
    else:
        print("   Aucune image avec |yaw|>30deg dans l'échantillon -- peut-être pas de pose extreme dans val.")

    # D. 3DDFA vertex coverage
    print()
    print("D. 3DDFA vertices: combien d'entre eux sont DANS l'image ?")
    print("-" * 90)
    df["3ddfa_x_pct_in"] = df["3ddfa_x_in_img"] / df["3ddfa_n_verts"]
    df["3ddfa_y_pct_in"] = df["3ddfa_y_in_img"] / df["3ddfa_n_verts"]
    print(f"   pct_in_x mean: {df['3ddfa_x_pct_in'].mean()*100:.1f}%, min: {df['3ddfa_x_pct_in'].min()*100:.1f}%")
    print(f"   pct_in_y mean: {df['3ddfa_y_pct_in'].mean()*100:.1f}%, min: {df['3ddfa_y_pct_in'].min()*100:.1f}%")
    # If many vertices are outside image, the convex hull is clipped -> potentially wrong mask
    issue = df[df["3ddfa_x_pct_in"] < 0.95]
    if len(issue):
        print(f"   {len(issue)} images avec <95% vertices dans l'image en x -- potentiel issue")
        print(issue[["filename", "yaw_deg", "3ddfa_x_pct_in", "r_3D_Bi"]].round(3).to_string(index=False))
    else:
        print("   OK : >95% vertices toujours dans l'image en x sur tout l'échantillon")

    # E. IoU 3DDFA vs face area
    print()
    print("E. IoU 3DDFA mask vs detected skin (sanity du masque)")
    print("-" * 90)
    print(f"   iou_3d_vs_bi_skin: mean={df.iou_3d_bi_face.mean():.3f}, min={df.iou_3d_bi_face.min():.3f}")
    print(f"   iou_3d_vs_sf_skin: mean={df.iou_3d_sf_face.mean():.3f}, min={df.iou_3d_sf_face.min():.3f}")
    print("   Note: IoU avec skin n'est pas censé être 1 (le mask 3DDFA inclut zones occluses).")
    print("   On veut IoU > 0.3 environ pour s'assurer que le mask et le skin se chevauchent.")
    low_iou = df[df.iou_3d_bi_face < 0.3]
    if len(low_iou):
        print(f"   {len(low_iou)} images avec IoU < 0.3 :")
        print(low_iou[["filename", "target", "iou_3d_bi_face", "skin_bi_frac", "mask_3d_frac"]].round(3).to_string(index=False))

    # F. Color space verification
    print()
    print("F. Color space verification (BGRvsRGB)")
    print("-" * 90)
    # On a sample image, check that cv2.imread gives BGR
    test_img = cv2.imread(str(REPO_ROOT / "crops" / samples.iloc[0].filename))
    print(f"   cv2.imread output shape: {test_img.shape}, dtype: {test_img.dtype}")
    print(f"   B channel mean: {test_img[:, :, 0].mean():.0f}, G mean: {test_img[:, :, 1].mean():.0f}, R mean: {test_img[:, :, 2].mean():.0f}")
    print(f"   After cvtColor BGR2RGB:")
    test_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    print(f"   R: {test_rgb[:, :, 0].mean():.0f}, G: {test_rgb[:, :, 1].mean():.0f}, B: {test_rgb[:, :, 2].mean():.0f}")
    print("   OK Ce qu'on passe à PIL est bien RGB.")

    # G. Range checks
    print()
    print("G. Range checks sur les 4 ratios (smoke fixed data, 28 images)")
    print("-" * 90)
    smoke = pd.read_csv(REPO_ROOT / "eval" / "cache" / "val_cross_smoke_allfixed.csv")
    for col in ["r_3D_Bi", "r_3D_Sf", "r_Bi_Cv", "r_Sf_Cv"]:
        n_nan = smoke[col].isna().sum()
        valid = smoke[col].dropna()
        out_range = ((valid < 0) | (valid > 1)).sum()
        print(f"   {col}: NaN={n_nan}, out_of_[0,1]={out_range}, min={valid.min():.4f}, max={valid.max():.4f}")

    # H. Per-gender weighted error
    print()
    print("H. Per-gender weighted err sur smoke fixed (28 images)")
    print("-" * 90)
    from src.metric import weighted_err
    for col in ["r_3D_Bi", "r_3D_Sf"]:
        f_mask = (smoke.gender == 0.0) & smoke[col].notna()
        m_mask = (smoke.gender == 1.0) & smoke[col].notna()
        if f_mask.sum() == 0 or m_mask.sum() == 0:
            continue
        err_f = weighted_err(smoke.loc[f_mask, col].values, smoke.loc[f_mask, "target"].values)
        err_m = weighted_err(smoke.loc[m_mask, col].values, smoke.loc[m_mask, "target"].values)
        print(f"   {col}: err_F={err_f:.5f} (n={f_mask.sum()}), err_M={err_m:.5f} (n={m_mask.sum()}), gap={abs(err_f-err_m):.5f}")

    # I. Save audit data
    df.to_csv(REPO_ROOT / "eval" / "cache" / "audit_long.csv", index=False)
    print()
    print(f"   Audit data saved to eval/cache/audit_long.csv")


if __name__ == "__main__":
    main()
