"""Run Julien's TF pipeline (InsightFace + 3DDFA-V2 + BiSeNet) on val or test.

Replicates the `occlusion_computation` function from notebooks/DataChallengeJulien_TF.ipynb
without requiring the Sim3DR/FaceBoxes C++ builds (uses pure TDDFA + InsightFace).

Output: filename, pred, target, gender CSV (val) or filename, FaceOcclusion, gender (test)

Usage:
    python scripts/zero_shot/run_julien_pipeline.py --source val --limit 50 --out eval/val_julien_sample.csv
    python scripts/zero_shot/run_julien_pipeline.py --source val --out eval/val_julien_baseline.csv
"""
from __future__ import annotations

import argparse
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
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "3DDFA_V2"))
sys.path.insert(0, str(REPO_ROOT / "face-parsing.PyTorch"))

from src.data import stratified_split  # noqa: E402


# BiSeNet CelebAMask-HQ classes treated as "visible face skin"
VISIBLE_FACE_CLASSES = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]
# 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye, 7=l_ear, 8=r_ear,
# 10=nose, 11=mouth, 12=u_lip, 13=l_lip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(REPO_ROOT / "occlusion_datasets"))
    p.add_argument("--image-dir", default=str(REPO_ROOT / "crops"))
    p.add_argument("--source", choices=["val", "test"], default="val")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_models():
    """Load InsightFace, TDDFA, BiSeNet. Return (face_app, tddfa, bisenet, device, transform)."""
    print("loading InsightFace (RetinaFace)...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BiSeNet(n_classes=19)
    net.load_state_dict(torch.load(REPO_ROOT / "weights/79999_iter.pth", map_location=device, weights_only=True))
    net.to(device).eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return app, tddfa, net, device, to_tensor


def compute_occlusion(img_bgr, app, tddfa, net, device, to_tensor):
    """Replicates Julien's occlusion_computation cell 39.

    Returns the raw occlusion ratio in [0, 1], or NaN if face detection failed.
    """
    # Face detection
    faces = app.get(img_bgr)
    if not faces:
        return float("nan")
    face = faces[0]
    bbox = face.bbox

    # 3D face estimation (3DDFA-V2)
    boxes = [bbox]
    param_lst, roi_box_lst = tddfa(img_bgr, boxes)
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

    # Theoretical face mask (convex hull of projected vertices)
    pts = ver_lst[0][:2, :].T.astype(np.int32)
    hull = cv2.convexHull(pts)
    h, w = img_bgr.shape[:2]
    mask_theoretical = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask_theoretical, hull, 1)

    # BiSeNet face parsing
    # IMPORTANT (fix 2026-06-02): BiSeNet (face-parsing.PyTorch) requires 512x512
    # input per its reference test.py. Without resize, the model misses many face
    # parts on 224x224 crops. Resize first, infer, then resize output back.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_512 = Image.fromarray(img_rgb).resize((512, 512), Image.BILINEAR)
    input_tensor = to_tensor(pil_512).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = net(input_tensor)[0]
    parsing_512 = out.squeeze(0).cpu().numpy().argmax(0).astype(np.uint8)
    parsing = cv2.resize(parsing_512, (w, h), interpolation=cv2.INTER_NEAREST)

    # Visible skin mask
    visible_skin_mask = np.isin(parsing, VISIBLE_FACE_CLASSES).astype(np.uint8)

    # Compute occlusion ratio (= 1 - visible_ratio)
    total_pixels = int(mask_theoretical.sum())
    if total_pixels == 0:
        return float("nan")
    visible_pixels = int((visible_skin_mask & mask_theoretical).sum())
    visible_ratio = visible_pixels / total_pixels
    return float(1.0 - visible_ratio)


def main() -> None:
    args = parse_args()

    if args.source == "val":
        train_csv = pd.read_csv(Path(args.data_dir) / "train.csv")
        _, df = stratified_split(train_csv, val_frac=args.val_frac, seed=args.seed)
        has_labels = True
    else:
        df = pd.read_csv(Path(args.data_dir) / "test_students.csv")
        has_labels = False

    if args.limit > 0:
        df = df.head(args.limit)
    print(f"running Julien's pipeline on {len(df)} {args.source} rows")

    app, tddfa, net, device, to_tensor = setup_models()
    print(f"all models loaded, device={device}")

    rows = []
    image_dir = Path(args.image_dir)
    for fn in tqdm(df["filename"].tolist(), desc="julien"):
        img = cv2.imread(str(image_dir / fn))
        if img is None:
            rows.append((fn, float("nan")))
            continue
        try:
            pred = compute_occlusion(img, app, tddfa, net, device, to_tensor)
        except Exception as e:
            print(f"\nfailed on {fn}: {e}")
            pred = float("nan")
        rows.append((fn, pred))

    pred_df = pd.DataFrame(rows, columns=["filename", "pred"])
    if has_labels:
        merged = df[["filename", "FaceOcclusion", "gender"]].merge(pred_df, on="filename")
        merged = merged.rename(columns={"FaceOcclusion": "target"})
        out_df = merged[["filename", "pred", "target", "gender"]]
    else:
        out_df = pred_df.rename(columns={"pred": "FaceOcclusion"})
        out_df["gender"] = "x"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(out_df)} rows)")
    n_nan = out_df.iloc[:, 1].isna().sum()
    if n_nan > 0:
        print(f"  {n_nan} NaN predictions (face detection failed)")
    print(f"  pred mean={out_df.iloc[:, 1].mean():.4f}  std={out_df.iloc[:, 1].std():.4f}")


if __name__ == "__main__":
    main()
