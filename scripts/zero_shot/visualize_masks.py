"""Visualize the 3DDFA mask, BiSeNet skin, SegFormer skin on a few sample images.

Saves side-by-side images to figures/mask_alignment/ so we can verify mask
positioning matches the expectation (and detect any "off by N pixels" issue
that could match Julien's "correction de position du masque" remark).
"""
from __future__ import annotations

import sys
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

BISENET_VISIBLE = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]
SEGFORMER_VISIBLE = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def setup():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                       allowed_modules=["detection"])
    app.prepare(ctx_id=0, det_size=(224, 224))

    from TDDFA import TDDFA
    cfg = yaml.load(open(REPO_ROOT / "3DDFA_V2/configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
    cfg["checkpoint_fp"] = str(REPO_ROOT / "3DDFA_V2/weights/mb1_120x120.pth")
    cfg["bfm_fp"] = str(REPO_ROOT / "3DDFA_V2/configs/bfm_noneck_v3.pkl")
    cfg["param_mean_std_fp"] = str(REPO_ROOT / "3DDFA_V2/configs/param_mean_std_62d_120x120.pkl")
    tddfa = TDDFA(**cfg)

    from model import BiSeNet
    net_bi = BiSeNet(n_classes=19)
    net_bi.load_state_dict(
        torch.load(REPO_ROOT / "weights/79999_iter.pth", map_location="cpu", weights_only=True)
    )
    net_bi.eval()

    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    sf = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").eval()
    sf_proc = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return app, tddfa, net_bi, sf, sf_proc, to_tensor


def visualize(img_path, target, app, tddfa, net_bi, sf, sf_proc, to_tensor):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    faces = app.get(img)
    if not faces:
        return None
    bbox = faces[0].bbox

    param_lst, roi_box_lst = tddfa(img, [bbox])
    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
    pts = ver[:2, :].T.astype(np.int32)
    hull = cv2.convexHull(pts)
    mask_3d = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask_3d, hull, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = to_tensor(Image.fromarray(img_rgb)).unsqueeze(0)
    with torch.inference_mode():
        out_bi = net_bi(inp)[0]
    parsing_bi = out_bi.squeeze(0).cpu().numpy().argmax(0)
    if parsing_bi.shape != (h, w):
        parsing_bi = cv2.resize(parsing_bi.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    skin_bi = np.isin(parsing_bi, BISENET_VISIBLE).astype(np.uint8)

    pil_img = Image.fromarray(img_rgb)
    inputs = sf_proc(images=pil_img, return_tensors="pt")
    with torch.inference_mode():
        sf_out = sf(**inputs)
    seg_sf = torch.nn.functional.interpolate(
        sf_out.logits, size=(h, w), mode="bilinear", align_corners=False
    ).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    skin_sf = np.isin(seg_sf, SEGFORMER_VISIBLE).astype(np.uint8)

    # Compute ratios
    a3d = max(int(mask_3d.sum()), 1)
    r_3D_Bi = 1.0 - float((skin_bi & mask_3d).sum()) / a3d
    r_3D_Sf = 1.0 - float((skin_sf & mask_3d).sum()) / a3d

    # Build a visualization image: 4 panels side-by-side
    # 1. Original image
    # 2. 3DDFA mask overlay (red)
    # 3. BiSeNet skin overlay (green) + 3DDFA mask outline
    # 4. SegFormer skin overlay (blue) + 3DDFA mask outline
    def overlay(img, mask, color, alpha=0.4):
        out = img.copy()
        out[mask.astype(bool)] = (1 - alpha) * out[mask.astype(bool)] + alpha * np.array(color)
        return out.astype(np.uint8)

    img_rgb_arr = img_rgb.copy()
    panel1 = img_rgb_arr.copy()
    panel2 = overlay(img_rgb_arr, mask_3d, [255, 0, 0])
    panel3 = overlay(img_rgb_arr, skin_bi & mask_3d, [0, 255, 0])
    panel3 = overlay(panel3, mask_3d & (~skin_bi.astype(bool)), [255, 0, 0])
    panel4 = overlay(img_rgb_arr, skin_sf & mask_3d, [0, 200, 255])
    panel4 = overlay(panel4, mask_3d & (~skin_sf.astype(bool)), [255, 0, 0])

    # Add text labels
    def put_text(img, text, y=20):
        return cv2.putText(img.copy(), text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 2, cv2.LINE_AA)

    panel1 = put_text(panel1, f"target={target:.3f}", 20)
    panel2 = put_text(panel2, "3DDFA mask (red)", 20)
    panel3 = put_text(panel3, f"BiSe skin+mask, r={r_3D_Bi:.3f}", 20)
    panel4 = put_text(panel4, f"SegF skin+mask, r={r_3D_Sf:.3f}", 20)

    # Resize to consistent size and concat horizontally
    target_h = 224
    def resize_to_h(img, h_target):
        ratio = h_target / img.shape[0]
        new_w = int(img.shape[1] * ratio)
        return cv2.resize(img, (new_w, h_target))

    panels = [resize_to_h(p, target_h) for p in [panel1, panel2, panel3, panel4]]
    grid = np.concatenate(panels, axis=1)
    return cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)


def main():
    out_dir = REPO_ROOT / "figures" / "mask_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    j = pd.read_csv(REPO_ROOT / "eval" / "val_julien_baseline.csv").rename(columns={"pred": "pj"})

    # Sample: 2 low-target, 2 mid, 2 high
    j_sorted = j.sort_values("target")
    sample = pd.concat([
        j_sorted.head(2),
        j_sorted[(j_sorted.target > 0.18) & (j_sorted.target < 0.25)].head(2),
        j_sorted.tail(2),
    ])
    print(f"will visualize {len(sample)} samples")

    app, tddfa, net_bi, sf, sf_proc, to_tensor = setup()

    image_dir = REPO_ROOT / "crops"
    for fn, target in zip(sample.filename, sample.target):
        print(f"  {fn[-30:]} target={target:.3f}")
        grid = visualize(image_dir / fn, target, app, tddfa, net_bi, sf, sf_proc, to_tensor)
        if grid is not None:
            stem = fn.replace("/", "_").replace(".webp", "")
            out_path = out_dir / f"{stem[-50:]}.png"
            cv2.imwrite(str(out_path), grid)

    print(f"\nwrote visualizations to {out_dir}")


if __name__ == "__main__":
    main()
