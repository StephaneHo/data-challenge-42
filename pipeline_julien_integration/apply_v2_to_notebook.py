"""Refactorise le notebook DataChallengeJulien_TF.ipynb pour integrer le v2 fallback
proprement, avec une fonction `occlusion_computation` splittee en sous-fonctions.

Etapes appliquees :
  Etape 1 : ajoute SKIN_ONLY_B, SKIN_ONLY_S, BG_B dans la cellule des constantes (cell 35)
  Etape 2 : insere une nouvelle cellule d'import du module pipeline_julien_integration
  Etape 3 : SUPPRIME la fonction monolithique occlusion_computation
  Etape 4 : INSERE plusieurs cellules avec des sous-fonctions reutilisables :
              - detect_face_and_gender
              - compute_theoretical_face_mask
              - segment_with_bisenet
              - segment_with_segformer
              - compute_ratios_from_parsing
              - compute_plante_features
              - occlusion_computation (orchestrateur)

L'orchestrateur garde la meme signature occlusion_computation(app, img, display_results=False)
pour compat avec les cellules suivantes (39, 42, 57) qui l'appellent.

Usage:
    python apply_v2_to_notebook.py
"""
import json
import shutil
from pathlib import Path

NB_PATH = Path(__file__).parent / "DataChallengeJulien_TF_with_v2_fallback.ipynb"
NB_SOURCE = Path(__file__).parent.parent / "notebooks" / "DataChallengeJulien_TF.ipynb"


# ============================================================================
# CELLULE ETAPE 1 : ajout des constantes pour le fallback v2
# (inseree dans la cell 35 existante, apres HAT_S = [14])
# ============================================================================
ETAPE_1_INSERT = """
# === [Etape 1] Constantes pour la detection plante du fallback v2 ===
# NB : SKIN_ONLY_B/S incluent les ears (7,8 BiSeNet ; 8,9 SegFormer) alors que
# VISIBLE_FACE_CLASSES_B/S de Julien ne les incluent pas. C'est volontaire :
# les seuils 0.70 et 0.30 ont ete calibres en CV sur cette definition AVEC ears.
SKIN_ONLY_B = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]
SKIN_ONLY_S = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
BG_B        = [0]
"""


# ============================================================================
# CELLULE ETAPE 2 : import du module pipeline_julien_integration
# (nouvelle cellule inseree avant les sous-fonctions)
# ============================================================================
ETAPE_2_IMPORT = """# === [Etape 2] Import du module v2 fallback ===
# Setup path pour acceder au module pipeline_julien_integration
import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))  # 2 niveaux car notebook dans pipeline_julien_integration/
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import des 2 versions (avec et sans dampening) pour pouvoir comparer
from pipeline_julien_integration import (
    apply_v2_fallback,                   # version originale (bit-exact sur cas normaux)
    apply_v2_fallback_with_dampening,    # version avec dampening hair/hat (+2% CV)
    detect_plante_regime,
)
print("v2 fallback charge OK            :", apply_v2_fallback)
print("v2 fallback + dampening charge OK :", apply_v2_fallback_with_dampening)
"""


# ============================================================================
# CELLULES DES SOUS-FONCTIONS (etape 4)
# ============================================================================
CELL_DETECT_FACE = """# === Sous-fonction 1 : detection visage + genre (InsightFace) ===
def detect_face_and_gender(app, img):
    \"\"\"Detecte le visage principal dans l'image avec InsightFace.

    Args:
        app: instance FaceAnalysis InsightFace
        img: numpy array BGR

    Returns:
        (face_bbox, pred_gender) ou (None, None) si aucun visage detecte.
        pred_gender est 'F' ou 'M' (sortie face.sex).
    \"\"\"
    faces = app.get(img)
    if not faces:
        return None, None
    face = faces[0]
    return face.bbox, face.sex
"""

CELL_COMPUTE_MASK = """# === Sous-fonction 2 : calcul du mask theorique (3DDFA-V2 + correction Julien) ===
def compute_theoretical_face_mask(tddfa, img, bbox):
    \"\"\"Calcule le mask theorique du visage par reconstruction 3D et applique la
    correction empirique de Julien (scale_x=0.9, scale_y=1.05, tx=15, ty=-10).

    Args:
        tddfa: instance TDDFA
        img: numpy array BGR
        bbox: face bbox (sortie InsightFace)

    Returns:
        (mask_theoretical, total_pixels) : matrice HxW uint8 et nombre total de pixels du mask.
    \"\"\"
    # Reconstruction 3D du visage
    param_lst, roi_box_lst = tddfa(img, [bbox])
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

    # Convex hull des vertices projetes -> mask 2D
    mask_theoretical = np.zeros(img.shape[:2], dtype=np.uint8)
    pts = ver_lst[0][:2, :].T.astype(np.int32)
    cv2.fillConvexPoly(mask_theoretical, cv2.convexHull(pts), 1)

    # Correction empirique de Julien
    scale_x = 0.9
    scale_y = 1.05
    tx = 15
    ty = -10
    M = np.array([[scale_x, 0, tx], [0, scale_y, ty]], dtype=np.float32)
    mask_theoretical = cv2.warpAffine(mask_theoretical, M, dsize=img.shape[:2])

    return mask_theoretical, float(np.sum(mask_theoretical))
"""

CELL_SEGMENT_BISENET = """# === Sous-fonction 3 : segmentation BiSeNet ===
def segment_with_bisenet(net, to_tensor, img, device='cpu'):
    \"\"\"Run BiSeNet face parsing sur l'image et retourne le parsing argmax.

    Args:
        net: modele BiSeNet
        to_tensor: torchvision transforms compose (ToTensor + Normalize)
        img: numpy array BGR (sera converti automatiquement par to_tensor)
        device: 'cpu' ou 'cuda'

    Returns:
        parsing: numpy array HxW d'entiers (indices de classe 0-18 pour BiSeNet)
    \"\"\"
    input_tensor = to_tensor(img).unsqueeze(0).to(device)
    out = net(input_tensor)[0]
    return out.squeeze(0).cpu().numpy().argmax(0)
"""

CELL_SEGMENT_SEGFORMER = """# === Sous-fonction 4 : segmentation SegFormer ===
def segment_with_segformer(seg_model, seg_processor, img, device='cpu'):
    \"\"\"Run SegFormer face parsing sur l'image et retourne le parsing argmax.

    Args:
        seg_model: modele SegFormer (transformers SegformerForSemanticSegmentation)
        seg_processor: SegformerImageProcessor associe
        img: numpy array BGR
        device: 'cpu' ou 'cuda'

    Returns:
        parsing: numpy array HxW d'entiers (indices de classe 0-18 pour SegFormer)
    \"\"\"
    inputs = seg_processor(images=img, return_tensors="pt").to(device)
    outputs = seg_model(**inputs)
    # Upsample les logits a la taille de l'image originale
    h, w = img.shape[:2]
    upsampled = nn.functional.interpolate(
        outputs.logits, size=(h, w), mode='bilinear', align_corners=False
    )
    return upsampled.argmax(dim=1)[0].cpu().numpy()
"""

CELL_COMPUTE_RATIOS = """# === Sous-fonction 5 : calcul generique des ratios hair/hat/other ===
def compute_ratios_from_parsing(parsing, mask_theoretical, total_pixels,
                                 VISIBLE_CLASSES, HAIR_CLASSES, HAT_CLASSES):
    \"\"\"Calcule (hair_ratio, hat_ratio, other_ratio) a partir d'un parsing.

    Generique : fonctionne pour BiSeNet ou SegFormer en passant les bonnes listes
    de classes.

    Args:
        parsing: numpy array HxW d'indices de classe
        mask_theoretical: numpy array HxW (uint8 0/1) du mask
        total_pixels: nombre total de pixels du mask (float)
        VISIBLE_CLASSES: liste des classes 'visibles' (sera filtree)
        HAIR_CLASSES: liste des classes 'cheveux' (typiquement [17] BiSeNet, [13] SegFormer)
        HAT_CLASSES: liste des classes 'chapeau' (typiquement [18] BiSeNet, [14] SegFormer)

    Returns:
        (hair_ratio, hat_ratio, other_ratio) : 3 floats dont la somme + visible_ratio = 1.0
    \"\"\"
    # Masks binaires par categorie
    visible_mask = np.isin(parsing, VISIBLE_CLASSES).astype(np.uint8)
    hair_mask = np.isin(parsing, HAIR_CLASSES).astype(np.uint8)
    hat_mask = np.isin(parsing, HAT_CLASSES).astype(np.uint8)

    # Pixels par categorie dans le mask theorique
    visible_pixels = float(np.sum(visible_mask & mask_theoretical))
    hair_pixels    = float(np.sum(hair_mask    & mask_theoretical))
    hat_pixels     = float(np.sum(hat_mask     & mask_theoretical))
    # Other = total - (visible + hair + hat). Conventionnellement clip a 0.
    other_pixels = max(0.0, total_pixels - visible_pixels - hair_pixels - hat_pixels)

    return (
        hair_pixels / total_pixels,
        hat_pixels / total_pixels,
        other_pixels / total_pixels,
    )
"""

CELL_COMPUTE_PLANTE = """# === Sous-fonction 6 : calcul des 3 features pour detection plante v2 ===
def compute_plante_features(parsing_b, parsing_s, mask_theoretical, total_pixels):
    \"\"\"Calcule les 3 features supplementaires necessaires au fallback v2.

    Ces features sont independantes de la formule principale et servent uniquement
    a detecter quand BiSeNet ou SegFormer plante.

    Args:
        parsing_b: parsing BiSeNet (sortie de segment_with_bisenet)
        parsing_s: parsing SegFormer (sortie de segment_with_segformer)
        mask_theoretical: mask theorique (sortie de compute_theoretical_face_mask)
        total_pixels: nombre total de pixels du mask

    Returns:
        (skin_only_b_ratio, bg_b_ratio, skin_only_s_ratio) : 3 floats dans [0, 1]
    \"\"\"
    skin_only_b = np.isin(parsing_b, SKIN_ONLY_B).astype(np.uint8)
    bg_b        = np.isin(parsing_b, BG_B).astype(np.uint8)
    skin_only_s = np.isin(parsing_s, SKIN_ONLY_S).astype(np.uint8)
    return (
        float(np.sum(skin_only_b & mask_theoretical)) / total_pixels,
        float(np.sum(bg_b        & mask_theoretical)) / total_pixels,
        float(np.sum(skin_only_s & mask_theoretical)) / total_pixels,
    )
"""

CELL_ORCHESTRATOR = """# === Orchestrateur : occlusion_computation (appelle toutes les sous-fonctions) ===
def occlusion_computation(app, img, display_results=False):
    \"\"\"Pipeline complete : detecte le visage, segmente, calcule les ratios,
    applique le v2 fallback + dampening, retourne le score d'occlusion.

    Signature compatible avec les cellules suivantes du notebook qui appellent
    occlusion_computation(app, img) pour la generation des resultats val / test.

    Args:
        app: instance FaceAnalysis InsightFace
        img: numpy array BGR
        display_results: si True, affiche les overlays (debug visuel)

    Returns:
        occlusion_score : float dans [0, 1] (= ton ancien y_pred)
    \"\"\"
    # Resize unique a 512x512 pour unifier le pipeline (logique Julien)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)

    # === Etape 1 : detection visage + genre ===
    bbox, pred_gender = detect_face_and_gender(app, img)
    if bbox is None:
        return 0.0

    # === Etape 2 : mask theorique 3D avec correction Julien ===
    mask_theoretical, total_pixels = compute_theoretical_face_mask(tddfa, img, bbox)
    if total_pixels == 0:
        return 0.0

    # === Etape 3 : segmentations BiSeNet + SegFormer ===
    parsing_b = segment_with_bisenet(net, to_tensor, img, device=device)
    parsing_s = segment_with_segformer(seg_model, seg_processor, img, device=device)

    # === Etape 4 : ratios hair/hat/other pour chaque modele ===
    hair_ratio_b, hat_ratio_b, other_ratio_b = compute_ratios_from_parsing(
        parsing_b, mask_theoretical, total_pixels,
        VISIBLE_FACE_CLASSES_B, HAIR_B, HAT_B,
    )
    hair_ratio_s, hat_ratio_s, other_ratio_s = compute_ratios_from_parsing(
        parsing_s, mask_theoretical, total_pixels,
        VISIBLE_FACE_CLASSES_S, HAIR_S, HAT_S,
    )

    # === Etape 5 : features de detection plante (pour le fallback v2) ===
    skin_only_b_ratio, bg_b_ratio, skin_only_s_ratio = compute_plante_features(
        parsing_b, parsing_s, mask_theoretical, total_pixels,
    )

    # === Etape 6 : application du v2 fallback + dampening ===
    occlusion_score = apply_v2_fallback_with_dampening(
        pred_gender=pred_gender,
        hair_ratio_b=hair_ratio_b, hat_ratio_b=hat_ratio_b, other_ratio_b=other_ratio_b,
        hair_ratio_s=hair_ratio_s, hat_ratio_s=hat_ratio_s, other_ratio_s=other_ratio_s,
        skin_only_b_ratio=skin_only_b_ratio,
        bg_b_ratio=bg_b_ratio,
        skin_only_s_ratio=skin_only_s_ratio,
        M_WEIGHTS=M_WEIGHTS,
        F_WEIGHTS=F_WEIGHTS,
    )

    # === Visualisation optionnelle (debug) ===
    if display_results:
        visible_skin_mask_b = np.isin(parsing_b, VISIBLE_FACE_CLASSES_B).astype(np.uint8)
        visible_skin_mask_s = np.isin(parsing_s, VISIBLE_FACE_CLASSES_S).astype(np.uint8)
        vis1 = overlay_mask(img, mask_theoretical, color=(0, 255, 0), alpha=0.35)
        vis2 = overlay_mask(vis1, visible_skin_mask_b, color=(255, 0, 0), alpha=0.35)
        vis3 = overlay_mask(vis2, visible_skin_mask_s, color=(255, 0, 0), alpha=0.35)
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(vis3, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Pred occlusion = {occlusion_score:.3f}  |  gender = {pred_gender}")
        plt.show()

    return occlusion_score
"""


def _make_code_cell(source: str) -> dict:
    """Cree une cellule code Jupyter au format JSON."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main():
    # On part TOUJOURS du notebook original de Julien pour assurer une regeneration propre
    if NB_SOURCE.exists():
        shutil.copy(NB_SOURCE, NB_PATH)
        print(f"Notebook regenere depuis l'original : {NB_SOURCE.name}")
    else:
        print(f"WARN : notebook source introuvable, on patch le fichier existant : {NB_PATH.name}")

    nb = json.load(open(NB_PATH, encoding="utf-8"))
    n_initial = len(nb["cells"])
    print(f"Notebook charge : {n_initial} cellules")

    # ===== ETAPE 1 : ajouter SKIN_ONLY_B/S, BG_B dans la cell 35 (constantes) =====
    cell_35 = nb["cells"][35]
    src_35 = "".join(cell_35["source"])
    if "SKIN_ONLY_B" not in src_35:
        marker = "HAT_S = [14]"
        idx = src_35.find(marker)
        if idx == -1:
            print("ERREUR : marker pour Etape 1 introuvable dans cell 35")
            return
        end = idx + len(marker)
        new_src = src_35[:end] + "\n" + ETAPE_1_INSERT + src_35[end:]
        cell_35["source"] = new_src.splitlines(keepends=True)
        print("OK Etape 1 : constantes SKIN_ONLY_B/S et BG_B ajoutees a cell 35")
    else:
        print("SKIP Etape 1 : SKIN_ONLY_B deja present")

    # ===== ETAPE 2 : nouvelle cellule import du module v2 (avant occlusion_computation) =====
    # On cherche l'index de la cellule monolithique occlusion_computation actuelle
    occl_idx_orig = None
    for i, c in enumerate(nb["cells"]):
        if "def occlusion_computation" in "".join(c.get("source", "")):
            occl_idx_orig = i
            break
    if occl_idx_orig is None:
        print("ERREUR : cellule occlusion_computation introuvable")
        return

    # Insere la cellule d'import juste avant la cellule occlusion_computation
    already_imported = any(
        "from pipeline_julien_integration import apply_v2_fallback" in "".join(c.get("source", ""))
        for c in nb["cells"]
    )
    if not already_imported:
        nb["cells"].insert(occl_idx_orig, _make_code_cell(ETAPE_2_IMPORT))
        occl_idx_orig += 1  # decalage du index suite a l'insertion
        print(f"OK Etape 2 : cellule import inseree, occlusion_computation est maintenant en position {occl_idx_orig}")
    else:
        print("SKIP Etape 2 : import deja present")

    # ===== ETAPE 3 : SUPPRIMER la cellule monolithique occlusion_computation =====
    if "def detect_face_and_gender" not in "".join(nb["cells"][occl_idx_orig].get("source", "")):
        print(f"OK Etape 3 : suppression de la cellule monolithique occlusion_computation (position {occl_idx_orig})")
        del nb["cells"][occl_idx_orig]
        insert_at = occl_idx_orig  # on va inserer les 7 nouvelles cellules ici
    else:
        print("SKIP Etape 3 : refactor deja applique")
        insert_at = None

    # ===== ETAPE 4 : INSERER les 7 cellules avec sous-fonctions + orchestrateur =====
    if insert_at is not None:
        new_cells = [
            _make_code_cell(CELL_DETECT_FACE),
            _make_code_cell(CELL_COMPUTE_MASK),
            _make_code_cell(CELL_SEGMENT_BISENET),
            _make_code_cell(CELL_SEGMENT_SEGFORMER),
            _make_code_cell(CELL_COMPUTE_RATIOS),
            _make_code_cell(CELL_COMPUTE_PLANTE),
            _make_code_cell(CELL_ORCHESTRATOR),
        ]
        for i, cell in enumerate(new_cells):
            nb["cells"].insert(insert_at + i, cell)
        print(f"OK Etape 4 : {len(new_cells)} cellules sous-fonctions inserees a partir de position {insert_at}")
    else:
        print("SKIP Etape 4 : sous-fonctions deja presentes")

    # Sauvegarde
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    n_final = len(nb["cells"])
    print()
    print(f"Notebook sauvegarde : {NB_PATH}")
    print(f"Total cellules : {n_initial} -> {n_final} (delta {n_final - n_initial:+d})")


if __name__ == "__main__":
    main()
