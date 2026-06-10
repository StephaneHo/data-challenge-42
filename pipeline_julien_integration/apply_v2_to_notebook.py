"""Modifie le notebook .ipynb pour integrer le v2 fallback (3 etapes du README).

Usage:
    python apply_v2_to_notebook.py
"""
import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).parent / "DataChallengeJulien_TF_with_v2_fallback.ipynb"

# Code a INSERER ou REMPLACER
NEW_IMPORT_CELL = """# === [Etape 2 du README pipeline_julien_integration] ===
# Setup path pour acceder au module pipeline_julien_integration
import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))  # 2 niveaux car notebook deplace dans pipeline_julien_integration/
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline_julien_integration import apply_v2_fallback, detect_plante_regime
print("v2 fallback charge OK :", apply_v2_fallback)
"""

# Texte a ajouter dans cell 35 (apres la definition de HAT_S = [14])
ETAPE_1_INSERT = """
# === [Etape 1 du README pipeline_julien_integration] ===
# Pour la detection plante du fallback v2.
# NB : SKIN_ONLY_B/S incluent les ears (7,8 BiSeNet ; 8,9 SegFormer) alors que
# VISIBLE_FACE_CLASSES_B/S de Julien ne les incluent pas. C'est volontaire :
# les seuils 0.70 et 0.30 ont ete calibres en CV sur cette definition AVEC ears.
SKIN_ONLY_B = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]
SKIN_ONLY_S = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
BG_B        = [0]
"""

# Texte a inserer dans cell 37 apres "other_ratio_s = other_s_in_mask / total_pixels"
ETAPE_3_INSERT = """
    ### [Etape 3 du README] Detection plante v2 (3 nouvelles features)
    skin_only_b = np.isin(parsing_b, SKIN_ONLY_B).astype(np.uint8)
    bg_b        = np.isin(parsing_b, BG_B).astype(np.uint8)
    skin_only_s = np.isin(parsing_s, SKIN_ONLY_S).astype(np.uint8)
    skin_only_b_ratio = float(np.sum(skin_only_b & mask_theoretical)) / float(total_pixels)
    bg_b_ratio        = float(np.sum(bg_b        & mask_theoretical)) / float(total_pixels)
    skin_only_s_ratio = float(np.sum(skin_only_s & mask_theoretical)) / float(total_pixels)
"""

# Bloc a REMPLACER dans cell 37 (le if pred_gender == 'M' / elif 'F')
OLD_GENDER_BLOCK_START = "if pred_gender == 'M':"

ETAPE_4_REPLACE = """    ### [Etape 4 du README] Appel du v2 fallback (remplace le bloc if/elif gender)
    occlusion_score = apply_v2_fallback(
        pred_gender=pred_gender,
        hair_ratio_b=hair_ratio_b, hat_ratio_b=hat_ratio_b, other_ratio_b=other_ratio_b,
        hair_ratio_s=hair_ratio_s, hat_ratio_s=hat_ratio_s, other_ratio_s=other_ratio_s,
        skin_only_b_ratio=skin_only_b_ratio,
        bg_b_ratio=bg_b_ratio,
        skin_only_s_ratio=skin_only_s_ratio,
        M_WEIGHTS=M_WEIGHTS,
        F_WEIGHTS=F_WEIGHTS,
    )
"""


def main():
    nb = json.load(open(NB_PATH, encoding="utf-8"))
    print(f"Notebook charge : {len(nb['cells'])} cellules")

    # === Etape 1 : ajouter SKIN_ONLY_B/S, BG_B dans la cell 35 ===
    cell_35 = nb["cells"][35]
    src_35 = "".join(cell_35["source"])
    if "SKIN_ONLY_B" not in src_35:
        # On insere apres "HAT_S = [14]"
        marker = "HAT_S = [14]"
        idx = src_35.find(marker)
        if idx == -1:
            print("ERREUR : marker pour Etape 1 introuvable dans cell 35")
            return
        end = idx + len(marker)
        new_src = src_35[:end] + "\n" + ETAPE_1_INSERT + src_35[end:]
        cell_35["source"] = new_src.splitlines(keepends=True)
        print("OK Etape 1 : SKIN_ONLY_B/S et BG_B ajoutes a cell 35")
    else:
        print("SKIP Etape 1 : SKIN_ONLY_B deja present")

    # === Etape 2 : nouvelle cellule import avant cell 37 ===
    already_imported = any(
        "from pipeline_julien_integration import apply_v2_fallback" in "".join(c.get("source", ""))
        for c in nb["cells"]
    )
    if not already_imported:
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": NEW_IMPORT_CELL.splitlines(keepends=True),
        }
        # On l'insere juste avant cell 37 (occlusion_computation)
        nb["cells"].insert(37, new_cell)
        print(f"OK Etape 2 : import insere a la position 37 (nb cells maintenant : {len(nb['cells'])})")
    else:
        print("SKIP Etape 2 : import deja present")

    # On recupere l'index de la cell occlusion_computation (qui a maintenant decale d'1 si on a insere)
    occl_idx = None
    for i, c in enumerate(nb["cells"]):
        s = "".join(c.get("source", ""))
        if "def occlusion_computation" in s:
            occl_idx = i
            break
    if occl_idx is None:
        print("ERREUR : cell occlusion_computation introuvable")
        return

    # === Etape 3 : ajouter les 3 nouvelles features dans occlusion_computation ===
    cell_occl = nb["cells"][occl_idx]
    src_occl = "".join(cell_occl["source"])
    if "skin_only_b_ratio" not in src_occl:
        # On insere apres "other_ratio_s = other_s_in_mask / total_pixels"
        marker = "other_ratio_s = other_s_in_mask / total_pixels"
        idx = src_occl.find(marker)
        if idx == -1:
            print("ERREUR : marker Etape 3 introuvable")
            return
        end = idx + len(marker)
        src_occl = src_occl[:end] + ETAPE_3_INSERT + src_occl[end:]
        print(f"OK Etape 3 : 3 nouvelles features ajoutees a occlusion_computation (cell {occl_idx})")
    else:
        print("SKIP Etape 3 : skin_only_b_ratio deja present")

    # === Etape 4 : remplacer le bloc if pred_gender == 'M' / elif 'F' ===
    if "apply_v2_fallback(" not in src_occl:
        # On veut remplacer depuis "if pred_gender == 'M':" jusqu'a la fin du bloc elif F
        start_idx = src_occl.find("if pred_gender == 'M':")
        if start_idx == -1:
            print("ERREUR : marker 'if pred_gender' introuvable")
            return
        # La fin du bloc : la derniere ligne du elif 'F', qui se termine par "0,1)"
        # On cherche le dernier "0,1)" apres le start_idx
        elif_idx = src_occl.find("elif pred_gender == 'F':", start_idx)
        if elif_idx == -1:
            print("ERREUR : marker 'elif F' introuvable")
            return
        # On cherche le clip final de l'elif
        end_marker = "0,1)"
        last_end = src_occl.find(end_marker, elif_idx)
        if last_end == -1:
            print("ERREUR : fin du bloc gender introuvable")
            return
        end_idx = last_end + len(end_marker)
        # On detecte aussi le commentaire/lignes apres pour ne pas le supprimer
        # On verifie que la ligne suivante est blanche
        print(f"OK Etape 4 : suppression du bloc gender ({start_idx} -> {end_idx})")
        # Calculer l'indentation initiale (4 espaces normalement)
        # On va supprimer aussi les espaces au debut de "if pred_gender" qui peuvent etre 4 espaces
        # Reculer si ya des espaces avant "if pred_gender"
        actual_start = start_idx
        while actual_start > 0 and src_occl[actual_start - 1] == ' ':
            actual_start -= 1
        src_occl = src_occl[:actual_start] + ETAPE_4_REPLACE + src_occl[end_idx:]
        print("OK Etape 4 : bloc gender remplace par apply_v2_fallback()")
    else:
        print("SKIP Etape 4 : apply_v2_fallback deja present")

    cell_occl["source"] = src_occl.splitlines(keepends=True)

    # Save
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"\nNotebook sauvegarde : {NB_PATH}")
    print(f"Total cellules : {len(nb['cells'])}")


if __name__ == "__main__":
    main()
