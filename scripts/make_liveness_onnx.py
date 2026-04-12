#!/usr/bin/env python3
"""
Convertit les modèles Silent-Face-Anti-Spoofing (.pth) en ONNX pour le Raspberry Pi.

Usage
-----
1. Cloner le dépôt Silent-Face :
   git clone --depth 1 https://github.com/minivision-ai/Silent-Face-Anti-Spoofing /tmp/sfa

2. Installer PyTorch (CPU suffit, uniquement pour la conversion) :
   pip install torch --index-url https://download.pytorch.org/whl/cpu

3. Lancer la conversion :
   python scripts/make_liveness_onnx.py --sfa-path /tmp/sfa --output-dir models/liveness

Les fichiers .onnx générés dans models/liveness/ seront automatiquement
déployés sur le Raspberry Pi par le workflow GitHub Actions.
"""
from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Silent-Face .pth models to ONNX")
    p.add_argument(
        "--sfa-path",
        default=os.environ.get("SFA_PATH", "/tmp/sfa"),
        help="Chemin vers le dépôt Silent-Face-Anti-Spoofing cloné",
    )
    p.add_argument(
        "--output-dir",
        default="models/liveness",
        help="Répertoire de sortie pour les fichiers .onnx",
    )
    return p.parse_args()


def convert(sfa_path: str, output_dir: str) -> None:
    import torch  # noqa: PLC0415

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, sfa_path)

    # Importer les utilitaires du dépôt Silent-Face.
    try:
        utility = importlib.import_module("src.utility")
        parse_model_name = utility.parse_model_name
    except (ImportError, AttributeError) as exc:
        print(f"❌ Impossible d'importer src.utility depuis {sfa_path}: {exc}")
        print("   Vérifiez que --sfa-path pointe vers le dépôt cloné.")
        sys.exit(1)

    models_dir = os.path.join(sfa_path, "resources", "anti_spoof_models")
    if not os.path.isdir(models_dir):
        print(f"❌ Répertoire de modèles introuvable: {models_dir}")
        sys.exit(1)

    pth_files = sorted(
        f for f in os.listdir(models_dir)
        if f.endswith(".pth") and "80x80" in f
    )
    if not pth_files:
        print(f"❌ Aucun fichier .pth (80x80) trouvé dans {models_dir}")
        sys.exit(1)

    success = 0
    for pth_file in pth_files:
        pth_path = os.path.join(models_dir, pth_file)
        name_no_ext = os.path.splitext(pth_file)[0]
        onnx_path = out / (name_no_ext + ".onnx")

        print(f"\n▶ {pth_file}")

        # Extraire l'architecture et la taille d'entrée depuis le nom de fichier.
        try:
            h_in, w_in, model_type, _scale = parse_model_name(name_no_ext)
        except Exception as exc:
            print(f"  [SKIP] parse_model_name a échoué: {exc}")
            continue

        # Importer la classe d'architecture.
        try:
            minifasnet = importlib.import_module("src.model_lib.MiniFASNet")
            model_cls = getattr(minifasnet, model_type)
        except (ImportError, AttributeError) as exc:
            print(f"  [SKIP] Classe '{model_type}' introuvable: {exc}")
            continue

        # Tenter le chargement avec num_classes=3 puis 2.
        model = None
        for num_classes in (3, 2):
            try:
                candidate = model_cls(num_classes=num_classes, img_channel=3)
                state = torch.load(pth_path, map_location="cpu")
                candidate.load_state_dict(state)
                candidate.eval()
                model = candidate
                print(f"  ✅ Chargé avec num_classes={num_classes}")
                break
            except RuntimeError:
                continue

        if model is None:
            print(f"  [SKIP] Impossible de charger {pth_file}")
            continue

        # Export ONNX.
        dummy = torch.zeros(1, 3, int(h_in), int(w_in))
        try:
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=11,
                do_constant_folding=True,
            )
            print(f"  ✅ Exporté: {onnx_path}")
            success += 1
        except Exception as exc:
            print(f"  [ERROR] Export ONNX échoué: {exc}")

    print(f"\n{'✅' if success else '❌'} {success}/{len(pth_files)} modèle(s) converti(s) → {output_dir}")
    if success == 0:
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    convert(args.sfa_path, args.output_dir)
